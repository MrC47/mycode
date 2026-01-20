# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from domainbed.lib import wide_resnet
import copy

import timm


def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs
        self.activation = nn.Identity() # for URM; does not affect other algorithms

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        x = self.activation(x) # for URM; does not affect other algorithms
        return x

class DinoV2(torch.nn.Module):
    """ """
    def __init__(self,input_shape, hparams):
        super(DinoV2, self).__init__()

        self.network = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.n_outputs =  5 * 768

        nc = input_shape[0]

        if nc != 3:
            raise RuntimeError("Inputs must have 3 channels")

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['vit_dropout'])

        if hparams["vit_attn_tune"]:
            for n,p in self.network.named_parameters():
                if 'attn' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False


    def forward(self, x):
        x = self.network.get_intermediate_layers(x, n=4, return_class_token=True)
        linear_input = torch.cat([
            x[0][1],
            x[1][1],
            x[2][1],
            x[3][1],
            x[3][0].mean(1)
            ], dim=1)
        return self.dropout(linear_input)


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048

        if hparams['resnet50_augmix']:
            self.network = timm.create_model('resnet50.ram_in1k', pretrained=True, features_only=True)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            conv1  = getattr(self.network, 'conv1', None)
            if conv1 is not None:
                tmp = self.network.conv1.weight.data.clone()

                self.network.conv1 = nn.Conv2d(
                    nc, 64, kernel_size=(7, 7),
                    stride=(2, 2), padding=(3, 3), bias=False)

                for i in range(nc):
                    self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        if hasattr(self.network, 'fc'):
            del self.network.fc
            self.network.fc = Identity()

        if hparams["freeze_bn"]:
            self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])
        self.activation = nn.Identity() # for URM; does not affect other algorithms

    def forward(self, x):
        x = self.network(x)[-1]
        # 1. flatten(2): (B, C, H*W) -> flatten 7x7 into 49
        # 2. transpose(1, 2): (B, 49, C) -> convert to token squence format 
        x = x.flatten(2).transpose(1, 2)
        return self.activation(self.dropout(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.hparams["freeze_bn"]:
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.activation = nn.Identity() # for URM; does not affect other algorithms

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return self.activation(x)


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        if hparams["vit"]:
            if hparams["dinov2"]:
                return DinoV2(input_shape, hparams)
            else:
                raise NotImplementedError
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)


class ViT(torch.nn.Module):
    def __init__(self, input_shape, hparams):
        super(ViT, self).__init__()
        # 在 PyTorch 的神经网络模型中，input_shape 通常不包含 Batch Size。它指的是单张图像的维度
        # Batch Size 只会在 forward(self, x) 函数被调用时，伴随输入张量 x 出现。在那时，x 的形状才是 (BatchSize, Channels, Height, Width)。
        nc = input_shape[0]
        if nc != 3:
            raise RuntimeError("ViT inputs must have 3 channels")
        
        # num_classes=0: 去掉最后的分类层，只保留特征提取部分
        self.network = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.n_outputs = 768
        self.num_tokens = 197
        
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams.get('vit_dropout', 0.))
        # 恒等映射，不改变输入，预留给需要激活函数的特定算法。
        self.activation = nn.Identity()
        
    def forward(self, x):
        # 输出形状通常为 (BatchSize, 197, 768)
        x = self.network.forward_features(x)
        # 某些 timm 版本可能会返回 (features, global_pool_feat) 的元组。这里确保我们只拿到特征张量。
        if isinstance(x, tuple):
            x = x[0]
        # 正常的 ViT 输出 (BatchSize, Sequence_Length, Embedding_Dim)
        if x.dim() == 3:
            return self.activation(self.dropout(x))
        # 如果模型意外进行了全局平均池化变成了 (Batch, Embedding_Dim)。我们手动增加一个维度变成 (Batch, 1, Embedding_Dim) 以适配 Cross-Attention 层
        elif x.dim() == 2:
            return self.activation(self.dropout(x.unsqueeze(1)))
        else:
            return self.activation(self.dropout(x))


class EfficientNet(torch.nn.Module):
    def __init__(self, input_shape, hparams):
        super(EfficientNet, self).__init__()
        nc = input_shape[0]
        
        self.network = torchvision.models.efficientnet_b0(pretrained=True)
        # the output feature dimension of efficientnet_b0 is 1280
        self.n_outputs = 1280
        
        # channel adaption logic
        if nc != 3:
            tmp = self.network.features[0][0].weight.data.clone()
            self.network.features[0][0] = nn.Conv2d(
                nc, 32, kernel_size=(3, 3),
                stride=(2, 2), padding=(1, 1), bias=False)
            for i in range(nc):
                self.network.features[0][0].weight.data[:, i, :, :] = tmp[:, i % 3, :, :]
        
        del self.network.classifier
        self.network.classifier = Identity()
        
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams.get('resnet_dropout', 0.))
        self.activation = nn.Identity()
        
        self.feature_proj = nn.Conv2d(1280, 1280, kernel_size=1)
        
        # compute number of tokens to match CrossAttention input requirements
        with torch.no_grad():
            dummy_input = torch.zeros(1, nc, input_shape[1], input_shape[2])
            dummy_output = self.network.features(dummy_input)
            _, _, self.spatial_h, self.spatial_w = dummy_output.shape
            self.num_tokens = self.spatial_h * self.spatial_w
        self.register_buffer('num_tokens', torch.tensor(self.num_tokens))
        
    def forward(self, x):
        x = self.network.features(x)
        B, C, H, W = x.shape
        
        x = self.feature_proj(x)
        x = x.view(B, C, H, W)
        # (B, 1280, H, W) -> (B, 1280, H*W)
        x = x.view(B, C, -1)
        # (B, 1280, 49) -> (B, 49, 1280)
        # transform (B, C, N) to (B, N, C)
        x = x.transpose(1, 2)
        x = self.activation(self.dropout(x))
        return x


class AlexNet(torch.nn.Module):
    def __init__(self, input_shape, hparams):
        super(AlexNet, self).__init__()
        nc = input_shape[0]
        
        self.network = torchvision.models.alexnet(pretrained=True)
        self.n_outputs = 4096
        
        if nc != 3:
            tmp = self.network.features[0].weight.data.clone()
            self.network.features[0] = nn.Conv2d(
                nc, 64, kernel_size=(11, 11),
                stride=(4, 4), padding=(2, 2))
            for i in range(nc):
                self.network.features[0].weight.data[:, i, :, :] = tmp[:, i % 3, :, :]
        
        del self.network.classifier
        self.network.classifier = Identity()
        
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams.get('resnet_dropout', 0.))
        self.activation = nn.Identity()
        
        self.feature_proj = nn.Conv2d(256, 4096, kernel_size=1)
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, nc, input_shape[1], input_shape[2])
            dummy_output = self.network.features(dummy_input)
            _, _, self.spatial_h, self.spatial_w = dummy_output.shape
            self.num_tokens = self.spatial_h * self.spatial_w
        self.register_buffer('num_tokens', torch.tensor(self.num_tokens))
        
    def forward(self, x):
        x = self.network.features(x)
        B, C, H, W = x.shape
        
        x = self.feature_proj(x)
        x = x.view(B, 4096, H, W)
        x = x.view(B, 4096, -1)
        x = x.transpose(1, 2)
        x = self.activation(self.dropout(x))
        return x


class ResNetDecoder(nn.Module):
    def __init__(self, feature_dim, input_shape, hparams):
        super(ResNetDecoder, self).__init__()
        self.input_shape = input_shape # (C, H, W)
        self.feature_dim = feature_dim # input feature dimension(from CrossAttention class)
        
        # ResNetDecoder base on ResNet50(2048 dimension)
        # use 1x1 convolution to project to 2048 channels if feature_dim != 2048
        self.channel_adjust = nn.Conv2d(feature_dim, 2048, kernel_size=1) if feature_dim != 2048 else nn.Identity

        # gradually upsample 7x7 features to original resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, input_shape[0], kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, N, C = x.shape
        # reshape x to 4-dimension tensor (B, C, H, W)
        x = x.transpose(1, 2).reshape(B, C, 7, 7)
        x = self.channel_adjust(x)
        x = self.decoder(x)
        if x.shape[2:] != self.input_shape[1:]:
            x = F.interpolate(x, size=self.input_shape[1:], mode='bilinear', align_corners=False)
        return x


class EfficientNetDecoder(nn.Module):
    def __init__(self, feature_dim, input_shape, hparams, spatial_h=7, spatial_w=7):
        super(EfficientNetDecoder, self).__init__()
        self.input_shape = input_shape # (3, 244, 244)
        self.spatial_h = spatial_h # feature graph height, default 7
        self.spatial_w = spatial_w # feature graph width, default 7

        self.channel_adjust = nn.Conv2d(feature_dim, 1280, kernel_size=1) if feature_dim != 1280 else nn.Identity()
        
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, input_shape[0], kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, N, C = x.shape
        # reshape (B, 49, 1280) to (B, 1280, 7, 7)
        x = x.transpose(1, 2).reshape(B, C, self.spatial_h, self.spatial_w)
        x = self.channel_adjust(x)
        x = self.decoder(x)
        if x.shape[2:] != self.input_shape[1:]:
            x = F.interpolate(x, size=self.input_shape[1:], mode='bilinear', align_corners=False)
        return x


class AlexNetDecoder(nn.Module):
    def __init__(self, feature_dim, input_shape, hparams, spatial_h=6, spatial_w=6):
        super(AlexNetDecoder, self).__init__()
        self.input_shape = input_shape
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w
        
        self.channel_adjust = nn.Conv2d(feature_dim, 256, kernel_size=1) if feature_dim != 256 else nn.Identity()
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, input_shape[0], kernel_size=14, stride=10, padding=2), 
            nn.Sigmoid()
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, self.spatial_h, self.spatial_w)
        x = self.channel_adjust(x)
        x = self.decoder(x)
        
        if x.shape[2:] != self.input_shape[1:]:
            x = F.interpolate(x, size=self.input_shape[1:], mode='bilinear')
        return x


class ViTDecoder(nn.Module):
    def __init__(self, feature_dim, input_shape, hparams):
        super(ViTDecoder, self).__init__()
        self.input_shape = input_shape # original image shape (C, H, W)
        self.feature_dim = feature_dim # feature dimension (E or C), typically 768
        
        patch_size = 16
        img_size = input_shape[1] # usually 224
        num_patches = (img_size // patch_size) ** 2 # the number of patches
        
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        # linear projection layer for initial transformation of input feature 
        self.decoder_embed = nn.Linear(feature_dim, feature_dim)
        # define position coding
        # +1 for CLS token position
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, feature_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            activation='gelu'
        )
        # stack 6 decoder_layer
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        self.decoder_norm = nn.LayerNorm(feature_dim)
        self.decoder_pred = nn.Linear(feature_dim, patch_size * patch_size * input_shape[0], bias=True)
        
        # initialize position encoding weights
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        
    def forward(self, x):
        B, N, C = x.shape
        x = self.decoder_embed(x) 
        if x.size(1) == self.decoder_pos_embed.size(1):
            x = x + self.decoder_pos_embed
        else:
            x = x + self.decoder_pos_embed[:, 1:, :]

        tgt = self.decoder_pos_embed[:, 1:, :].expand(B, -1, -1)
        x = self.decoder(tgt, x) 
        
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)         
        p = self.patch_size
        h = w = int(self.num_patches ** 0.5)
        x = x.reshape(B, h, w, p, p, self.input_shape[0])
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, self.input_shape[0], h * p, w * p)
        
        return x