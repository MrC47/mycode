# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from domainbed.lib import wide_resnet
import copy

import timm


def _build_torchvision_model(model_name, weights_name):
    """Build torchvision model with the modern weights API and a safe fallback."""
    builder = getattr(torchvision.models, model_name)
    weights_enum = getattr(torchvision.models, weights_name, None)
    if weights_enum is not None:
        return builder(weights=weights_enum.DEFAULT)
    return builder(pretrained=True)


def _resolve_backbone_name(hparams):
    """
    Resolve backbone from `hparams["backbone"]`.
    Accepted names are case-insensitive:
    ResNet, ViT, AlexNet, EfficientNet, DINOv2.
    If `backbone` is missing, default to ResNet.
    """
    backbone = hparams.get("backbone", None)
    if backbone is None:
        return "resnet"
    if not isinstance(backbone, str) or not backbone.strip():
        raise ValueError("Invalid `backbone`: expected a non-empty string.")

    normalized = backbone.strip().lower()
    mapping = {
        "resnet": "resnet",
        "vit": "vit",
        "alexnet": "alexnet",
        "efficientnet": "efficientnet",
        "dinov2": "dinov2"
    }
    if normalized not in mapping:
        raise ValueError(
            f"Invalid backbone '{backbone}': choose from ResNet, ViT, AlexNet, EfficientNet, DINOv2"
        )
    return mapping[normalized]


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
            self.network = _build_torchvision_model("resnet18", "ResNet18_Weights")
            self.n_outputs = 512
        else:
            self.network = _build_torchvision_model("resnet50", "ResNet50_Weights")
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
            self.network.fc = nn.Identity()

        if hparams["freeze_bn"]:
            self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])
        self.activation = nn.Identity() # for URM; does not affect other algorithms

    def forward(self, x):
        if self.hparams['resnet50_augmix']:
            x = self.network(x)[-1]
        else:
            x = self.network(x)
        # x.shape = (B, C, H, W)
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
        backbone = _resolve_backbone_name(hparams)
        if backbone == "dinov2":
            return DinoV2(input_shape, hparams)
        if backbone == "vit":
            return ViT(input_shape, hparams)
        if backbone == "alexnet":
            return AlexNet(input_shape, hparams)
        if backbone == "efficientnet":
            return EfficientNet(input_shape, hparams)
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError


def Decoder(feature_dim, input_shape, hparams):
    """Auto-select an appropriate decoder for the given input shape."""
    if len(input_shape) == 3:
        backbone = _resolve_backbone_name(hparams)
        if backbone == "vit":
            return ViTDecoder(feature_dim, input_shape, hparams)
        elif backbone == "alexnet":
            return AlexNetDecoder(feature_dim, input_shape, hparams)
        elif backbone == "efficientnet":
            return EfficientNetDecoder(feature_dim, input_shape, hparams)
        elif backbone == "dinov2":
            raise ValueError("Decoder for DINOv2 is not implemented.")
        else:
            return ResNet50Decoder(feature_dim, input_shape, hparams)



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

class PrivateHead(nn.Module):
    def __init__(self, input_dim, hparams):
        super(PrivateHead, self).__init__()
        dropout = hparams.get('mlp_dropout', 0.1)
        mid_dim = max(input_dim // 4, 16)
        vec_dim = max(input_dim // 2, 64)

        # Learnable residual scale keeps the adapter stable at initialization.
        self.res_scale = nn.Parameter(torch.tensor(1.0))

        # Spatial adapter for CNN feature maps: (B, C, H, W)
        self.spatial_adapter = nn.Sequential(
            nn.Conv2d(input_dim, mid_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(mid_dim, input_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(input_dim)
        )

        # Token adapter for sequence features: (B, N, C)
        self.token_adapter = nn.Sequential(
            nn.Linear(input_dim, vec_dim, bias=False),
            nn.LayerNorm(vec_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(vec_dim, input_dim, bias=False),
            nn.LayerNorm(input_dim)
        )

        # Vector adapter for pooled features: (B, C)
        self.vector_adapter = nn.Sequential(
            nn.Linear(input_dim, vec_dim, bias=False),
            nn.LayerNorm(vec_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(vec_dim, input_dim, bias=False),
            nn.LayerNorm(input_dim)
        )

        # Lightweight CLS <-> patch interaction for ViT-like tokens.
        self.patch_to_cls = nn.Linear(input_dim, input_dim, bias=False)
        self.cls_to_patch = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, x):
        # CNN feature map
        if x.dim() == 4:
            delta = self.spatial_adapter(x)
            return x + self.res_scale * delta

        # Token feature
        if x.dim() == 3:
            delta = self.token_adapter(x)

            n = x.size(1)
            side = int(math.sqrt(max(n - 1, 1)))
            has_cls = n > 1 and side * side == (n - 1)
            if has_cls:
                cls = delta[:, :1, :]
                patches = delta[:, 1:, :]
                patch_mean = patches.mean(dim=1, keepdim=True)
                cls = cls + self.patch_to_cls(patch_mean)
                patches = patches + self.cls_to_patch(cls).expand_as(patches)
                delta = torch.cat([cls, patches], dim=1)

            return x + self.res_scale * delta

        # Vector feature
        if x.dim() == 2:
            delta = self.vector_adapter(x)
            return x + self.res_scale * delta

        raise ValueError(f"PrivateHead expects 2D, 3D or 4D input, got {x.dim()}D.")

class ViT(nn.Module):
    def __init__(self, input_shape, hparams):
        super(ViT, self).__init__()
        nc = input_shape[0]
        if nc != 3:
            raise RuntimeError("ViT inputs must have 3 channels")

        # Standard ViT-B/16 encoder with classifier head removed.
        self.network = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.n_outputs = 768

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams.get('vit_dropout', 0.))
        self.activation = nn.Identity()

        with torch.no_grad():
            dummy = torch.zeros(1, nc, input_shape[1], input_shape[2])
            self.num_tokens = self._forward_tokens(dummy).shape[1]

    def _forward_tokens(self, x):
        x = self.network.forward_features(x)
        if isinstance(x, tuple):
            x = x[0]
        if isinstance(x, dict):
            # timm may return dicts in some versions/configs.
            if "x" in x:
                x = x["x"]
            elif "x_norm_patchtokens" in x:
                patches = x["x_norm_patchtokens"]
                cls = x.get("x_norm_clstoken", None)
                x = torch.cat([cls.unsqueeze(1), patches], dim=1) if cls is not None else patches
            else:
                raise RuntimeError("Unsupported ViT feature dict format from timm.")
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x

    def forward(self, x):
        x = self._forward_tokens(x)
        return self.activation(self.dropout(x))

class EfficientNet(nn.Module):
    def __init__(self, input_shape, hparams):
        super(EfficientNet, self).__init__()
        nc = input_shape[0]

        self.network = _build_torchvision_model("efficientnet_b0", "EfficientNet_B0_Weights")
        self.n_outputs = 1280

        # Adapt first conv when input channels != 3.
        if nc != 3:
            old_conv = self.network.features[0][0]
            tmp = old_conv.weight.data.clone()
            new_conv = nn.Conv2d(
                nc,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            for i in range(nc):
                new_conv.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]
            self.network.features[0][0] = new_conv

        # Remove classifier head and keep encoder features only.
        self.network.classifier = Identity()

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams.get('resnet_dropout', 0.))
        self.activation = nn.Identity()

        with torch.no_grad():
            dummy_input = torch.zeros(1, nc, input_shape[1], input_shape[2])
            dummy_output = self.network.features(dummy_input)
            _, _, self.spatial_h, self.spatial_w = dummy_output.shape
            self.num_tokens = self.spatial_h * self.spatial_w

    def forward(self, x):
        x = self.network.features(x)
        b, c, _, _ = x.shape
        x = x.reshape(b, c, -1).transpose(1, 2)
        return self.activation(self.dropout(x))

class AlexNet(nn.Module):
    def __init__(self, input_shape, hparams):
        super(AlexNet, self).__init__()
        nc = input_shape[0]

        self.network = _build_torchvision_model("alexnet", "AlexNet_Weights")
        self.n_outputs = 256

        # Adapt first conv when input channels != 3.
        if nc != 3:
            old_conv = self.network.features[0]
            tmp = old_conv.weight.data.clone()
            new_conv = nn.Conv2d(
                nc,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            for i in range(nc):
                new_conv.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]
            self.network.features[0] = new_conv

        # Remove classifier and keep spatial encoder output.
        self.network.classifier = Identity()

        self.dropout = nn.Dropout(hparams.get('resnet_dropout', 0.))
        self.activation = nn.Identity()

        with torch.no_grad():
            dummy = torch.zeros(1, nc, input_shape[1], input_shape[2])
            feat = self.network.avgpool(self.network.features(dummy))
            self.spatial_h, self.spatial_w = feat.shape[2:]
            self.num_tokens = self.spatial_h * self.spatial_w

    def forward(self, x):
        x = self.network.features(x)
        x = self.network.avgpool(x)
        b, c, _, _ = x.shape
        x = x.reshape(b, c, -1).transpose(1, 2)
        return self.activation(self.dropout(x))

class ReverseBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ReverseBottleneck, self).__init__()
        mid_channels = out_channels // 4
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=False) if stride > 1 else nn.Identity()
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride > 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.upsample(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return F.relu(out)

class ResNet50Decoder(nn.Module):
    def __init__(self, feature_dim, input_shape, hparams):
        super(ResNet50Decoder, self).__init__()
        # input_shape: (3, 224, 224)
        self.input_shape = input_shape
        
        # 鏍稿績鏀瑰姩锛氫娇鐢?1x1 鍗风Н灏嗘嫾鎺ュ悗鐨勯€氶亾鏁?(濡?4096) 闄嶇淮鍒?2048
        # 杩欐牱鏃犺 feature_dim 鏄?2048 杩樻槸 4096锛岄兘鑳藉榻愬埌鍚庣画灞?
        self.prep_conv = nn.Conv2d(feature_dim, 2048, kernel_size=1)

        # 閫嗗悜璺緞淇濇寔涓嶅彉
        self.layer4 = ReverseBottleneck(2048, 1024, stride=2) 
        self.layer3 = ReverseBottleneck(1024, 512, stride=2)  
        self.layer2 = ReverseBottleneck(512, 256, stride=2)   
        self.layer1 = ReverseBottleneck(256, 64, stride=2)    
        
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, input_shape[0], kernel_size=3, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        if x.dim() == 2:
            # ResNet without features_only may output pooled vectors (B, C).
            x = x.unsqueeze(-1).unsqueeze(-1)
        elif x.dim() == 3:
            # Token-like input fallback.
            x = _tokens_to_feature_map(x)
        elif x.dim() != 4:
            raise ValueError(f"ResNet50Decoder expects 2D/3D/4D input, got {x.dim()}D")

        x = self.prep_conv(x)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.final(x)
        if x.shape[2:] != self.input_shape[1:]:
            x = F.interpolate(x, size=self.input_shape[1:], mode='bilinear', align_corners=False)
        return x


def _tokens_to_feature_map(x, expected_h=None, expected_w=None):
    """
    Convert feature tokens (B, N, C) into feature map (B, C, H, W).
    Also supports a legacy case where tokens were concatenated along N:
    N = 2 * H * W -> fold into channels as (B, 2C, H, W).
    """
    if x.dim() == 4:
        return x
    if x.dim() != 3:
        raise ValueError(f"Expected 3D/4D features, got {x.dim()}D.")

    b, n, c = x.shape
    if expected_h is not None and expected_w is not None:
        expected_n = expected_h * expected_w
        if n == expected_n:
            return x.transpose(1, 2).reshape(b, c, expected_h, expected_w)
        if n == 2 * expected_n:
            x = x.reshape(b, 2, expected_n, c).permute(0, 2, 1, 3).reshape(b, expected_n, 2 * c)
            return x.transpose(1, 2).reshape(b, 2 * c, expected_h, expected_w)

    side = int(math.sqrt(n))
    if side * side == n:
        return x.transpose(1, 2).reshape(b, c, side, side)

    # Fallback for legacy fused tokens: N = 2 * (H * W)
    if n % 2 == 0:
        half = n // 2
        side = int(math.sqrt(half))
        if side * side == half:
            x = x.reshape(b, 2, half, c).permute(0, 2, 1, 3).reshape(b, half, 2 * c)
            return x.transpose(1, 2).reshape(b, 2 * c, side, side)

    raise ValueError(f"Cannot infer spatial size from token count N={n}.")


class EfficientNetDecoder(nn.Module):
    def __init__(self, feature_dim, input_shape, hparams, stride=32):
        super(EfficientNetDecoder, self).__init__()
        self.input_shape = input_shape
        self.spatial_h = input_shape[1] // stride
        self.spatial_w = input_shape[2] // stride
        self.channel_adjust = nn.Conv2d(feature_dim, 1280, kernel_size=1) if feature_dim != 1280 else nn.Identity()

        self.up1 = nn.ConvTranspose2d(1280, 512, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.up5 = nn.ConvTranspose2d(64, input_shape[0], kernel_size=4, stride=2, padding=1)

        # Skip connections from bottleneck to each decoder stage.
        self.skip1 = nn.Conv2d(1280, 512, kernel_size=1, bias=False)
        self.skip2 = nn.Conv2d(1280, 256, kernel_size=1, bias=False)
        self.skip3 = nn.Conv2d(1280, 128, kernel_size=1, bias=False)
        self.skip4 = nn.Conv2d(1280, 64, kernel_size=1, bias=False)

    def _fuse_skip(self, x, bottleneck, proj):
        skip = F.interpolate(bottleneck, size=x.shape[2:], mode='bilinear', align_corners=False)
        skip = proj(skip)
        return F.relu(x + skip, inplace=True)

    def forward(self, x):
        x = _tokens_to_feature_map(x, self.spatial_h, self.spatial_w)
        x = self.channel_adjust(x)  # (B, 1280, 7, 7)
        bottleneck = x

        x = self.bn1(self.up1(x))
        x = self._fuse_skip(x, bottleneck, self.skip1)
        x = self.bn2(self.up2(x))
        x = self._fuse_skip(x, bottleneck, self.skip2)
        x = self.bn3(self.up3(x))
        x = self._fuse_skip(x, bottleneck, self.skip3)
        x = self.bn4(self.up4(x))
        x = self._fuse_skip(x, bottleneck, self.skip4)
        x = torch.sigmoid(self.up5(x))

        if x.shape[2:] != self.input_shape[1:]:
            x = F.interpolate(x, size=self.input_shape[1:], mode='bilinear', align_corners=False)
        return x


class AlexNetDecoder(nn.Module):
    def __init__(self, feature_dim, input_shape, hparams):
        super(AlexNetDecoder, self).__init__()
        self.input_shape = input_shape
        self.channel_adjust = nn.Conv2d(feature_dim, 256, kernel_size=1) if feature_dim != 256 else nn.Identity()

        self.upsample_blocks = nn.ModuleList([
            self._make_up_block(256, 128),
            self._make_up_block(128, 64),
            self._make_up_block(64, 32),
            self._make_up_block(32, 16),
            self._make_up_block(16, 16),
        ])

        self.final_conv = nn.Sequential(
            nn.Conv2d(16, input_shape[0], kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def _make_up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = _tokens_to_feature_map(x)
        x = self.channel_adjust(x)

        for block in self.upsample_blocks:
            x = block(x)

        x = self.final_conv(x)
        if x.shape[2:] != self.input_shape[1:]:
            x = F.interpolate(x, size=self.input_shape[1:], mode='bilinear', align_corners=False)
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
            activation='gelu',
            batch_first=True
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
