# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

# --- Mamba2 依赖 ---
from einops import rearrange, repeat
from mamba_ssm.modules.mamba2 import RMSNormGated
from mamba_ssm.ops.triton.ssd_combined import (
    mamba_chunk_scan_combined,
    mamba_split_conv1d_scan_combined,
)

from domainbed.lib import wide_resnet

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


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
                        setattr(
                            bottleneck,
                            name2,
                            fuse(module2, getattr(bottleneck, bn_name)),
                        )
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(
                        bottleneck.downsample[0], bottleneck.downsample[1]
                    )
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
        self.input = nn.Linear(n_inputs, hparams["mlp_width"])
        self.dropout = nn.Dropout(hparams["mlp_dropout"])
        self.hiddens = nn.ModuleList(
            [
                nn.Linear(hparams["mlp_width"], hparams["mlp_width"])
                for _ in range(hparams["mlp_depth"] - 2)
            ]
        )
        self.output = nn.Linear(hparams["mlp_width"], n_outputs)
        self.n_outputs = n_outputs
        self.activation = nn.Identity()  # for URM; does not affect other algorithms

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        x = self.activation(x)  # for URM; does not affect other algorithms
        return x


class DinoV2(torch.nn.Module):
    """ """

    def __init__(self, input_shape, hparams):
        super(DinoV2, self).__init__()

        self.network = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.n_outputs = 5 * 768

        nc = input_shape[0]

        if nc != 3:
            raise RuntimeError("Inputs must have 3 channels")

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams["vit_dropout"])

        if hparams["vit_attn_tune"]:
            for n, p in self.network.named_parameters():
                if "attn" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

    def forward(self, x):
        x = self.network.get_intermediate_layers(x, n=4, return_class_token=True)
        linear_input = torch.cat(
            [x[0][1], x[1][1], x[2][1], x[3][1], x[3][0].mean(1)], dim=1
        )
        return self.dropout(linear_input)


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""

    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams["resnet18"]:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048

        if hparams["resnet50_augmix"]:
            self.network = timm.create_model("resnet50.ram_in1k", pretrained=True)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        if hparams["freeze_bn"]:
            self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams["resnet_dropout"])
        self.activation = nn.Identity()  # for URM; does not affect other algorithms

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        x = self.network(x)
        if x.dim() == 4:
            x = x.mean(dim=[2, 3])  # GAP: [B, C, H, W] → [B, C]
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
        self.activation = nn.Identity()  # for URM; does not affect other algorithms

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
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.0)
    elif input_shape[1:3] == (224, 224):
        backbone = _resolve_backbone_name(hparams)
        if backbone == "resnet50":
            return ResNet50(input_shape, hparams)
        if backbone == "dinov2":
            return DinoV2(input_shape, hparams)
        if backbone == "vit":
            return ViT(input_shape, hparams)
        if backbone == "alexnet":
            return AlexNet(input_shape, hparams)
        if backbone == "efficientnet":
            return EfficientNet(input_shape, hparams)
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
            torch.nn.Linear(in_features // 4, out_features),
        )
    else:
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs, num_classes, hparams["nonlinear_classifier"]
        )
        self.net = nn.Sequential(featurizer, classifier)
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)


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
        return "resnet50"
    if not isinstance(backbone, str) or not backbone.strip():
        raise ValueError("Invalid `backbone`: expected a non-empty string.")

    normalized = backbone.strip().lower()
    mapping = {
        "resnet50": "resnet50",
        "vit": "vit",
        "alexnet": "alexnet",
        "efficientnet": "efficientnet",
        "dinov2": "dinov2",
    }

    if normalized not in mapping:
        raise ValueError(
            f"Invalid backbone '{backbone}': choose from ResNet, ViT, AlexNet, EfficientNet, DINOv2"
        )
    return mapping[normalized]


class ResNet50(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""

    def __init__(self, input_shape, hparams):
        super(ResNet50, self).__init__()

        self.network = _build_torchvision_model("resnet50", "ResNet50_Weights")
        self.n_outputs = 2048

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            conv1 = getattr(self.network, "conv1", None)
            if conv1 is not None:
                tmp = self.network.conv1.weight.data.clone()

                self.network.conv1 = nn.Conv2d(
                    nc,
                    64,
                    kernel_size=(7, 7),
                    stride=(2, 2),
                    padding=(3, 3),
                    bias=False,
                )

                for i in range(nc):
                    self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        if hparams["freeze_bn"]:
            self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams["resnet_dropout"])
        self.activation = nn.Identity()  # for URM; does not affect other algorithms

    def forward(self, x):
        x = self.network.conv1(x)
        x = self.network.bn1(x)
        x = self.network.relu(x)
        x = self.network.maxpool(x)

        x = self.network.layer1(x)
        x = self.network.layer2(x)
        x = self.network.layer3(x)
        x = self.network.layer4(x)
        # x.shape = (B, C, H, W), (B,2048,7,7)
        if x.dim() == 4:
            x = x.mean(dim=[2, 3])
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


class PrivateHead(nn.Module):
    def __init__(self, input_dim, hparams):
        super(PrivateHead, self).__init__()
        dropout = hparams.get("mlp_dropout", 0.1)
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
            nn.BatchNorm2d(input_dim),
        )

        # Token adapter for sequence features: (B, N, C)
        self.token_adapter = nn.Sequential(
            nn.Linear(input_dim, vec_dim, bias=False),
            nn.LayerNorm(vec_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(vec_dim, input_dim, bias=False),
            nn.LayerNorm(input_dim),
        )

        # Vector adapter for pooled features: (B, C)
        self.vector_adapter = nn.Sequential(
            nn.Linear(input_dim, vec_dim, bias=False),
            nn.LayerNorm(vec_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(vec_dim, input_dim, bias=False),
            nn.LayerNorm(input_dim),
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
        self.network = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=0
        )
        self.n_outputs = 768

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams.get("vit_dropout", 0.0))
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
                x = (
                    torch.cat([cls.unsqueeze(1), patches], dim=1)
                    if cls is not None
                    else patches
                )
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

        self.network = _build_torchvision_model(
            "efficientnet_b0", "EfficientNet_B0_Weights"
        )
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
                bias=False,
            )
            for i in range(nc):
                new_conv.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]
            self.network.features[0][0] = new_conv

        # Remove classifier head and keep encoder features only.
        self.network.classifier = Identity()

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams.get("resnet_dropout", 0.0))
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
                bias=False,
            )
            for i in range(nc):
                new_conv.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]
            self.network.features[0] = new_conv

        # Remove classifier and keep spatial encoder output.
        self.network.classifier = Identity()

        self.dropout = nn.Dropout(hparams.get("resnet_dropout", 0.0))
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

        self.upsample = (
            nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False)
            if stride > 1
            else nn.Identity()
        )
        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride > 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
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

        # Use 1x1 conv to project concatenated features (e.g., 4096 channels)
        # back to ResNet50's internal 2048 channels for decoder reconstruction.
        self.prep_conv = nn.Conv2d(feature_dim, 2048, kernel_size=1)

        # 閫嗗悜璺緞淇濇寔涓嶅彉
        self.layer4 = ReverseBottleneck(2048, 1024, stride=2)
        self.layer3 = ReverseBottleneck(1024, 512, stride=2)
        self.layer2 = ReverseBottleneck(512, 256, stride=2)
        self.layer1 = ReverseBottleneck(256, 64, stride=2)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, input_shape[0], kernel_size=3, padding=1),
            nn.Sigmoid(),
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
            x = F.interpolate(
                x, size=self.input_shape[1:], mode="bilinear", align_corners=False
            )
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
            x = (
                x.reshape(b, 2, expected_n, c)
                .permute(0, 2, 1, 3)
                .reshape(b, expected_n, 2 * c)
            )
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
        self.channel_adjust = (
            nn.Conv2d(feature_dim, 1280, kernel_size=1)
            if feature_dim != 1280
            else nn.Identity()
        )

        self.up1 = nn.ConvTranspose2d(1280, 512, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.up5 = nn.ConvTranspose2d(
            64, input_shape[0], kernel_size=4, stride=2, padding=1
        )

        # Skip connections from bottleneck to each decoder stage.
        self.skip1 = nn.Conv2d(1280, 512, kernel_size=1, bias=False)
        self.skip2 = nn.Conv2d(1280, 256, kernel_size=1, bias=False)
        self.skip3 = nn.Conv2d(1280, 128, kernel_size=1, bias=False)
        self.skip4 = nn.Conv2d(1280, 64, kernel_size=1, bias=False)

    def _fuse_skip(self, x, bottleneck, proj):
        skip = F.interpolate(
            bottleneck, size=x.shape[2:], mode="bilinear", align_corners=False
        )
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
            x = F.interpolate(
                x, size=self.input_shape[1:], mode="bilinear", align_corners=False
            )
        return x


class AlexNetDecoder(nn.Module):
    def __init__(self, feature_dim, input_shape, hparams):
        super(AlexNetDecoder, self).__init__()
        self.input_shape = input_shape
        self.channel_adjust = (
            nn.Conv2d(feature_dim, 256, kernel_size=1)
            if feature_dim != 256
            else nn.Identity()
        )

        self.upsample_blocks = nn.ModuleList(
            [
                self._make_up_block(256, 128),
                self._make_up_block(128, 64),
                self._make_up_block(64, 32),
                self._make_up_block(32, 16),
                self._make_up_block(16, 16),
            ]
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(16, input_shape[0], kernel_size=3, padding=1), nn.Sigmoid()
        )

    def _make_up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = _tokens_to_feature_map(x)
        x = self.channel_adjust(x)

        for block in self.upsample_blocks:
            x = block(x)

        x = self.final_conv(x)
        if x.shape[2:] != self.input_shape[1:]:
            x = F.interpolate(
                x, size=self.input_shape[1:], mode="bilinear", align_corners=False
            )
        return x


class ViTDecoder(nn.Module):
    def __init__(self, feature_dim, input_shape, hparams):
        super(ViTDecoder, self).__init__()
        self.input_shape = input_shape  # original image shape (C, H, W)
        self.feature_dim = feature_dim  # feature dimension (E or C), typically 768

        patch_size = 16
        img_size = input_shape[1]  # usually 224
        num_patches = (img_size // patch_size) ** 2  # the number of patches

        self.patch_size = patch_size
        self.num_patches = num_patches

        # linear projection layer for initial transformation of input feature
        self.decoder_embed = nn.Linear(feature_dim, feature_dim)
        # define position coding
        # +1 for CLS token position
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, feature_dim)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        # stack 6 decoder_layer
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.decoder_norm = nn.LayerNorm(feature_dim)
        self.decoder_pred = nn.Linear(
            feature_dim, patch_size * patch_size * input_shape[0], bias=True
        )

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
        h = w = int(self.num_patches**0.5)
        x = x.reshape(B, h, w, p, p, self.input_shape[0])
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, self.input_shape[0], h * p, w * p)

        return x


class MambaFusionBlock(nn.Module):
    """
    Mamba2 融合模块（通道交替版本）：专门用于融合正交的因果特征和私有特征。

    核心设计思想：
      - 因果特征 C_causal 和私有特征 C_priv 在通道维度正交（不相关但互补）
      - 按通道交替拼接：[c_0, p_0, c_1, p_1, ..., c_{C-1}, p_{C-1}]
      - Mamba 沿空间序列 (H*W) 扫描，但每个 token 的通道维度已经编码了因果-私有的配对
      - 这样 Mamba 学习的是"同一空间位置上，因果 dim i 和私有 dim i 如何交互"

    工作流程：
      1. 接收两个输入：causal_features (B, C, H, W) + private_features (B, C, H, W)
      2. 按通道交替拼接 → (B, 2C, H, W)
      3. reshape 为序列：(B, H*W, 2C)
      4. 投影到 d_model → Mamba2Simple 扫描 → 投影回 2C
      5. 恢复空间格式 → (B, 2C, H, W) 供解码器使用

    参数:
        in_channels: 单个特征的通道数 C（因果或私有的通道数，拼接后为 2C）
        d_model:     Mamba 内部的隐藏维度
        d_state:     SSM 的状态维度
        headdim:     每个注意力头的维度
        expand:      Mamba 内部扩展倍数
    """

    def __init__(self, in_channels, d_model=256, d_state=64, headdim=64, expand=2):
        super().__init__()
        self.in_channels = in_channels  # 注意：这是单个特征的通道数 C
        self.concat_channels = in_channels * 2  # 拼接后为 2C
        self.d_model = d_model

        # 输入投影：将 2C 维特征降维到 Mamba 的 d_model
        self.in_proj = nn.Linear(self.concat_channels, d_model)
        self.norm_in = nn.LayerNorm(d_model)

        # Mamba2Simple (ViM 双向版本)：沿空间序列扫描，学习因果-私有交互
        self.mamba = Mamba2Simple(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            expand=expand,
        )

        # 输出投影：恢复回 2C 通道
        self.out_proj = nn.Linear(d_model, self.concat_channels)
        self.norm_out = nn.LayerNorm(self.concat_channels)

        # 残差缩放因子：小初始值保证训练稳定性
        self.res_scale = nn.Parameter(torch.tensor(0.2))

    def _interleave_channels(self, causal, private):
        """
        按通道交替拼接因果特征和私有特征。

        输入:
            causal:  (B, C, H, W)
            private: (B, C, H, W)
        输出:
            interleaved: (B, 2C, H, W)
            排列方式: [c_0, p_0, c_1, p_1, ..., c_{C-1}, p_{C-1}]
        """
        # stack → (B, 2, C, H, W) → reshape → (B, 2C, H, W)
        stacked = torch.stack([causal, private], dim=1)
        return stacked.reshape(causal.size(0), -1, causal.size(2), causal.size(3))

    def forward(self, causal_features, private_features):
        """
        因果特征和私有特征的通道交替融合。

        输入:
            causal_features:  (B, C, H, W) CNN 特征  或  (B, N, C) ViT token
            private_features: (B, C, H, W) CNN 特征  或  (B, N, C) ViT token
        输出:
            fused: (B, 2C, H, W) 或 (B, N, 2C) — 融合后的特征供解码器使用
        """
        # --- 第一步：记住原始形状，判断是 CNN 还是 ViT ---
        is_4d = causal_features.dim() == 4

        if is_4d:
            # CNN 特征 (B, C, H, W)
            B, C, H, W = causal_features.shape
            # 通道交替拼接: (B, C, H, W) + (B, C, H, W) → (B, 2C, H, W)
            x_concat = self._interleave_channels(causal_features, private_features)
            # reshape 为序列: (B, 2C, H, W) → (B, H*W, 2C)
            x_seq = x_concat.reshape(B, self.concat_channels, H * W).permute(0, 2, 1)
        elif causal_features.dim() == 3:
            # ViT token (B, N, C)
            B, N, C = causal_features.shape
            # 通道交替拼接: (B, N, C) + (B, N, C) → (B, N, 2C)
            x_concat = torch.cat([causal_features, private_features], dim=2)
            # 重新排列为交替: (B, N, 2C) → 需要逐维交织
            # reshape: (B, N, C, 2) → permute: (B, N, 2, C) → (B, N, 2C)
            x_concat = torch.stack(
                [causal_features, private_features], dim=3
            )  # (B, N, C, 2)
            x_concat = x_concat.permute(0, 1, 3, 2)  # (B, N, 2, C)
            x_concat = x_concat.reshape(B, N, self.concat_channels)  # (B, N, 2C)
            x_seq = x_concat  # 已经是 (B, N, 2C)
        else:
            raise ValueError(
                f"MambaFusionBlock 期望 3D 或 4D 输入, 收到 {causal_features.dim()}D"
            )

        # 保存残差
        residual = x_seq

        # --- 第二步：投影到 Mamba 的 d_model ---
        h = self.in_proj(x_seq)  # (B, L, d_model), L=H*W 或 N
        h = self.norm_in(h)

        # --- 第三步：Mamba2 序列建模（ViM 双向扫描）---
        # Mamba 沿序列轴 L 扫描，但每个 token 的 2C 维已经编码了因果-私有配对
        h = self.mamba(h)  # (B, L, d_model)

        # --- 第四步：投影回拼接后的通道维度 2C ---
        h = self.out_proj(h)  # (B, L, 2C)
        h = self.norm_out(h)

        # --- 第五步：残差连接 ---
        x_seq = residual + self.res_scale * h

        # --- 第六步：恢复原始形状 ---
        if is_4d:
            # (B, H*W, 2C) → (B, 2C, H, W)
            x_out = x_seq.permute(0, 2, 1).reshape(B, self.concat_channels, H, W)
        else:
            # (B, N, 2C) 保持不变
            x_out = x_seq

        return x_out


class Mamba2Simple(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=128,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(
                torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs)
            )
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(
            *A_init_range
        )
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(
            self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs
        )

        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

    def forward(self, u, seq_idx=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states = (
            repeat(self.init_states, "... -> b ...", b=batch)
            if self.learnable_init_states
            else None
        )
        dt_limit_kwargs = (
            {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        )

        if self.use_mem_eff_path:
            # Fully fused path
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight,
                rmsnorm_eps=self.norm.eps,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=False,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
        else:
            z, xBC, dt = torch.split(
                zxbcdt,
                [
                    self.d_inner,
                    self.d_inner + 2 * self.ngroups * self.d_state,
                    self.nheads,
                ],
                dim=-1,
            )
            dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
            assert self.activation in ["silu", "swish"]

            # 1D Convolution
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
                )  # (B, L, self.d_inner + 2 * ngroups * d_state)
                xBC = xBC[:, :seqlen, :]
            else:
                xBC = causal_conv1d_fn(
                    x=xBC.transpose(1, 2),
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)

            # Split into 3 main branches: X, B, C
            # These correspond to V, K, Q respectively in the SSM/attention duality
            x, B, C = torch.split(
                xBC,
                [
                    self.d_inner,
                    self.ngroups * self.d_state,
                    self.ngroups * self.d_state,
                ],
                dim=-1,
            )
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                seq_idx=seq_idx,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
            y = rearrange(y, "b l h p -> b l (h p)")

            # Multiply "gate" branch and apply extra normalization layer
            y = self.norm(y, z)
            out = self.out_proj(y)
        return out


class ResNet50FeatureMap(nn.Module):
    """
    ============================================================================
    ResNet50 — 输出特征图而非向量
    ============================================================================
    标准的 DomainBed ResNet 去掉了 avgpool 和 fc，但保留了 avgpool 之前的平均池化，
    因此输出 shape 为 [B, 2048]（向量）。而本模块的 MoE 专家需要保留空间结构
    （特征图）才能做 cross-attention 注入风格 tokens，所以需要 layer4 的原生输出。

    设计决策：
      - 去掉 avgpool 和 fc → 保留 [B, 2048, 7, 7] 的空间结构
      - 1x1 conv 降维 2048 → 512：减少后续 cross-attention 的计算量，
        同时 512 维足够编码风格信息（经验值，StyleGAN 的 w 空间也是 512 维）

    冻结策略：
      冻结 conv1、bn1、layer1（前几层学的是通用低级特征，不需要微调）。
      layer2~layer4 参与训练，让高层特征适应目标任务。
    ============================================================================
    """

    def __init__(self, input_shape, feat_dim=512, freeze_early_layers=True):
        super().__init__()
        # ------------------------------------------------------------------
        # 加载 ImageNet 预训练的 ResNet50
        # 通过保持 named attributes 而非 nn.Sequential，可以精确控制哪些层冻结
        # ------------------------------------------------------------------
        resnet = torchvision.models.resnet50(pretrained=True)

        # 保留 conv1~layer4（不含 avgpool 和 fc）
        self.conv1 = resnet.conv1  # [B, 3, 224, 224] → [B, 64, 112, 112]
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # [B, 64, 56, 56]
        self.layer1 = resnet.layer1  # [B, 256, 56, 56]
        self.layer2 = resnet.layer2  # [B, 512, 28, 28]
        self.layer3 = resnet.layer3  # [B, 1024, 14, 14]
        self.layer4 = resnet.layer4  # [B, 2048, 7, 7] ← 保留空间结构

        # 1×1 conv 降维: 2048 → feat_dim (default 512)
        # 1x1 conv 只做通道混合，不改变空间结构
        self.reducer = nn.Conv2d(2048, feat_dim, kernel_size=1, bias=False)
        self.feat_dim = feat_dim

        # ------------------------------------------------------------------
        # 冻结前几层：
        #   conv1 + bn1: 学的是颜色、边缘等通用低级特征（Gabor-like filters）
        #   layer1:      学的是局部纹理组合，也是通用的
        #   layer2~4:    学的是语义级别的特征，参与领域泛化训练
        # ------------------------------------------------------------------
        if freeze_early_layers:
            frozen_names = ["conv1", "bn1", "layer1"]
            for name in frozen_names:
                layer = getattr(self, name)
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, x):
        """
        前传: [B, 3, 224, 224] → [B, feat_dim, 7, 7]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B, 2048, 7, 7]

        x = self.reducer(x)  # [B, feat_dim, 7, 7]
        return x


class GradientReversal(torch.autograd.Function):
    """
    梯度反转层（Gradient Reversal Layer, GRL）

    前向: 恒等映射 y = x
    反向: 梯度取反并缩放 ∂L/∂x = -lambda * ∂L/∂y

    lambda 控制对抗强度：
      - 0.0: 无对抗（专家变温和）
      - 1.0: 完整反转（容易发散）
      - 0.1-0.3: 温和对抗（推荐）
    """

    @staticmethod
    def forward(ctx, x, lambda_=0.1):
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def gradient_reverse(x, lambda_=0.1):
    return GradientReversal.apply(x, lambda_)


class StyleQueryExpert(nn.Module):
    """
    Style-Query Cross-Attention Expert

    核心设计：
      - IN 移除通道 μ,σ（风格），content 只保留空间激活模式
      - Q = style_query（从 μ,σ 编码而来），回答"面对这个风格，怎么混合 tokens"
      - K, V = style_tokens（可学习的反制基底）
      - content 路径保留正常梯度，backbone 正常学习

    训练后每个专家擅长反制特定风格域的分类器捷径。
    """

    def __init__(self, feat_dim=512, n_tokens=8, n_heads=8):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(feat_dim, affine=False)
        self.style_tokens = nn.Parameter(torch.randn(1, n_tokens, feat_dim) * 0.02)
        self.style_enc = nn.Linear(feat_dim * 2, feat_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, feat):
        B, C, H, W = feat.shape

        # 内容分支：IN → 只保留空间激活模式（无风格）
        content = self.instance_norm(feat)  # [B, C, H, W]
        content_flat = content.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # 风格分支：μ,σ → style_enc → style_query
        mu = feat.mean(dim=[2, 3]).detach()       # [B, C]
        sigma = feat.std(dim=[2, 3]).detach()     # [B, C]
        style_query = self.style_enc(torch.cat([mu, sigma], dim=1))  # [B, C]
        style_query = style_query.unsqueeze(1)    # [B, 1, C]

        # Q=style_query, K=V=style_tokens
        # 注意力回答："面对这个风格，应该怎么混合反制基底"
        tokens = self.style_tokens.expand(B, -1, -1)  # [B, N, C]
        attn_out, _ = self.cross_attn(
            query=style_query,
            key=tokens,
            value=tokens,
        )  # [B, 1, C]

        # 残差连接，broadcast attn_out 到所有空间位置
        out = self.norm(content_flat + attn_out)  # [B, H*W, C]
        out = out.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        return out
