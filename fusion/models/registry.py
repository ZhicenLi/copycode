"""Model factory shared by both stages.

The two stages never share weights: each one calls `build_model` on its own and
gets a freshly initialised network. `pretrained` exists only so the FedDS stage
can be aligned with the published CIFAR-100 numbers when needed; it is off by
default so both stages start from scratch and stay comparable.
"""

from __future__ import annotations


import torch.nn as nn

from .cnn import CNNCifar
from .resnet import count_parameters, resnet18, resnet34


def build_model(
    name: str,
    num_classes: int,
    in_channels: int = 3,
    pretrained: bool = False,
) -> nn.Module:
    if name == "cnncifar":
        return CNNCifar(num_classes=num_classes, in_channels=in_channels)
    if name == "resnet18":
        return resnet18(num_classes=num_classes, in_channels=in_channels)
    if name == "resnet34":
        return resnet34(num_classes=num_classes, in_channels=in_channels)
    if name == "resnet18_cifar100":
        return _resnet18_torchvision(num_classes, in_channels, pretrained)
    raise ValueError(f"unknown model '{name}'")


def _resnet18_torchvision(num_classes: int, in_channels: int, pretrained: bool) -> nn.Module:
    """CIFAR-shaped ResNet-18.

    With ``pretrained=False`` (the default) the project's own ResNet is used, so
    the local CPU smoke test and the GPU server run *identical* code. Only the
    pretrained path needs torchvision, because that is the sole way to obtain
    ImageNet weights.
    """
    if not pretrained:
        return resnet18(num_classes=num_classes, in_channels=in_channels)

    try:
        from torchvision import models
    except ImportError as exc:  # pragma: no cover - torchvision is optional locally
        raise RuntimeError(
            "pretrained=True requires torchvision; install it on the server "
            "or set model.fedds_pretrained=false"
        ) from exc

    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    in_features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Linear(in_features, in_features * 2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(in_features * 2, num_classes),
    )
    return net


def describe(model: nn.Module) -> str:
    total = count_parameters(model)
    return f"{model.__class__.__name__}({total / 1e6:.2f}M params)"


__all__ = ["build_model", "describe", "count_parameters"]
