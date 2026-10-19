"""Network definitions for the two stages."""

from .cnn import CNNCifar
from .registry import build_model, count_parameters, describe
from .resnet import BasicBlock, ResNet, resnet18, resnet34

__all__ = [
    "CNNCifar",
    "build_model",
    "count_parameters",
    "describe",
    "BasicBlock",
    "ResNet",
    "resnet18",
    "resnet34",
]
