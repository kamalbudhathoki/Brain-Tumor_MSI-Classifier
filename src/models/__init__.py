"""Model definitions: architectures and the factory that builds them."""

from .classifier import (
    ARCH_NAME,
    CUSTOM_CNN_ARCH,
    NUM_CLASSES,
    RESNET18_ARCH,
    RESNET18_WEIGHTS,
    BrainTumorCNN,
    BrainTumorResNet18,
    build_model,
    trainable_parameters,
)

__all__ = [
    "ARCH_NAME",
    "CUSTOM_CNN_ARCH",
    "NUM_CLASSES",
    "RESNET18_ARCH",
    "RESNET18_WEIGHTS",
    "BrainTumorCNN",
    "BrainTumorResNet18",
    "build_model",
    "trainable_parameters",
]
