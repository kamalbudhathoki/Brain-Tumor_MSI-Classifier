"""Model definitions: architectures and the factory that builds them."""

from .classifier import ARCH_NAME, NUM_CLASSES, BrainTumorCNN, build_model

__all__ = ["ARCH_NAME", "NUM_CLASSES", "BrainTumorCNN", "build_model"]
