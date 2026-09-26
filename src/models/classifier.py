"""
classifier.py -- a small custom CNN for brain tumor MRI classification.

WHAT THIS FILE DOES
-------------------
`src/data/` turns folders of MRI scans into tensors. This module turns those
tensors into class predictions, and owns the network definition:

    BrainTumorCNN   -- the network itself
    build_model()   -- the factory the rest of the project calls

    from src.models.classifier import build_model

    model = build_model(num_classes=4)
    logits = model(images)          # images: [batch, 3, 224, 224]

WHY A CUSTOM CNN WHEN A PRETRAINED ONE EXISTS
---------------------------------------------
Honest answer: on a few thousand MRI slices, a ResNet-50 finetuned from ImageNet
will beat this network, and you should use that as your baseline. See the
"Caveats" section at the bottom -- this architecture has a well-known parameter
problem that the pretrained path does not have.

A custom CNN is still worth building, for three reasons:

  1. It is the clearest way to understand what a convolutional network actually
     does. Every layer below is explained.
  2. It is a *controlled baseline*. If finetuning ResNet-50 gives 97% and this
     gives 88%, that gap tells you something real about how much the data and
     the pretrained features are carrying you.
  3. It has ~150k parameters instead of ~25M, so it trains in seconds on a CPU
     and is fast enough to iterate on.

THE LAYER STACK, TOP TO BOTTOM
------------------------------
Input [batch, 3, 224, 224] -> output [batch, num_classes] (raw logits).

    Layer            Output shape        What it does
    ---------------  ------------------  ----------------------------
    Conv1 + ReLU     [B, 32, 224, 224]   find 32 low-level edge maps
    MaxPool1         [B, 32, 112, 112]   halve the grid, keep the strongest
    Conv2 + ReLU     [B, 64, 112, 112]   combine edges into textures
    MaxPool2         [B, 64,  56,  56]   halve again
    Flatten          [B, 200704]         56*56*64 numbers, no structure
    FC1 + ReLU       [B, 128]            200k numbers -> 128 learned features
    Dropout          [B, 128]            randomly zero 50% during training
    FC2              [B, num_classes]    128 numbers -> one score per class

A NOTE ON THE MISSING ReLU
-------------------------
The original layer list was Conv/ReLU/MaxPool x2, Flatten, FC, Dropout, Output.
A ReLU after FC1 is added here, and it is not optional: without a nonlinearity
between two linear layers, the pair collapses mathematically into a *single*
linear layer (the composition of two linear maps is linear), so the hidden layer
buys you nothing but parameters and overfitting. Dropout would still work, but
the network would be strictly weaker for no reason.

CAVEATS -- THE PARAMETER BLOW-UP
-------------------------------
This architecture, as specified, is front-loaded with parameters in the wrong
place. After two 2x2 pools a 224x224 image is a 56x56x64 grid, and flattening
that gives 56 * 56 * 64 = 200,704 numbers going into one fully-connected layer:

    conv1    32 * (3*3*3)   + 32   =      896   (0.003%)
    conv2    64 * (32*3*3)  + 64   =   18,496   (0.07%)
    fc1      200704 * 128   + 128  = 25,690,240  (99.92%)
    fc2      128 * 4        + 4    =        516
                                         ---------
    total                                    25,710,148

The convolutional layers, which do the actual visual work, hold 0.08% of the
weights. A single dense layer holds 99.9%. That is backwards, and it is the
classic mistake in hand-written CNNs.

Two costs:

  * **Memory.** 25.7M floats is ~100MB of weights plus gradients plus Adam
    optimizer state -- on the order of 400MB, for a network that could fit in
    1MB.
  * **Overfitting.** 25.7M parameters learning from a few thousand slices is
    exactly the memorisation regime that src/data/transforms.py's augmentation
    exists to fight. Convolutional layers generalise because they share weights
    across position; a flattened dense layer does not, so each of its 200,704
    inputs gets a private weight free to memorise its own pixel neighbourhood.

The standard fix is one line -- an adaptive average pool right before the
flatten, averaging each 56x56 feature map down to 4x4:

    BrainTumorCNN(num_classes=4, pool_to=4)
    # fc1 becomes 1024 * 128 + 128 = 131,200 parameters
    # total drops to ~151,000 -- roughly 170x smaller

Average pooling also keeps *whether* a feature fired while discarding *where*,
which for tumour type is usually the part you want, and is translation-invariant
to boot.

The default is left at `pool_to=None` -- the architecture exactly as specified,
so the numbers above are what you get out of the box. Pass `pool_to=4`, or start
from a pretrained ResNet-50, for real work.
"""

from __future__ import annotations

import torch
from torch import nn

# The model's input size must equal the size src/data/transforms.py resizes to.
# Imported rather than re-typed as a literal: if preprocessing and architecture
# ever disagree, the model fails with a confusing shape error deep in the
# forward pass. One source of truth is cheaper than debugging that.
from ..data.transforms import IMAGE_SIZE

#: The four classes in the Kaggle "Brain Tumor MRI Dataset" this project targets.
NUM_CLASSES: int = 4

#: Channel count of the first conv block. Doubled by the second: 32 -> 64.
#: CNNs conventionally widen as they deepen, because early layers only need a few
#: filters to detect edges while later layers need many to describe textures.
BASE_CHANNELS: int = 32

#: Width of the hidden fully-connected layer.
HIDDEN_SIZE: int = 128

#: Fraction of hidden units zeroed during training. 0.5 is the standard default
#: and is a good fit for a network this small.
DROPOUT_P: float = 0.5

#: The name this architecture is registered under.
ARCH_NAME: str = "custom_cnn"


class BrainTumorCNN(nn.Module):
    """A two-block convolutional network for 4-class brain tumor MRI scans.

    Every layer is a named attribute rather than hidden inside a
    `nn.Sequential`. That is deliberate: explainability work like Grad-CAM
    (planned in `src/inference.py`) works by registering forward hooks on
    individual layers by name, and `self.features[-2]` is not a usable handle.

    Args:
        num_classes: how many classes to score. 4 for glioma / meningioma /
            pituitary / no_tumor.
        in_channels: 3, because `BrainTumorDataset` converts every grayscale
            scan to RGB so ImageNet-style weights and normalisation apply.
        image_size: side length of the square input. Used only to work out the
            flatten width, so the model adapts if preprocessing changes.
        base_channels: output channels of conv1; conv2 uses 2x this.
        hidden_size: width of the hidden fully-connected layer.
        dropout_p: dropout probability.
        pool_to: if not None, insert a global average pool down to this size
            (e.g. 4) immediately before flattening. This is the fix for the
            parameter blow-up described in `Caveats`; leave it None for the
            plain architecture.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = 3,
        image_size: int = IMAGE_SIZE,
        base_channels: int = BASE_CHANNELS,
        hidden_size: int = HIDDEN_SIZE,
        dropout_p: float = DROPOUT_P,
        pool_to: int | None = None,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p}")
        if pool_to is not None and pool_to < 1:
            raise ValueError(f"pool_to must be >= 1, got {pool_to}")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.image_size = image_size
        self.base_channels = base_channels
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.pool_to = pool_to

        channels2 = base_channels * 2

        # --- convolutional blocks ------------------------------------------
        # Two blocks, each "convolution -> ReLU -> max pool". This pairing is the
        # standard CNN unit: conv extracts, ReLU adds nonlinearity, pooling
        # compacts. `padding=1` with a 3x3 kernel is called "same" padding and
        # keeps the spatial size unchanged, so the pooling maths stays obvious.
        self.features = nn.Sequential(
            # --- Block 1: edges -------------------------------------------
            # 32 filters, each 3x3x3. 3*3*3 = 27 weights per filter, so 864
            # weights total. A filter this small is enough to detect a straight
            # edge at any angle, or a bright blob boundary. 32 of them covers
            # the handful of edge orientations that actually occur in a scan.
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 2x2 window, stride 2: keeps the max response in each 2x2 block and
            # halves width and height, so 4x fewer values to process downstream.
            # The brain is mostly smooth background, so throwing away 3 of every
            # 4 pixels costs very little signal.
            nn.MaxPool2d(kernel_size=2, stride=2),

            # --- Block 2: textures -----------------------------------------
            # Now each filter sees a 3x3 patch *of the previous block's output*,
            # i.e. of learned edge maps rather than raw pixels. That is what
            # makes hierarchical features: edges in layer 1 become shapes and
            # textures here. 64 filters, receptive field now 5x5 of the original.
            nn.Conv2d(base_channels, channels2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Optional global pool. `nn.Identity()` is a no-op module, which lets the
        # plain and pooled architectures share one code path.
        self.reduce = (
            nn.Identity()
            if pool_to is None
            else nn.AdaptiveAvgPool2d(pool_to)
        )

        # Work out the flatten width by running one dummy image through the
        # convolutional part. Hardcoding 56*56*64 would be correct only for
        # exactly 224x224 input and would break silently the first time the
        # image size or pooling changed.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            self.flatten_features = int(self.reduce(self.features(dummy)).flatten(1).shape[1])

        # --- classifier head ----------------------------------------------
        # This is where the network stops being spatial and starts being
        # "which of the four classes is this".
        self.classifier = nn.Sequential(
            # [B, C, H, W] -> [B, C*H*W]. Flatten only the four leading dims
            # apart: start_dim=1 keeps the batch dimension intact, so one image
            # in gives one row out.
            nn.Flatten(start_dim=1),

            # The only genuinely large layer in the network. 200704 weights in,
            # 128 learned combinations out. Each of the 128 units is a
            # hand-designed-ish detector for some high-level pattern.
            nn.Linear(self.flatten_features, hidden_size),
            # See "A NOTE ON THE MISSING ReLU" in the module docstring.
            nn.ReLU(inplace=True),

            # During training, each hidden unit is zeroed with probability 0.5,
            # independently per example per step. The network is forced to
            # spread the job across many units instead of relying on a few, and
            # averaging over the ~2^n dropout patterns approximates ensembling.
            # Inert at eval time, so predictions are deterministic.
            nn.Dropout(p=dropout_p),

            # The output layer: one raw score (logit) per class. No softmax here
            # on purpose -- `nn.CrossEntropyLoss` expects logits and applies
            # log-softmax itself. Adding a softmax now would be applying it
            # twice, and it would produce log-probabilities fed to a function
            # expecting logits, which trains badly.
            nn.Linear(hidden_size, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Set sensible starting weights.

        Default PyTorch initialisation is uniform in a tiny range chosen for
        logistic regression, which is far too small for a network with ReLUs:
        activations shrink at every layer, the gradients vanish, and training
        crawls. Kaiming initialisation scales each layer by the gain its
        activation preserves, so signal neither explodes nor dies.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of images to class logits.

        Args:
            x: float32 tensor of shape [batch, in_channels, image_size, image_size],
               already resized and normalised by `src/data/transforms.py`.

        Returns:
            float32 tensor of shape [batch, num_classes] -- unnormalised scores,
            higher means more likely. Softmax if you want probabilities.
        """
        x = self.features(x)   # edges, then textures
        x = self.reduce(x)     # no-op unless pool_to was given
        return self.classifier(x)

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters. Handy for sanity-checking the architecture."""
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad or not trainable_only
        )


def build_model(
    num_classes: int = NUM_CLASSES,
    name: str = ARCH_NAME,
    **kwargs,
) -> nn.Module:
    """Factory that returns a model by name.

    `src/README.md` reserves this signature so the training and inference code
    can swap architectures without changing: `build_model("resnet50", 4)` will
    work the day that backbone is added, and every caller already goes through
    the registry instead of hardcoding a class.

    Args:
        num_classes: how many classes to predict.
        name: architecture key. Currently only `ARCH_NAME`.
        **kwargs: forwarded to the architecture, e.g. `dropout_p=0.3`.

    Returns:
        An uninitialised-output `nn.Module` ready for training.
    """
    architectures: dict[str, type[nn.Module]] = {
        ARCH_NAME: BrainTumorCNN,
    }

    if name not in architectures:
        raise ValueError(
            f"Unknown architecture {name!r}. "
            f"Available: {sorted(architectures)}"
        )

    return architectures[name](num_classes=num_classes, **kwargs)
