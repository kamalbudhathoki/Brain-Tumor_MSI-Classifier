"""
classifier.py -- the network architectures for brain tumor MRI classification,
and the factory that builds them.

WHAT THIS FILE DOES
-------------------
`src/data/` turns folders of MRI scans into tensors. This module turns those
tensors into class predictions, and owns the network definitions:

    BrainTumorResNet18 -- ImageNet-pretrained ResNet-18, frozen + new head
                         (the default -- see "TRANSFER LEARNING" below)
    BrainTumorCNN     -- a small CNN trained from scratch (the controlled
                         baseline the pretrained number is measured against)
    build_model()     -- the factory the rest of the project calls

    from src.models.classifier import build_model

    model = build_model(num_classes=4)                        # ResNet-18
    model = build_model(num_classes=4, name="custom_cnn")     # the baseline
    logits = model(images)          # images: [batch, 3, 224, 224]

ResNet-18 is the default. It replaced the custom CNN as the primary model because
on a few thousand MRI slices a pretrained backbone wins, and the custom CNN's own
docstring said so before this change was made. The custom CNN is kept, registered
under `"custom_cnn"`, for one reason: it is the *controlled baseline*. A 97%
finetuned model and an 88% scratch-trained model say something real about how
much the data and the pretrained features are carrying you; either number on its
own says nothing. It is also the clearest possible demonstration of what a
convolutional network actually does, with every layer explained below.

    python -m src.train                          # ResNet-18, frozen backbone
    python -m src.train --arch custom_cnn        # the from-scratch baseline

================================================================================
TRANSFER LEARNING: WHAT IT IS, AND WHY IT WORKS ON THIS PROBLEM
================================================================================

THE ONE-SENTENCE VERSION
A pretrained network is a feature extractor that already knows what to look for.
Transfer learning reuses it and retrains only the last layer.

THE PROBLEM IT SOLVES HERE
--------------------------
Training a network from scratch means *learning every weight from the data you
have*. The custom CNN needs to discover from scratch that edges exist, that
edges make textures, that textures make shapes, and which of those shapes indicate
a meningioma rather than a glioma. That is an enormous amount of learning to ask
of a few thousand 2D slices, and the failure mode is not "underfitting" -- it is
overfitting: with 25.7M parameters and a few thousand images, gradient descent
finds the fastest route to a low training loss, and memorising the training scans
is faster than learning tumour morphology. You get a model that reports 99% on
the data it saw and 61% on anything new.

THE INSIGHT
-----------
A ResNet-18 trained on ImageNet has already done the first two thirds of that
work, on 1.28 million images. Its early layers detect edges and colour blobs;
its middle layers combine them into textures and repeating motifs; only its final
layers are about the *specific* thing it was trained to name, and for a brain
tumour that last part is worthless -- the 1000 ImageNet classes have nothing to
do with glioma.

So throw that last part away and keep the rest. The retained 18 layers do not
know what a tumour is, but they do know what texture, shape, boundary and
intensity structure *look like*, which is most of what a radiologist is reacting
to when they name a tumour type. That vocabulary is domain-agnostic enough to
transfer; the ImageNet label head is not.

WHAT IS ACTUALLY TRAINED, AND WHY FREEZING IS THE RIGHT FIRST MOVE
-------------------------------------------------------------------
A pretrained ResNet-18 is split in two:

    the backbone   conv1 .. layer4 + avgpool     11,176,512 parameters
    the head       fc (512 -> 1000)                   513,000 parameters

`BrainTumorResNet18` keeps the backbone verbatim, **freezes it**, deletes the
ImageNet head, and builds a fresh 512 -> 4 linear layer. On this dataset that
leaves 2,052 trainable parameters out of 11,178,564 -- 0.018%. Everything else
is a constant.

The decision to freeze rather than finetune everything is not a shortcut, it is
the point. Three reasons, in order of importance:

  1. **It is what the data can support.** 2,052 parameters is a model you can
     fit to a few thousand slices without memorising them. Unfreezing all 11.2M
     gives every one of those weights a private opportunity to fit noise, and
     on a dataset this small that reliably costs accuracy rather than gaining
     it.
  2. **It preserves the features you imported.** The pretrained weights encode
     knowledge from 1.28M images. Gradient descent on 3,000 images will happily
     overwrite that knowledge to shave a fraction of a point off the training
     loss. Freezing is the only way to keep it.
  3. **It is cheap.** A frozen backbone produces no gradients. Backpropagation
     stops at the head's input, so a training step costs a forward pass through
     ResNet-18 and a gradient for one 512x4 matmul. This is what makes CPU
     training viable for the first time in this project.

WHAT IS FROZEN, EXACTLY -- AND THE BATCHNORM TRAP
------------------------------------------------
`requires_grad_(False)` is necessary but NOT sufficient, and this is the single
most common bug in hand-rolled transfer learning. Every `BatchNorm2d` in ResNet-18
holds two things: affine `weight`/`bias` parameters, and `running_mean`/
`running_var` *buffers*.

  * The parameters stop receiving gradients the moment you freeze them. Expected.
  * The buffers keep updating anyway, because `model.train()` moves BatchNorm to
    training mode, where it recomputes batch statistics and overwrites the
    running estimates in place -- and buffers are not parameters, so
    `requires_grad=False` does not stop them.

Left alone, the backbone's feature normalisation silently drifts away from the
ImageNet values it was trained with, for the entire duration of training. The
model is no longer "pretrained features + new head"; it is "features slowly
being destroyed + a head trained on top of the wreckage", and it fails *quietly*
-- training loss still falls, so nothing looks wrong.

`BrainTumorResNet18.train()` therefore forces the frozen backbone back into
`eval()` mode after every call to `super().train()`, which is what disables the
running-statistic updates and switches BatchNorm to the frozen running estimates.
Unfreezing the backbone (`unfreeze_backbone()`) releases that override, because
at that point updating the statistics is exactly what you want.

WHY THE HEAD IS RANDOMLY INITIALISED AND THE OLD ONE IS DELETED
----------------------------------------------------------------
`model.fc` cannot be kept. It is 512 -> 1000 and its 513,000 weights encode
ImageNet's label space; there is no way to reuse them for four brain-tumour
classes, and no partial reuse that is not simply arbitrary. `nn.Linear` starts
from PyTorch's default initialisation, which is appropriate here because a linear
layer has no ReLU to worry about -- the vanishing-gradient problem that makes
Kaiming initialisation necessary in `BrainTumorCNN` does not apply to a single
linear layer. That asymmetry is the reason `_initialize_weights` exists on one
class and not the other.

There is one thing worth NOT doing, and it is tempting: keeping the old head and
truncating it to the first four rows. Those rows are ImageNet's arbitrary class
ordering -- whichever classes happened to get index 0-3 (tench, goldfish, great
white shark, tiger shark). Their weights encode "looks like a shark", which is
not a useful starting point for "looks like a meningioma".

WHY THE PRETRAINED PATH ALSO FIXES THE PARAMETER BLOW-UP
--------------------------------------------------------
The custom CNN's "CAVEATS" section below shows a 25.7M-parameter network that is
99.9% one dense layer -- the classic hand-written-CNN mistake. ResNet-18 has an
order of magnitude *more* parameters, and the distribution is the right way round:
convolutions hold ~100% of the weights, and the classifier is 2,052 of them. Its
`avgpool` averages each 7x7 feature map to a single number before the head, so
the head sees 512 numbers rather than 200,704. Same number of trainable
parameters in the head as a sanely-designed scratch network, with an order of
magnitude more feature extraction behind it -- and none of it learned from
3,000 images.

WHAT IS FROZEN IS NOT FOREVER
-----------------------------
The next step up, once the frozen head stops improving, is *progressive
unfreezing* (also called discriminative finetuning): unfreeze `layer4`, give it a
learning rate 10-100x smaller than the head's, and train again. Early layers stay
frozen because low-level filters are the most transferable and the least worth
risk. `unfreeze_backbone()` exists to make that a one-liner. Do not start there:
it is a second experiment, not a first one, and the frozen model is the
measurement you compare it against.

ONE PRACTICAL WARNING
---------------------
ImageNet normalisation is not optional here, and it is not a detail.
`src/data/transforms.py` normalises with `IMAGENET_MEAN` / `IMAGENET_STD` because
those are the constants the pretrained weights were trained with. Feed the same
network differently-normalised inputs and the features it produces are shifted
far enough that the head is reading noise -- again, with training loss that looks
fine. That is why the checkpoint stores `norm_mean` / `norm_std` and why
`src/evaluate.py` is not allowed to guess them.

================================================================================
THE CUSTOM CNN BASELINE -- `BrainTumorCNN`
================================================================================
Retained as the from-scratch baseline. Read the rest of this docstring for why it
is a useful control even though ResNet-18 is now the default.

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
so the numbers above are what you get out of the box. Pass `pool_to=4` for the
fairest possible comparison against the pretrained model, since an unpooled head
over 200k inputs is not a baseline anybody should be proud of.

WHY THE CUSTOM CNN IS STILL WORTH KEEPING
-----------------------------------------
  1. It is the clearest way to understand what a convolutional network actually
     does. Every layer above is explained.
  2. It is a *controlled baseline*. If finetuning ResNet-18 gives 97% and this
     gives 88%, that gap tells you something real about how much the data and
     the pretrained features are carrying you.
  3. It has ~150k parameters instead of ~25M, so it trains in seconds on a CPU
     and is fast enough to iterate on.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

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

#: The ImageNet checkpoint the backbone is loaded from. Named explicitly rather
#: than left as `weights="DEFAULT"` so a torchvision upgrade cannot silently
#: change which weights -- and therefore which reported accuracy -- this project
#: means by "pretrained". The enum also carries the preprocessing constants the
#: weights expect, which is where `IMAGENET_MEAN` / `IMAGENET_STD` come from.
RESNET18_WEIGHTS = ResNet18_Weights.IMAGENET1K_V1

#: The registry key for the transfer-learning model. This is the default.
RESNET18_ARCH: str = "resnet18"

#: The registry key for the from-scratch baseline. Kept so the pretrained
#: architecture has something to be measured against.
CUSTOM_CNN_ARCH: str = "custom_cnn"

#: The architecture `build_model()` reaches for when no name is given, and the
#: default `TrainConfig.arch`. ResNet-18, because the pretrained backbone beats
#: the custom CNN on this dataset -- which the custom CNN's own docstring
#: predicted before either existed. `src/evaluate.py` also uses this as the
#: fallback arch for a checkpoint that somehow carries no `arch` key, so
#: changing it changes what a malformed checkpoint loads as. That is deliberate:
#: the best default for a broken checkpoint is the model you actually ship.
ARCH_NAME: str = RESNET18_ARCH


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


# ---------------------------------------------------------------------------
# The transfer-learning model (the default)
# ---------------------------------------------------------------------------


class BrainTumorResNet18(nn.Module):
    """ImageNet-pretrained ResNet-18 with a frozen backbone and a new 4-way head.

    This is the architecture `build_model()` returns by default. The module
    docstring's "TRANSFER LEARNING" section is the conceptual explanation; this
    docstring is only about the code.

    What the constructor does, in order:

        1. `resnet18(weights=IMAGENET1K_V1)` -- load the pretrained backbone.
        2. Read `fc.in_features` (512) off the head, then replace `fc` with
           `nn.Identity()` so the ImageNet label space is gone rather than merely
           unused.
        3. Build a fresh `nn.Linear(512, num_classes)`.
        4. `freeze_backbone()` -- every backbone parameter gets
           `requires_grad_(False)` and the backbone is put in `eval()` mode.

    After step 4 the split is:

        self.backbone    conv1 .. layer4, avgpool   frozen, 11,176,512 params
        self.dropout     Identity by default        no params
        self.classifier  Linear(512, num_classes)   trained, 512*C + C params

    The attribute names are not arbitrary. `backbone` and `classifier` are the
    names this class and `BrainTumorCNN` share, which is what lets
    `src/evaluate.py` load a checkpoint of either without knowing which it is, and
    `backbone` is the single handle `freeze_backbone()` /
    `unfreeze_backbone()` need. Splitting the head out as a separate attribute
    (rather than leaving a `nn.Sequential` called `classifier`) is also what makes
    the freeze boundary an explicit, greppable line in the source instead of a
    list of module names to remember.

    Args:
        num_classes: how many classes to score. 4 for glioma / meningioma /
            pituitary / no_tumor.
        in_channels: must be 3. `BrainTumorDataset` replicates every grayscale
            scan to 3 channels precisely so the ImageNet stem applies unchanged;
            a different value is a configuration error rather than something to
            paper over, and is rejected below instead of silently loading
            3-channel weights into a different-shaped conv1.
        pretrained: load ImageNet weights. `True` on first use of a run, and
            `False` when *rebuilding a model from a checkpoint* -- the checkpoint
            overwrites every parameter anyway, so the download would be pure
            waste, and it would be a network call in the middle of `load_checkpoint`.
        freeze_backbone: freeze on construction. `False` trains all 11.2M
            parameters, which this project should not do on this dataset -- see
            the module docstring.
        dropout_p: dropout on the pooled 512-d feature vector before the head.
            **0.0 by default**, i.e. the head is a plain linear probe, which is
            the standard starting point and the cleanest thing to compare
            against. Raise it if the head starts memorising. Dropout holds no
            parameters, so changing this does not change the checkpoint's
            `state_dict` keys and old checkpoints still load.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = 3,
        *,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p}")
        if in_channels != 3:
            # Silently adapting conv1 (slicing or averaging the 3 pretrained
            # input channels down to 1) is a real technique, but it changes what
            # the frozen early layers compute, so it belongs in its own
            # experiment with its own checkpoint -- not as a quiet default in the
            # constructor of the model this project ships.
            raise ValueError(
                f"in_channels must be 3 for the ImageNet-pretrained ResNet-18 "
                f"stem, got {in_channels}. BrainTumorDataset already replicates "
                f"grayscale scans to 3 channels; do the conversion in the "
                f"transform pipeline instead of here."
            )

        # `weights=None` builds the same architecture from scratch. Only ever
        # reachable via pretrained=False, which build_model() uses to rebuild a
        # model from a checkpoint.
        net = resnet18(weights=RESNET18_WEIGHTS if pretrained else None)

        # The width of the vector the backbone hands to its head. Read it off the
        # layer about to be deleted rather than written as the literal 512, so a
        # different backbone cannot silently disagree with this head's width.
        self.in_features: int = int(net.fc.in_features)
        # Delete the ImageNet head. `nn.Identity()` rather than `None` because
        # `nn.Module` refuses a None child, and the attribute has to keep
        # existing: `state_dict()` keys are derived from the module tree, and
        # dropping the attribute would drop `fc.*` from the keys.
        net.fc = nn.Identity()

        self.backbone = net
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.pretrained = pretrained
        #: Read by `train()` to decide whether to keep the backbone in eval mode.
        self.backbone_frozen: bool = False

        # `Identity` when p == 0.0 rather than a zero-probability Dropout, so the
        # module tree is the same shape as every other architecture's.
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()
        self.classifier = nn.Linear(self.in_features, num_classes)

        if freeze_backbone:
            self.freeze_backbone()

    # --- the freeze boundary -----------------------------------------------

    def freeze_backbone(self) -> None:
        """Stop the backbone learning, and stop its BatchNorm statistics moving.

        Two separate things, and the second is the one that is easy to miss:

        * `requires_grad_(False)` on every parameter, so backpropagation stops
          at the head and never reaches 11.2M weights it has no business
          updating on a few thousand images.
        * `self.backbone.eval()`, so the BatchNorm layers keep using their
          ImageNet `running_mean` / `running_var` instead of overwriting them
          with this run's batch statistics. Those buffers are not parameters, so
          `requires_grad=False` does not protect them -- only eval mode does.
          Without this the pretrained feature normalisation silently drifts for
          the whole run.

        Safe to call twice, and on a model that was never trained.
        """
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        self.backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        """Make all 11.2M backbone parameters trainable again.

        For the *second* experiment, not the first: progressive (discriminative)
        finetuning, where the late layers adapt to MRI and the early ones keep
        their ImageNet features. Pair it with a learning rate an order of
        magnitude or two below the head's -- one learning rate for 11.2M
        pretrained weights and 2,052 new ones is almost always too large for
        one of the two groups, and the pretrained group is the one that loses.

        This also lifts the `train()` override's hold on the backbone, so its
        BatchNorm layers start updating their running statistics again. That is
        correct here and would have been wrong in `freeze_backbone()`.
        """
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(True)
        self.backbone_frozen = False

    def train(self, mode: bool = True) -> "BrainTumorResNet18":
        """Switch modes, keeping a frozen backbone in eval mode.

        `nn.Module.train(mode)` walks the whole tree and sets every submodule,
        BatchNorm included, so it would undo the `eval()` in `freeze_backbone()`
        on the very first training epoch -- the `src/train.py` loop calls
        `model.train()` at the top of `run_epoch`. Re-asserting eval afterwards
        is what makes the freeze actually hold, and it is why `model.train()`
        cannot simply be trusted to be the last thing that touched the mode.
        """
        super().train(mode)
        if mode and self.backbone_frozen:
            self.backbone.eval()
        return self

    # --- forward ------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of images to class logits.

        Args:
            x: float32 tensor of shape [batch, 3, H, W] already resized and
               normalised by `src/data/transforms.py`. H and W need not be 224:
               everything before `avgpool` is convolutional, and `avgpool` is an
               adaptive pool, so any input size >= 32 produces a [batch, 512]
               feature vector. The transforms still resize to 224 because that
               is what the pretrained weights were trained at.

        Returns:
            float32 tensor of shape [batch, num_classes] -- unnormalised scores,
            higher means more likely. Softmax if you want probabilities.
        """
        features = self.backbone(x)   # [B, 512] frozen features
        features = self.dropout(features)
        # No softmax: `nn.CrossEntropyLoss` applies log-softmax itself, and a
        # softmax here would be applied twice. Same rule as BrainTumorCNN.
        return self.classifier(features)

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters.

        Defaults to trainable-only, which for this architecture is the number
        that matters -- 2,052 for 4 classes, out of 11,178,564. `train.py` logs
        both so the freeze is visible in the run log rather than a claim in a
        docstring.
        """
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad or not trainable_only
        )


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """The parameters an optimizer should be given.

    `model.parameters()` returns everything, frozen or not. Handing that to
    `torch.optim.Adam` is not *wrong* -- Adam skips parameters whose `.grad` is
    None, and a frozen parameter never gets one -- but it makes the freeze
    invisible in the code and easy to undo by accident. Filtering here states the
    intent at the call site, and means a future optimizer that does allocate
    state for every parameter it is handed (AdamW with foreach, say) cannot waste
    11.2M slots of optimizer state on weights that will never move.
    """
    return [p for p in model.parameters() if p.requires_grad]


#: Architectures that have ImageNet weights to load, and therefore accept the
#: `pretrained` flag. `BrainTumorCNN` is absent because it has none, and passing
#: the flag to it would be a `TypeError` naming an argument rather than a
#: meaning.
_PRETRAINED_ARCHITECTURES: frozenset[str] = frozenset({RESNET18_ARCH})


def build_model(
    num_classes: int = NUM_CLASSES,
    name: str = ARCH_NAME,
    *,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """Factory that returns a model by name.

    `src/README.md` reserves this signature so the training and inference code
    can swap architectures without changing. Two architectures are registered,
    and every caller goes through this registry instead of hardcoding a class:

        build_model()                        # ResNet-18, pretrained, frozen
        build_model(4, "custom_cnn")         # the from-scratch baseline
        build_model(4, "resnet18", dropout_p=0.2)
        build_model(4, "resnet18", pretrained=False)   # from a checkpoint

    The last form is the one `src/evaluate.py` uses. Rebuilding from a
    checkpoint overwrites every parameter with the saved ones, so asking for
    ImageNet weights first is a ~45MB download whose result is immediately
    discarded -- and it would put a network call inside `load_checkpoint`, which
    should work offline.

    Args:
        num_classes: how many classes to predict.
        name: architecture key, one of `RESNET18_ARCH` or `CUSTOM_CNN_ARCH`.
        pretrained: load ImageNet weights, where the architecture has any. Applied
            only to architectures in `_PRETRAINED_ARCHITECTURES`; a
            `BrainTumorCNN` silently ignores it, because there is nothing to
            load.
        **kwargs: forwarded to the architecture, e.g. `dropout_p=0.2`,
            `freeze_backbone=False`.

    Returns:
        An `nn.Module` ready for training.

    Raises:
        ValueError: `name` is not a registered architecture.
    """
    architectures: dict[str, type[nn.Module]] = {
        RESNET18_ARCH: BrainTumorResNet18,
        CUSTOM_CNN_ARCH: BrainTumorCNN,
    }

    if name not in architectures:
        raise ValueError(
            f"Unknown architecture {name!r}. "
            f"Available: {sorted(architectures)}"
        )

    if name in _PRETRAINED_ARCHITECTURES and not pretrained:
        kwargs["pretrained"] = False

    return architectures[name](num_classes=num_classes, **kwargs)
