"""
transforms.py -- the image preprocessing pipelines, one for training and one for
evaluating.

WHAT THIS FILE DOES
-------------------
A `transforms.Compose` is just a list of image operations run in order. This module
defines two of them:

  * `build_train_transform()`  -- random augmentation, then resize/tensor/normalize.
  * `build_eval_transform()`   -- resize/tensor/normalize only, fully deterministic.

Keeping the two separate is not a style preference. Augmenting the test set would make
your test score change every time you ran it, so you could never tell whether a model
got better or you just rolled different dice. Validation and test must see the exact
same preprocessing every single time.

    from src.data.dataset import build_dataloaders

    # augmented training, clean evaluation (the normal setup)
    train_loader, test_loader = build_dataloaders("data/processed", augment=True)

    # no augmentation at all -- the baseline you compare against
    train_loader, test_loader = build_dataloaders("data/processed", augment=False)

    for images, labels in train_loader:
        ...


WHY AUGMENTATION REDUCES OVERFITTING
------------------------------------
An MRI dataset is small -- typically a few thousand slices -- while the model we want
to use (an ImageNet-pretrained ResNet-50 or EfficientNet) has tens of millions of
parameters. That is a losing ratio. Given only a few thousand images, the model can
drive its training loss toward zero *memorising the exact pixels it was shown*:

    training accuracy 99%,  test accuracy 61%   <- the gap is overfitting

The model has not learned "this texture means glioma". It has learned the specific
arrangement of pixels in scan 0042.jpg, which is noise, not signal.

Augmentation attacks that directly. Each random transform generates a *different but
equally valid* version of every image, so:

  1. **The dataset effectively gets bigger.** Ten variants of one scan is closer to ten
     examples than to one, which matters when you only have thousands to begin with.
  2. **The model is pushed off the exact pixels.** A horizontal flip puts the tumour on
     the other side of the skull. No memorised patch of pixels survives that, so
     memorisation stops being a viable shortcut and real feature learning becomes the
     cheaper path.
  3. **It encodes invariances we already believe in.** We *know* a glioma is still a
     glioma if the scan is tilted 10 degrees or shifted slightly. Encoding that as a
     built-in prior means the model never wastes capacity learning it, and that
     capacity goes to genuinely useful features instead.
  4. **It is a regulariser.** The model is forced to find features that are stable
     across a whole family of transformations, which are far more likely to be the real
     diagnostic ones (shape, texture, intensity distribution) than brittle
     pixel-exact cues.

The visible effect is that the train/test accuracy gap narrows and test accuracy rises.
The hidden effect, and the one that matters more, is that the model stops leaning on
coincidences specific to this dataset -- scanner vendor, slice thickness, one hospital
-- and those are exactly the shortcuts that collapse on data from a new hospital.

A WARNING SPECIFIC TO MEDICAL IMAGES
------------------------------------
Augmentation is a bias-variance trade, not a free win. Every transform is a bet that
the label is *unchanged* by that transformation. Break that bet and you are actively
injecting wrong labels:

  * Do **not** flip left/right if your task depends on anatomical side.
  * Do **not** rotate by 180 degrees -- that puts the brain upside down.
  * Do **not** use strong colour jitter -- tumour contrast is diagnostic signal.
  * Do **not** zoom in aggressively and crop the lesion out of frame.

So the defaults here are deliberately mild. Treat them as a starting point, and measure
them: `notebooks/03_augmentation_ablation.ipynb` is the place to check whether each
transform actually earned its place.
"""

from __future__ import annotations

from torchvision import transforms
from torchvision.transforms import InterpolationMode

# --- Hyperparameters, kept in one place so src/config.py can take them over ---

#: Side length of the square image fed to the model.
IMAGE_SIZE: int = 224

#: Per-channel mean of the ImageNet dataset. Our MRI scans are grayscales
#: replicated to 3 channels, so the "average" image is just a gray pixel.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)

#: Per-channel standard deviation of the ImageNet dataset.
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


# How the geometric transforms interpolate when they have to invent new pixels.
#
# Both RandomRotation and RandomAffine default to NEAREST, which copies pixels
# verbatim and leaves a jagged, pixelated staircase along every rotated edge. On
# MRI slices, where the whole image is smooth grey gradients and the lesion is
# defined by soft intensity boundaries, that aliasing is genuinely destructive
# signal loss. BILINEAR is the correct default here.
INTERPOLATION = InterpolationMode.BILINEAR

#: Value used to fill the corners exposed by rotation/translation. 0 is black in
#: PIL's 0-255 space, and black is a colour MRI scans already contain, so the
#: model has seen it. After normalization it lands near -2.1, which is also where
#: a genuinely black pixel lands, so the fill is in-distribution.
FILL = 0


def build_eval_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Build the deterministic resize -> tensor -> normalize pipeline.

    No randomness of any kind: the same file always produces the exact same
    tensor. This is what `test` (and any validation split) must use.
    """
    return transforms.Compose(
        [
            # Step 1: force every scan to the same size. `antialias` avoids the
            # jagged edges you get when downsampling MRI detail.
            transforms.Resize((image_size, image_size), antialias=True),
            # Step 2: PIL image -> float32 tensor, scaled from 0-255 to 0.0-1.0,
            # shaped [channels, height, width] (PyTorch puts channels first).
            transforms.ToTensor(),
            # Step 3: center the data and give it a standard spread of 1.
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_train_transform(
    image_size: int = IMAGE_SIZE,
    hflip_prob: float = 0.5,
    rotation_degrees: float = 15.0,
    translate: tuple[float, float] = (0.08, 0.08),
    scale: tuple[float, float] = (0.9, 1.1),
    shear_degrees: float = 8.0,
    jitter_brightness: float = 0.2,
    jitter_contrast: float = 0.2,
    jitter_saturation: float = 0.0,
    jitter_hue: float = 0.0,
) -> transforms.Compose:
    """Build the augmented training pipeline.

    Read the steps as two groups:

      * **Geometric** (flip / rotate / affine) -- move the anatomy around.
      * **Photometric** (colour jitter) -- change the intensities.

    ORDER MATTERS, twice over:

    1. All geometry runs *before* the resize, on the original-resolution PIL image.
       Resizing first and then rotating would resample an already-downsampled
       image, so every pixel would be interpolated twice and the lesion edges
       would blur. Doing geometry first means `Resize` is the only step that
       throws information away.
    2. `ToTensor` and `Normalize` always come last, because they need a PIL
       image in front of them and because augmentation in 0-255 pixel space is
       far easier to reason about than in normalized space.

    Args:
        image_size: side length of the square output.
        hflip_prob: chance of mirroring each image. 0.5 is the standard choice --
            it is symmetric, so the augmented set stays balanced.
        rotation_degrees: +/- range in degrees. Kept mild: a large rotation
            produces anatomy that does not correspond to a real head position.
        translate: +/- fraction of image size to shift (x, y).
        scale: zoom range; below 1.0 zooms in, above 1.0 zooms out.
        shear_degrees: +/- skew in degrees.
        jitter_brightness: +/- fraction of brightness change.
        jitter_contrast: +/- fraction of contrast change.
        jitter_saturation: +/- fraction of saturation change. **0.0 by default** --
            our scans are grayscale replicated to 3 channels, and changing the
            saturation of a grey image is mathematically a no-op. Raise it only if
            the pipeline ever receives genuinely colour images (e.g. fused
            T1/T2 overlays).
        jitter_hue: +/- hue shift. **0.0 by default**, same reasoning as
            `jitter_saturation`.

    Note on `RandomAffine`: it takes its own `degrees` argument, which would
    overlap with `RandomRotation`. We pass `degrees=0` so each transform has one
    clear job -- rotation handles angle, affine handles translate/scale/shear --
    and the two magnitudes do not silently compound.
    """
    return transforms.Compose(
        [
            # --- Group 1: geometry, on the original-resolution PIL image ----

            # Mirror left<->right. The brain is roughly symmetric and the label
            # (tumour *type*) does not depend on which side the lesion is on, so
            # this is a free, label-preserving symmetry.
            transforms.RandomHorizontalFlip(p=hflip_prob),

            # Tilt the head. Real scans are never perfectly axis-aligned, and this
            # also stops the model from keying on slice position in the frame.
            transforms.RandomRotation(
                degrees=rotation_degrees,
                interpolation=INTERPOLATION,
                fill=FILL,
            ),

            # Shift, zoom and skew. This is the transform that most directly
            # breaks "memorise where the tumour sits in this exact frame".
            transforms.RandomAffine(
                degrees=0,  # rotation is RandomRotation's job -- see note above
                translate=translate,
                scale=scale,
                shear=shear_degrees,
                interpolation=INTERPOLATION,
                fill=FILL,
            ),

            # --- Group 2: photometry, still on the PIL image ---------------

            # Change brightness and contrast to mimic scanner and protocol
            # differences between hospitals -- the kind of shift that otherwise
            # makes a model fail on data from a site it never saw.
            transforms.ColorJitter(
                brightness=jitter_brightness,
                contrast=jitter_contrast,
                saturation=jitter_saturation,
                hue=jitter_hue,
            ),

            # --- Group 3: the shared, deterministic tail -------------------
            # Identical to build_eval_transform(), on purpose: the only difference
            # between train and eval must be the random steps above.
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
