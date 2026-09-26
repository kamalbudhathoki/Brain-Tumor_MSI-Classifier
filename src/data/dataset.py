"""
dataset.py -- turn folders of MRI scans into batches of PyTorch tensors.

WHAT THIS FILE DOES
-------------------
A brain tumor dataset on disk looks like this (one folder per class, which is
the layout the Kaggle "Brain Tumor MRI Dataset" ships):

    data/
    |-- train/
    |   |-- glioma/      image1.jpg, image2.jpg, ...
    |   |-- meningioma/  image1.jpg, ...
    |   |-- pituitary/   image1.jpg, ...
    |   `-- no_tumor/    image1.jpg, ...
    `-- test/
        |-- glioma/
        |-- meningioma/
        |-- pituitary/
        `-- no_tumor/

PyTorch cannot read that layout on its own. It wants two objects:

  1. A **Dataset** -- answers "give me item number 7" with one image + label.
  2. A **DataLoader** -- wraps a Dataset and hands the training loop whole
     batches (say 32 images at a time), optionally shuffled and in parallel.

This module builds both.

THE FOUR PREPROCESSING STEPS (the ones you asked for)
-----------------------------------------------------
Every image goes through the same pipeline, in this order:

  1. **Resize to 224x224.** Real MRI slices are all different sizes. A neural
     network needs one fixed input size, and 224x224 is the size ImageNet
     backbones (ResNet-50, EfficientNet) were pretrained with, so it is the
     natural default.
  2. **ToTensor.** PIL images store pixels as integers 0-255. Tensors store
     floats. `ToTensor()` does the conversion *and* rescales 0-255 down to
     0.0-1.0, which is why you never see a 255 in your tensors.
  3. **Normalize.** After step 2 the average pixel is roughly 0.45, not 0.0.
     Normalizing subtracts the per-channel mean and divides by the per-channel
     standard deviation, so the data is centered at zero with a spread of ~1.
     This is what makes gradients behave and training converge. We reuse the
     official ImageNet mean/std, because that is the statistics the pretrained
     backbone already expects.
  4. **Label.** The class name ("glioma") is turned into an integer (0, 1, 2,
     3) because neural networks predict class *indices*, not words.

A NOTE ON DATA LEAKAGE
----------------------
Splitting by folder is fine while you are learning the mechanics. Before this
dataset is used for a real result, be aware that the medically correct split is
by *patient*, never by slice: every scan belonging to one patient must land in
exactly one of train/val/test, otherwise the model is graded on slices it has
already effectively memorized. See data/README.md.

Usage:

    from src.data.dataset import build_dataloaders

    train_loader, test_loader = build_dataloaders("data/processed", batch_size=32)

    for images, labels in train_loader:
        # images: float32 tensor of shape [32, 3, 224, 224]
        # labels: int64   tensor of shape [32]
        ...
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Library code must never call print() (see src/README.md) -- it goes through a
# logger instead, so notebook output and training logs stay readable.
logger = logging.getLogger(__name__)

# --- Hyperparameters, kept in one place so src/config.py can take them over ---

#: Side length of the square image fed to the model.
IMAGE_SIZE: int = 224

#: Per-channel mean of the ImageNet dataset. Our MRI scans are grayscales
#: replicated to 3 channels, so the "average" image is just a gray pixel.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)

#: Per-channel standard deviation of the ImageNet dataset.
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: Which file suffixes count as images. Anything else in a class folder
#: (e.g. a stray .txt or Thumbs.db on Windows) is ignored.
IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)

#: The two splits this project ships with.
SPLITS: tuple[str, str] = ("train", "test")


def build_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Build the resize -> tensor -> normalize pipeline.

    `transforms.Compose` just means "run these steps one after another".
    The same pipeline is used for training and testing on purpose: any random
    augmentation would have to be added for training only (see the
    `train=True` note in `build_dataloaders`).
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


def find_class_names(split_dir: Path) -> list[str]:
    """Return the class names for a split, read from its subfolder names.

    Sorting matters: it makes the name -> index mapping deterministic, so class
    0 means the same tumor type today as it does on your teammate's machine.
    """
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    # Each immediate subfolder of train/ or test/ is one class.
    names = sorted(entry.name for entry in split_dir.iterdir() if entry.is_dir())

    if not names:
        raise RuntimeError(
            f"No class subfolders inside {split_dir}. Expected one folder per "
            f"class, e.g. {split_dir / 'glioma'}."
        )

    logger.info("Found %d classes in %s: %s", len(names), split_dir, names)
    return names


class BrainTumorDataset(Dataset):
    """A PyTorch Dataset over a `train/` or `test/` folder of class subfolders.

    The `Dataset` protocol is only two methods:

      * `__len__`  -- how many images are in here?
      * `__getitem__` -- give me image number `index` as (tensor, label).

    The DataLoader calls those in a loop to build batches. We also record the
    file list up front (in `__init__`) instead of globbing on every access,
    which keeps training from hammering the disk.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        class_names: list[str] | None = None,
        transform: transforms.Compose | None = None,
    ) -> None:
        """Scan one split folder and remember every image path and its label.

        Args:
            root: folder that contains `train/` and `test/`, e.g. "data/processed".
            split: which subfolder to load, "train" or "test".
            class_names: reuse the training class names instead of re-reading
                them. This is what keeps label 0 meaning the same thing in both
                splits -- if `test/` happened to be missing a class, reading it
                independently would silently shift every later label.
            transform: preprocessing pipeline. Defaults to
                `build_transform()` when not supplied.
        """
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split

        # Either reuse the caller's class list, or discover it from the folders.
        self.class_names = (
            list(class_names) if class_names is not None else find_class_names(self.split_dir)
        )
        # "glioma" -> 0, "meningioma" -> 1, ... The reverse map is handy for
        # turning a predicted index back into a readable label.
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.idx_to_class = {i: name for name, i in self.class_to_idx.items()}

        self.transform = transform if transform is not None else build_transform()
        self.samples: list[tuple[Path, int]] = self._collect_samples()

        if not self.samples:
            raise RuntimeError(
                f"No images with extensions {IMAGE_EXTENSIONS} were found under "
                f"{self.split_dir}."
            )

        logger.info(
            "Loaded %d images from '%s' (%d classes: %s)",
            len(self.samples),
            self.split_dir,
            len(self.class_names),
            ", ".join(self.class_names),
        )

    def _collect_samples(self) -> list[tuple[Path, int]]:
        """Walk the class folders and build the (path, label index) list."""
        samples: list[tuple[Path, int]] = []

        for class_name in self.class_names:
            class_dir = self.split_dir / class_name

            if not class_dir.is_dir():
                # A missing training class is a hard error; a missing *test*
                # class is not, so warn and keep going.
                logger.warning("Class folder missing: %s", class_dir)
                continue

            label_idx = self.class_to_idx[class_name]
            class_images = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            samples.extend((path, label_idx) for path in class_images)

        return samples

    def __len__(self) -> int:
        """Number of images. The DataLoader uses this to know when to stop."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return one training example: (image tensor, label index).

        This runs once per image per epoch, so it is on the hot path -- keep it
        to the minimum work needed to produce one sample.
        """
        image_path, label = self.samples[index]

        # MRI slices are usually single-channel (grayscale). `convert("RGB")`
        # copies that one channel into three, which is what an ImageNet
        # pretrained backbone expects, and it also flattens the awkward
        # palette/mode variations that PNG and DICOM-derived images bring.
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        # This is where resize + ToTensor + Normalize all happen.
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def label_counts(self) -> dict[str, int]:
        """Return images-per-class, handy for spotting an imbalanced split."""
        counts: dict[str, int] = {name: 0 for name in self.class_names}
        for _, label_idx in self.samples:
            counts[self.idx_to_class[label_idx]] += 1
        return counts


def build_dataloaders(
    root: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = IMAGE_SIZE,
) -> tuple[DataLoader, DataLoader]:
    """Create the training and test DataLoaders.

    Args:
        root: folder containing `train/` and `test/`.
        batch_size: images per step. 32 is a safe starting point on a laptop GPU.
        num_workers: parallel loader processes. 0 means loading happens in the
            main process, which is simplest to debug. Raise it (2-4) to overlap
            disk reads with compute, but note that on Windows anything above 0
            requires guarding your entry point with
            `if __name__ == "__main__":` or you will hit a multiprocessing error.
        image_size: side length for the resize step.

    Returns:
        (train_loader, test_loader).
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    # Read the class list once, from train/, and hand the identical mapping to
    # test so that label indices line up across both splits.
    train_dataset = BrainTumorDataset(
        root=root,
        split="train",
        transform=build_transform(image_size),
    )
    test_dataset = BrainTumorDataset(
        root=root,
        split="test",
        class_names=train_dataset.class_names,
        transform=build_transform(image_size),
    )

    # `shuffle=True` on train so the model cannot memorize scan order, e.g. by
    # learning "early epoch files are always glioma". Never shuffle test: you
    # want predictions in a fixed, reproducible order when you score it.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    logger.info(
        "Dataloaders ready: %d train batches, %d test batches (batch_size=%d). "
        "Train class counts: %s",
        len(train_loader),
        len(test_loader),
        batch_size,
        train_dataset.label_counts(),
    )
    return train_loader, test_loader
