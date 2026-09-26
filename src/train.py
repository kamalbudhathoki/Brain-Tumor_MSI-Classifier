"""
train.py -- the training loop for the brain tumor MRI classifier.

WHAT THIS FILE DOES
-------------------
One command trains a model and leaves a self-describing checkpoint behind:

    python -m src.train --data-root data/processed --epochs 20

The pipeline, in order:

    BrainTumorDataset  ->  DataLoader  ->  model  ->  CrossEntropyLoss
                                             |
                                        Adam optimizer
                                             |
                                   train loop + val loop
                                             |
                                        best.pt

THE VALIDATION SPLIT -- READ THIS FIRST
---------------------------------------
`data/` ships two folders, `train/` and `test/`. There is no `val/`.

It is tempting to validate on `test/` and keep whichever epoch scores best. Do not
do that. Choosing the checkpoint by its test score makes the final test number
optimistically biased -- you have effectively trained on the test set by selecting
against it, and the reported accuracy will not survive contact with real data.

So this script carves a **stratified** holdout out of `train/` and uses that for
model selection. `test/` is touched exactly once, at the very end, with the
already-chosen best checkpoint. The proportions of every class are preserved
across the split, so a rare class does not end up with zero validation examples.

Caveat worth knowing: this split is by *slice*, not by *patient*. If two slices
come from the same patient they can still land on opposite sides of the
train/val boundary, which inflates validation accuracy. `data/README.md` calls
patient-level splitting the correct approach; once `data/splits/manifest.csv`
exists with a `patient_id` column, that is where you should switch.

Also note what is deliberately NOT here: mixed-precision (AMP) and LR
scheduling. Both belong in this module per the project plan, and both are worth
adding, but neither is needed to get a correct first number. Adding them before
the baseline works makes it harder to tell what helped.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from .data.dataset import BrainTumorDataset
from .data.transforms import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_eval_transform,
    build_train_transform,
)
from .models.classifier import ARCH_NAME, NUM_CLASSES, build_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Every hyperparameter for one training run.

    A dataclass rather than loose argparse namespaces because the whole thing gets
    written into the checkpoint: a run has to be reproducible from the artifact it
    produced, and that only works if the config is captured automatically.

    `src/config.py` will eventually own this; keeping it local until then means
    nothing here depends on a module that does not exist yet.
    """

    # --- data ---
    data_root: Path = Path("data/processed")
    val_fraction: float = 0.2
    image_size: int = IMAGE_SIZE
    num_classes: int = NUM_CLASSES

    # --- model ---
    arch: str = ARCH_NAME

    # --- optimisation ---
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    augment: bool = True
    class_weighted: bool = False
    early_stopping_patience: int | None = 5
    early_stopping_min_delta: float = 0.0

    # --- runtime ---
    num_workers: int = 0
    device: str = "auto"
    seed: int = 42
    show_progress: bool = True

    # --- output ---
    output_root: Path = Path("models")
    run_name: str | None = None

    def validate(self) -> None:
        """Fail early and loudly on nonsense, before spending GPU hours."""
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1), got {self.val_fraction}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError(
                f"early_stopping_patience must be >= 1 or None, "
                f"got {self.early_stopping_patience}"
            )
        if self.early_stopping_min_delta < 0:
            raise ValueError(
                f"early_stopping_min_delta must be >= 0, "
                f"got {self.early_stopping_min_delta}"
            )


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Make the run reproducible.

    Seeds all three RNGs because all three get used: `random` and `numpy` inside
    the dataset split, `torch` inside the transform pipeline and weight
    initialisation.

    The cuDNN flags have to be set together. `deterministic=True` picks algorithms
    that give bit-identical results run to run, and `benchmark=False` stops cuDNN
    from auto-tuning for your specific hardware -- autotuning picks different
    algorithms depending on what is free that moment, which breaks reproducibility.
    The cost is that training is somewhat slower.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(spec: str) -> torch.device:
    """Turn "auto" into an actual device, and be honest when falling back."""
    if spec != "auto":
        return torch.device(spec)

    if torch.cuda.is_available():
        return torch.device("cuda")

    # Apple's MPS backend is a large speedup over CPU for convolutions. Worth
    # checking for, and worth a clear log line when it is missing, because
    # "why is training so slow" is otherwise a confusing question to answer.
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")

    logger.warning("No GPU detected (cuda/mps unavailable) -- training on CPU.")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def stratified_split(
    dataset: BrainTumorDataset,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split a dataset's indices into train/val, preserving class proportions.

    `torch.utils.data.random_split` is not usable here: it shuffles uniformly, so
    a class holding 5% of the data gets 5% of validation too, which on a small
    dataset can round to zero examples and make that class unmeasurable.

    Returns:
        (train_indices, val_indices) as lists, ready for `Subset`.
    """
    indices = np.arange(len(dataset))
    labels = np.array([label for _, label in dataset.samples])

    # Stratification needs at least 2 members per class and a val slice of at
    # least 1 per class. sklearn raises a fairly opaque error when that fails, so
    # check it ourselves and say what is actually wrong.
    counts = np.bincount(labels, minlength=len(dataset.class_names))
    smallest = int(counts[counts > 0].min()) if (counts > 0).any() else 0
    if smallest < 2:
        rare = [
            dataset.idx_to_class[i]
            for i, c in enumerate(counts)
            if c == 1
        ]
        raise ValueError(
            f"Cannot stratify: class(es) {rare} have a single image. Stratified "
            "splitting needs >= 2 per class. Either add more data or pass "
            "--val-fraction with a non-stratified fallback."
        )

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
    )
    return train_idx.tolist(), val_idx.tolist()


@dataclass
class DataBundle:
    """The three loaders plus the metadata the training loop needs.

    Returned as one object rather than a five-element tuple, because the fifth
    element -- the training labels -- is easy to drop or reorder at the call
    site, and it is needed to compute class weights.
    """

    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_names: list[str]
    train_labels: list[int]


def build_loaders(cfg: TrainConfig) -> DataBundle:
    """Create the train, validation and test DataLoaders.

    Two `BrainTumorDataset` objects are pointed at the same `train/` folder: one
    carrying the augmented pipeline, one carrying the deterministic pipeline.
    `Subset` then hands each of them a different slice of the indices. Doing it
    this way means validation images are never augmented, which would make the
    validation score meaningless.
    """
    # Discovered from train/ and reused everywhere, so a label index means the
    # same class in all three loaders.
    train_full = BrainTumorDataset(
        root=cfg.data_root,
        split="train",
        transform=build_train_transform(cfg.image_size) if cfg.augment
        else build_eval_transform(cfg.image_size),
    )
    class_names = train_full.class_names

    train_idx, val_idx = stratified_split(train_full, cfg.val_fraction, cfg.seed)

    val_source = BrainTumorDataset(
        root=cfg.data_root,
        split="train",
        class_names=class_names,
        transform=build_eval_transform(cfg.image_size),
    )
    test_dataset = BrainTumorDataset(
        root=cfg.data_root,
        split="test",
        class_names=class_names,
        transform=build_eval_transform(cfg.image_size),
    )

    train_loader = _make_loader(
        Subset(train_full, train_idx),
        cfg,
        shuffle=True,
    )
    val_loader = _make_loader(
        Subset(val_source, val_idx),
        cfg,
        shuffle=False,
    )
    test_loader = _make_loader(test_dataset, cfg, shuffle=False)

    # Labels of the *training* half only. Weighting by the whole dataset would
    # leak validation class frequencies into the loss.
    all_labels = [label for _, label in train_full.samples]
    train_labels = [all_labels[i] for i in train_idx]

    logger.info(
        "Split: %d train / %d val / %d test images across classes %s",
        len(train_idx),
        len(val_idx),
        len(test_dataset),
        ", ".join(class_names),
    )
    return DataBundle(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        class_names=class_names,
        train_labels=train_labels,
    )


def _make_loader(
    dataset: Dataset,
    cfg: TrainConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    """Build one DataLoader with the settings that matter for reproducibility.

    * `generator` is seeded so the shuffle order is the same for a given seed.
      Without it, resuming a run continues a different shuffle sequence.
    * `pin_memory` lets the CPU hand tensors to the GPU through pinned memory,
      which is meaningfully faster. Only useful when a GPU is present.
    * `persistent_workers` keeps loader processes alive between epochs instead of
      respawning them, saving a second or two per epoch. It is invalid with
      `num_workers=0`, hence the guard.
    """
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
        persistent_workers=cfg.num_workers > 0,
    )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def build_criterion(
    cfg: TrainConfig,
    train_labels: Iterable[int],
    class_names: list[str],
    device: torch.device,
) -> nn.CrossEntropyLoss:
    """Build CrossEntropyLoss, optionally weighting rare classes.

    `CrossEntropyLoss` takes **raw logits**, not probabilities, and applies
    log-softmax internally. `BrainTumorCNN` already ends in a bare `Linear`, which
    is exactly right -- adding a `Softmax` in the model would apply it twice.

    The dataset is imbalanced (see `notebooks/01_data_exploration.ipynb`), which
    means the loss is dominated by whichever class is most common. `class_weighted`
    counteracts that by scaling each class's contribution by the inverse of its
    frequency, so a glioma costs roughly the same total loss as a `no_tumor`.

    One subtlety: the weights normalise a *per-batch average*, so a batch that
    happens to contain no rare images still reports a small loss. That makes
    weighted training loss a worse progress signal than accuracy -- watch the
    accuracy and the macro metrics, not the loss curve.
    """
    if not cfg.class_weighted:
        return nn.CrossEntropyLoss()

    counts = np.bincount(np.asarray(list(train_labels)), minlength=len(class_names))
    # Guard against a class with zero training examples, which would divide by 0.
    safe = np.maximum(counts, 1)
    weights = counts.sum() / (len(class_names) * safe)

    logger.info(
        "Class weights: %s",
        {name: round(float(w), 3) for name, w in zip(class_names, weights)},
    )
    return nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Stop training when the monitored validation metric stops improving.

    WHY EARLY STOPPING PREVENTS OVERFITTING
    ---------------------------------------
    Every epoch, gradient descent optimises the loss on the *training* set. Left
    alone it will happily drive that number toward zero, because a network with
    millions of parameters can memorise a finite dataset. Training loss falling
    forever is therefore not evidence of learning -- past a point it is evidence
    of memorisation.

    The validation set is the tell. It is data the model has never trained on, so
    it measures generalisation rather than recall. The two curves diverge:

        epoch   train loss   val loss     what is happening
        -----   ----------   -------     ---------------------------------
        1        0.85         0.80        both falling: real learning
        5        0.42         0.45        still learning, but slowing
        9        0.21         0.44        training loss keeps falling...
        14       0.06         0.52        ...while val loss climbs: memorising
        19       0.01         0.61        the training set, nothing more

    From epoch ~9 onward, each additional epoch makes the model *worse at the
    actual job*. The minimum of the validation loss curve is the point where
    generalisation peaked, and continuing past it only moves the weights further
    from that peak.

    So early stopping does two things at once:

      1. **It stops the damage.** Training halts while the model is still near its
         best, instead of running to a memorised epoch 19.
      2. **It is a model-selection rule.** Because it keeps the weights from the
         best epoch rather than the last, the artifact you ship is the best
         generalising one the run produced. Restoring those weights at the end is
         not optional -- without it, stopping early would leave you evaluating the
         final (worst) epoch.

    It is regularisation in the truest sense: an implicit constraint on *how long*
    you may fit the data, and one that needs no extra loss term and no extra data.
    On a dataset this small it is frequently worth more than dropout and colour
    jitter combined, because those shrink the model's capacity while early
    stopping simply refuses to let it use the capacity it has for the wrong thing.

    It is a heuristic, not a guarantee. A validation set that is too small is
    noisy, and a lucky dip can stop a run several epochs early -- which is what
    `min_delta` exists to blunt, and why the patience is 5 rather than 1.

    Args:
        patience: epochs without improvement tolerated before stopping. `None`
            disables early stopping entirely and runs all epochs.
        mode: `"min"` for a quantity that should decrease (validation **loss**,
            which is what this project monitors), `"max"` for one that should
            increase (validation accuracy).
        min_delta: smallest change that still counts as an improvement. Raising
            it above 0 stops the run from restarting on a new "best" that is
            within noise of the old one.

    Note this class deliberately has no torch dependency, so the stopping logic
    can be tested on its own without a model, a dataset, or a GPU.
    """

    def __init__(
        self,
        patience: int | None = 5,
        mode: str = "min",
        min_delta: float = 0.0,
    ) -> None:
        if patience is not None and patience < 1:
            raise ValueError(f"patience must be >= 1 or None, got {patience}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")

        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best: float | None = None
        self.best_epoch: int | None = None
        self.num_bad_epochs = 0

    def _is_improvement(self, value: float) -> bool:
        """Is `value` better than the best seen so far?"""
        if self.best is None:
            return True  # the first epoch always sets the bar
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float, epoch: int | None = None) -> bool:
        """Record one epoch's metric. Returns True if it is a new best.

        Note that `best` is only updated on a genuine improvement, so a worse
        epoch never destroys the high-water mark.
        """
        improved = self._is_improvement(value)

        if improved:
            self.best = value
            self.best_epoch = epoch
            # Patience resets only on real progress, so five consecutive
            # non-improvements are required -- not five bad epochs spread across
            # a run that kept improving in between.
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        return improved

    @property
    def should_stop(self) -> bool:
        """Derived, never stored, so it cannot go stale relative to the counter."""
        return self.patience is not None and self.num_bad_epochs >= self.patience

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, mode={self.mode!r}, "
            f"min_delta={self.min_delta}, best={self.best}, "
            f"best_epoch={self.best_epoch}, num_bad_epochs={self.num_bad_epochs})"
        )


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str = "",
    show_progress: bool = True,
) -> dict[str, float]:
    """Run one pass over `loader`, training if an optimizer is supplied.

    One function for both loops because they are 95% identical, and duplicated
    training/validation code drifts out of sync in ways that are hard to spot --
    the classic bug being a forgotten `model.eval()`, which leaves dropout and
    batchnorm active and quietly corrupts the validation number.

    Args:
        optimizer: if given, this is a training pass. If None, an evaluation pass
            with gradients disabled.

    Returns:
        {"loss": float, "accuracy": float}
    """
    is_training = optimizer is not None

    # This single line is why the two loops share code: dropout and (in a real
    # model) batchnorm behave differently in the two modes.
    model.train() if is_training else model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    # set_grad_enabled(False) is the memory-efficient way to run validation: no
    # graph is built, so activation memory stays flat instead of accumulating.
    with torch.set_grad_enabled(is_training):
        bar = tqdm(
            loader,
            desc=desc,
            leave=False,
            disable=not show_progress,
            ncols=88,
        )
        for images, labels in bar:
            # non_blocking=True only means anything alongside pin_memory=True.
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_training:
                # set_to_none=True is the modern form of zero_grad(). It sets the
                # gradients to None instead of zeroing them in place, which lets
                # Adam allocate fresh memory instead of writing zeros into the
                # existing buffer every step.
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_total += batch_size

            # Live readout on the progress bar itself, rather than a print per
            # batch, which would flood the terminal and slow training down.
            bar.set_postfix(
                loss=f"{running_loss / running_total:.3f}",
                acc=f"{running_correct / running_total:.3f}",
            )

    return {
        "loss": running_loss / max(running_total, 1),
        "accuracy": running_correct / max(running_total, 1),
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: nn.Module,
    cfg: TrainConfig,
    class_names: list[str],
    epoch: int,
    metrics: dict[str, float],
) -> None:
    """Write a self-describing checkpoint.

    `models/README.md` is explicit that a `.pt` must carry more than weights. The
    reason is practical: at inference time, `src/inference.py` must reproduce the
    exact preprocessing the model was trained with. If a checkpoint cannot answer
    "what image size, what normalisation, which classes, in which order", then
    inference has to guess -- and a wrong guess produces a confident, wrong
    prediction instead of an error.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            # Everything needed to rebuild the model and its preprocessing.
            "arch": cfg.arch,
            "num_classes": cfg.num_classes,
            "class_names": class_names,
            "image_size": cfg.image_size,
            "norm_mean": list(IMAGENET_MEAN),
            "norm_std": list(IMAGENET_STD),
            "augment": cfg.augment,
            # Provenance.
            "epoch": epoch,
            "seed": cfg.seed,
            "metrics": metrics,
            "config": asdict(cfg),
        },
        path,
    )
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def train(cfg: TrainConfig) -> Path:
    """Run the full training job and return the path to the best checkpoint."""
    cfg.validate()
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    logger.info("Device: %s", device)
    logger.info("Config: %s", asdict(cfg))

    data = build_loaders(cfg)
    class_names = data.class_names

    model = build_model(num_classes=cfg.num_classes, name=cfg.arch).to(device)
    logger.info(
        "Model: %s -- %s parameters", cfg.arch, f"{model.num_parameters():,}"
    )

    # Adam rather than SGD: with a handful of epochs on a small dataset, Adam's
    # per-parameter adaptive step size converges far faster. Note that Adam's
    # `weight_decay` is L2 added to the gradient, not the decoupled version in
    # AdamW -- close enough for regularisation here, but prefer
    # `torch.optim.AdamW` if you want the decoupled behaviour.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    criterion = build_criterion(cfg, data.train_labels, class_names, device)

    # One directory per experiment. `models/README.md` forbids overwriting
    # best.pt, because a run that improves on paper but overwrites the checkpoint
    # that produced your last reported number is a trap you cannot detect later.
    run_id = cfg.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.output_root / f"{cfg.arch}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", run_dir)

    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    # Monitors validation LOSS, not accuracy. Loss is the smoother signal: it
    # moves every epoch, whereas accuracy on a few hundred validation images is a
    # step function that can sit flat for five epochs and then jump, which makes a
    # patience window on accuracy either too twitchy or too slow depending on the
    # split size. The tradeoff is that the epoch chosen for lowest loss is not
    # necessarily the epoch with the highest accuracy -- with heavy class
    # weighting the two can disagree, which is why both are logged below.
    stopper = EarlyStopping(
        patience=cfg.early_stopping_patience,
        mode="min",
        min_delta=cfg.early_stopping_min_delta,
    )

    best_record: dict[str, float] | None = None
    history: list[dict[str, float]] = []
    started = time.time()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()

        train_metrics = run_epoch(
            model,
            data.train,
            criterion,
            device,
            optimizer=optimizer,
            desc=f"epoch {epoch:>3}/{cfg.epochs} [train]",
            show_progress=cfg.show_progress,
        )
        val_metrics = run_epoch(
            model,
            data.val,
            criterion,
            device,
            optimizer=None,  # no optimizer => evaluation pass
            desc=f"epoch {epoch:>3}/{cfg.epochs} [val]  ",
            show_progress=cfg.show_progress,
        )

        # A new best validation loss: save the weights and reset the patience.
        improved = stopper.step(val_metrics["loss"], epoch)

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "seconds": round(time.time() - epoch_start, 1),
            "is_best": improved,
        }
        history.append(record)
        if improved:
            best_record = record

        logger.info(
            "epoch %d/%d  train loss %.4f acc %.4f  |  val loss %.4f acc %.4f  |  %.1fs%s",
            epoch, cfg.epochs,
            train_metrics["loss"], train_metrics["accuracy"],
            val_metrics["loss"], val_metrics["accuracy"],
            record["seconds"],
            "  <- best" if improved else "",
        )

        save_checkpoint(
            last_path, model, cfg, class_names, epoch,
            {"train": train_metrics, "val": val_metrics},
        )
        if improved:
            save_checkpoint(
                best_path, model, cfg, class_names, epoch,
                {"train": train_metrics, "val": val_metrics},
            )

        # Patience exhausted: the validation loss has not improved for
        # `early_stopping_patience` consecutive epochs, so generalisation has
        # stopped getting better and further epochs will only overfit harder.
        if stopper.should_stop:
            logger.info(
                "Early stopping at epoch %d: validation loss has not improved for "
                "%d consecutive epochs (best %.4f at epoch %s).",
                epoch,
                stopper.num_bad_epochs,
                stopper.best,
                stopper.best_epoch,
            )
            break

    total_seconds = time.time() - started

    if best_record is None:
        # Unreachable in practice -- epoch 1 always sets a best -- but a bare
        # `None` subscript here would be a mystifying crash.
        raise RuntimeError("No epoch completed; best_record was never set.")

    # Roll back to the weights from the best epoch. Without this, early stopping
    # would leave you evaluating the *last* epoch, which is by definition worse
    # than the best one -- the whole point of stopping would be lost.
    logger.info("Restoring best weights from epoch %d", stopper.best_epoch)
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["state_dict"])


    # Now, and only now, look at the test set.
    test_metrics = run_epoch(
        model,
        data.test,
        criterion,
        device,
        optimizer=None,
        desc="final [test]   ",
        show_progress=cfg.show_progress,
    )
    logger.info(
        "TEST  loss %.4f  accuracy %.4f   "
        "(selected on best val loss %.4f at epoch %d, val acc there %.4f)",
        test_metrics["loss"],
        test_metrics["accuracy"],
        best_record["val_loss"],
        int(best_record["epoch"]),
        best_record["val_acc"],
    )

    # models/README.md calls this config.yaml. Written as JSON because PyYAML is
    # still in the optional section of requirements.txt, and adding a dependency
    # for one dump is not worth it. Swap in yaml.safe_dump when PyYAML lands.
    (run_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "history": history,
                # `selection_metric` records what early stopping watched, so a
                # metrics.json from a run that monitored accuracy instead of loss
                # is not silently comparable with this one.
                "selection_metric": "val_loss",
                "best_epoch": stopper.best_epoch,
                "best_val_loss": stopper.best,
                "best_epoch_val_accuracy": best_record["val_acc"],
                "epochs_run": len(history),
                "stopped_early": stopper.should_stop,
                "test": test_metrics,
                "total_seconds": round(total_seconds, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Done in %.1fs. Best checkpoint: %s", total_seconds, best_path)
    return best_path


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    """Build a TrainConfig from the command line, falling back to the defaults."""
    parser = argparse.ArgumentParser(
        description="Train the brain tumor MRI classifier.",
    )
    defaults = TrainConfig()

    parser.add_argument("--data-root", type=Path, default=defaults.data_root,
                        help="folder containing train/ and test/ (default: %(default)s)")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--val-fraction", type=float, default=defaults.val_fraction,
                        help="fraction of train/ held out for validation")
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers,
                        help="loader subprocesses; >0 requires the __main__ guard on Windows")
    parser.add_argument("--num-classes", type=int, default=defaults.num_classes)
    parser.add_argument("--arch", type=str, default=defaults.arch)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", type=str, default=defaults.device,
                        help="'auto', 'cpu', 'cuda', or 'cuda:0'")
    parser.add_argument("--output-root", type=Path, default=defaults.output_root)
    parser.add_argument("--run-name", type=str, default=defaults.run_name,
                        help="experiment folder name; defaults to a timestamp")
    # The three boolean flags below pass `default=defaults.<field>` explicitly.
    # Without it argparse silently wins: `store_const` and `store_false` both
    # default to None/True, which would override whatever TrainConfig says and
    # make the dataclass the wrong place to change a default. Threading the
    # default through keeps the dataclass the single source of truth.
    parser.add_argument("--no-augment", dest="augment", action="store_false",
                        default=defaults.augment,
                        help="disable training augmentation (for ablations)")
    parser.add_argument("--class-weighted", action="store_true",
                        default=defaults.class_weighted,
                        help="weight CrossEntropyLoss by inverse class frequency")
    parser.add_argument("--no-early-stopping", dest="early_stopping_patience",
                        action="store_const", const=None,
                        default=defaults.early_stopping_patience,
                        help="always run all --epochs")
    parser.add_argument("--early-stopping-min-delta", type=float,
                        default=defaults.early_stopping_min_delta,
                        help="min val-loss decrease that counts as an improvement")
    parser.add_argument("--no-progress", dest="show_progress", action="store_false",
                        default=defaults.show_progress)

    args = parser.parse_args(argv)

    cfg = TrainConfig(
        data_root=args.data_root,
        val_fraction=args.val_fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        num_classes=args.num_classes,
        arch=args.arch,
        seed=args.seed,
        device=args.device,
        output_root=args.output_root,
        run_name=args.run_name,
        augment=args.augment,
        class_weighted=args.class_weighted,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        show_progress=args.show_progress,
    )
    return cfg


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    train(parse_args(argv))


# The __main__ guard is not optional on Windows. DataLoader workers are spawned
# as fresh processes that re-import this module; without the guard each of them
# re-runs main() and you get one training run per CPU core, or a
# "RuntimeError: An attempt has been made to start a new process before the
# current process has finished its bootstrapping phase" error.
if __name__ == "__main__":
    main()
