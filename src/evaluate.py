"""
evaluate.py -- score a trained checkpoint on a held-out split.

WHAT THIS FILE DOES
-------------------
`src/train.py` trains a model and writes a self-describing checkpoint. This module
takes that checkpoint, runs it over a split it has never been selected on, and
reports the four numbers that decide whether the model is any good:

    Accuracy, Precision, Recall, F1, and the Confusion Matrix.

    python -m src.evaluate
    python -m src.evaluate --checkpoint models/<run>/best.pt --split test
    python -m src.evaluate --save-report outputs/eval/test_report.json \
                           --save-confusion-matrix outputs/eval/cm.png

The pipeline is four steps, and each one is a plain function you can call on its own:

    collect_predictions()  model + DataLoader  ->  y_true, y_pred, y_prob
    compute_metrics()      y_true, y_pred      ->  ClassificationMetrics
    render_report()        metrics             ->  a formatted text report
    plot_confusion_matrix  metrics             ->  a PNG (optional)

`compute_metrics()` deliberately takes numpy arrays and knows nothing about torch,
so the metric maths can be tested and reused without a model in the room.

WHAT EACH METRIC ACTUALLY MEANS
-------------------------------
All five are built out of four counts, computed per class, by comparing what the
scan really was (`y_true`) to what the model said (`y_pred`):

    true positives  (TP)  the scan was glioma, and the model said glioma
    false positives (FP)  the scan was NOT glioma, and the model said glioma
    false negatives (FN)  the scan WAS glioma, and the model said something else
    true negatives  (TN)  the scan was not glioma, and the model said not glioma

For a 4-class problem each of these is read out of one row and one column of the
confusion matrix: TP is the diagonal cell, FN the rest of the row, FP the rest of
the column, TN everything outside both.

ACCURACY = (TP + TN) / (TP + TN + FP + FN)
    The fraction of all scans classified correctly. It is the one number that is
    easy to compute, easy to report, and easy to over-trust: it is an *average*
    over all four classes, so it hides which class is carrying it. A model that
    never predicts `pituitary` can still post a high accuracy if `pituitary` is a
    small class, because all of those scans land in FN and drag the average down
    only by their share of the total. Accuracy is the right headline number for a
    balanced dataset and the wrong one for this one.

PRECISION = TP / (TP + FP)
    Of everything the model *claimed* was glioma, how much really was. It answers
    "when this model says tumor, how often is it right?" -- the question a
    clinician asks before acting on a positive. Low precision means false alarms.
    For this project it is the cost of an unnecessary biopsy or a follow-up scan.

RECALL (a.k.a. sensitivity) = TP / (TP + FN)
    Of everything that *was* glioma, how much the model found. It answers "how
    many real cases do we catch?" -- a missed tumour, not a nuisance. Low recall
    means the model is confidently reassuring about scans that do contain a
    tumour, which in a screening setting is the dangerous direction.

F1 = 2 * (precision * recall) / (precision + recall)
    The harmonic mean of the two. "Harmonic" is the important word: the harmonic
    mean punishes imbalance far harder than the arithmetic mean, so F1 only stays
    high when *both* numbers are high. A model with precision 1.0 and recall 0.1
    scores 0.18, not the flattering 0.55 an average would give. F1 is the right
    single number when you refuse to choose between missing a tumour and crying
    wolf.

CONFIDENCE MATRIX
    A square table, one row and one column per class, counting how often each true
    class was predicted as each class. Everything above the diagonal is one kind
    of mistake, everything below it is another. Accuracy is just the diagonal
    divided by the total; the matrix is what tells you *which* cells are the
    problem, which a single scalar cannot. A model with 93% accuracy whose errors
    are all meningioma-called-glioma is a very different artefact from one that is
    93% accurate by spreading errors evenly, and only the second table shows it.

AVERAGING, AND WHY MACRO IS THE DEFAULT HERE
--------------------------------------------
Precision/recall/F1 are *per class*. To get one number you average them, and the
averaging choice changes the answer:

    macro     -- unweighted mean over the four classes. Every class counts the
                  same, however many scans it has. This is the number to quote
                  when a rare class matters as much as a common one, and it is
                  the project's default (README step 5 asks for macro).
    weighted  -- each class weighted by how many scans it has. Mathematically it
                  agrees with macro only when the classes are perfectly balanced,
                  and it is closer to what the pixel-weighted loss optimised.
    micro     -- pooled over all classes, which for single-label multiclass is
                  *identical to accuracy* by construction. Not worth a second
                  column.

Also printed: per-class support (how many true examples of that class were in the
split) and per-class F1, because the one number that best summarises a per-class
table is the F1 of the row you actually care about.

A macro average on a 4-class table is worth nothing without a floor under it, so
read it next to the per-class rows. A macro F1 of 0.72 can be one collapsed class
and three perfect ones, or four mediocre ones. The table tells you which.

WHAT TO LOOK AT FIRST IN THIS PROJECT
-------------------------------------
`no_tumor` and `glioma` are the two rows that carry the risk. Recall on
`no_tumor` is the false-reassurance rate -- healthy brains the model is willing to
call healthy -- and it is the number most worth raising with extra training data
rather than a bigger network. Glioma vs meningioma is the classic pair of similar
appearing masses; if one off-diagonal cell dominates the matrix, that pair is where
the remaining error lives.

TWO THINGS THAT WILL BITE IF IGNORED
------------------------------------
1. **Only the test split is meaningful.** `--split train` works, and it is the
   number the model already fits, so it is reported as "seen during training" and
   should never be quoted. `--split val` is legitimate, but the holdout is
   recreated from the *checkpoint's own* `val_fraction` and `seed`, which only
   reproduces the training-time split if the checkpoint still carries its config
   dict (see `load_checkpoint`).

2. **This split is by slice, not by patient.** Every metric below is an upper
   bound on real performance for the reason spelled out in `src/data/dataset.py`:
   two slices from one patient can sit on opposite sides of the train/test
   boundary, so the model is being graded on slices whose neighbours it has
   effectively already seen. `data/README.md` calls patient-level splitting the
   correct approach.

A note on `zero_division=0`
---------------------------
`compute_metrics` passes `zero_division=0` to scikit-learn. When a class is
predicted zero times, precision is 0/0 and the value is genuinely undefined. Both
0.0 and 1.0 are arbitrary; 0.0 is used because a class the model never predicts is
a problem that should be visible, and it silences the runtime warning sklearn
otherwise emits. A class with zero *support* (not present in the split at all) is
a different case -- the metric was never measurable -- and is printed as `n/a`
rather than as a number, so a missing class can never be mistaken for a good one.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .data.dataset import BrainTumorDataset
from .data.transforms import build_eval_transform
from .models.classifier import ARCH_NAME, NUM_CLASSES, build_model
from .train import resolve_device, stratified_split

# Library code must never call print() (see src/README.md) -- reports go through
# the logger, so they interleave correctly with the loader progress bars.
logger = logging.getLogger(__name__)

#: Placeholder printed for a metric that was never measurable (zero support).
NOT_AVAILABLE = "n/a"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    """Everything one evaluation run needs.

    Mirrors `src/train.py`'s `TrainConfig`: a dataclass, defaults declared in one
    place, and a `validate()` that fails before any GPU time is spent. The
    `None` defaults are not placeholders -- each one means "take it from the
    checkpoint", which is the project rule that preprocessing is never guessed.
    """

    # --- what to score ---
    checkpoint: Path | None = None
    data_root: Path = Path("data/processed")
    #: "test" (default, the only honest number), "val", or "train".
    split: str = "test"

    # --- runtime ---
    batch_size: int = 64
    num_workers: int = 0
    device: str = "auto"
    show_progress: bool = True

    # --- taken from the checkpoint unless overridden here ---
    image_size: int | None = None
    num_classes: int | None = None

    # --- output ---
    report_path: Path | None = None
    confusion_matrix_path: Path | None = None
    normalize_confusion_matrix: bool = False

    def validate(self) -> None:
        """Reject nonsense before loading a model."""
        if self.split not in ("train", "val", "test"):
            raise ValueError(
                f"Unknown split {self.split!r}. Expected 'test', 'val', or 'train'."
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.image_size is not None and self.image_size < 1:
            raise ValueError(f"image_size must be >= 1, got {self.image_size}")
        if self.num_classes is not None and self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerClassMetrics:
    """Precision / recall / F1 for one class, plus how many examples it had.

    `support` is stored rather than recomputed because it is the denominator of
    recall, and a metric whose denominator is invisible invites misreading a
    2-example class as equal to a 400-example one.
    """

    name: str
    precision: float
    recall: float
    f1: float
    support: int

    @property
    def is_measurable(self) -> bool:
        """False when the class had no examples at all, so its metrics are 0/0."""
        return self.support > 0

    def to_dict(self) -> dict[str, float | int | str]:
        """JSON-ready view, with unmeasurable metrics reported as null."""
        return {
            "name": self.name,
            "precision": self.precision if self.is_measurable else None,
            "recall": self.recall if self.is_measurable else None,
            "f1": self.f1 if self.is_measurable else None,
            "support": self.support,
        }


@dataclass
class ClassificationMetrics:
    """Every number this module computes, for one split of one checkpoint.

    Holds the per-class rows, the macro/weighted averages, and the confusion
    matrix, so a report, a JSON dump, and a PNG all read from the same object and
    cannot disagree with each other.
    """

    class_names: list[str]
    per_class: list[PerClassMetrics]
    confusion_matrix: np.ndarray
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float

    @property
    def n_samples(self) -> int:
        """How many scans were scored. Equals the sum of every per-class support."""
        return int(sum(row.support for row in self.per_class))

    @property
    def errors(self) -> list[tuple[str, str, int]]:
        """Every off-diagonal cell, as (true, predicted, count), largest first.

        This is the error analysis: a pair missing 20 scans is a different problem
        to fix than twenty pairs missing one each, and the confusion matrix is the
        only place that distinction is visible.
        """
        pairs = [
            (self.class_names[i], self.class_names[j], int(self.confusion_matrix[i, j]))
            for i in range(len(self.class_names))
            for j in range(len(self.class_names))
            if i != j and self.confusion_matrix[i, j] > 0
        ]
        return sorted(pairs, key=lambda item: item[2], reverse=True)

    def to_dict(self) -> dict[str, object]:
        """JSON-ready view of the whole result."""
        return {
            "class_names": list(self.class_names),
            "n_samples": self.n_samples,
            "accuracy": self.accuracy,
            "macro": {
                "precision": self.macro_precision,
                "recall": self.macro_recall,
                "f1": self.macro_f1,
            },
            "weighted": {
                "precision": self.weighted_precision,
                "recall": self.weighted_recall,
                "f1": self.weighted_f1,
            },
            "per_class": [row.to_dict() for row in self.per_class],
            "confusion_matrix": {
                "labels": list(self.class_names),
                # Rows are true, columns predicted. Stated in the output too,
                # because a transposed matrix is the classic silent reporting bug.
                "orientation": "rows=true, cols=predicted",
                "matrix": self.confusion_matrix.tolist(),
            },
            "top_errors": [
                {"true": t, "predicted": p, "count": c} for t, p, c in self.errors
            ],
        }


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


@dataclass
class CheckpointInfo:
    """The self-describing metadata read out of a checkpoint.

    `models/README.md` requires a `.pt` to carry its own preprocessing. That is why
    image size, architecture, and the *order* of the class names are read from here
    rather than imported from constants: a class list that has silently shifted by
    one position produces a model that is confidently wrong instead of an error.
    """

    path: Path
    arch: str
    num_classes: int
    class_names: list[str]
    image_size: int
    epoch: int | None = None
    train_metrics: dict[str, float] = field(default_factory=dict)
    config: dict[str, object] = field(default_factory=dict)


def load_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, CheckpointInfo]:
    """Rebuild a model from a checkpoint and hand back its metadata.

    Args:
        path: a `.pt` written by `src/train.py`'s `save_checkpoint`.
        device: where to put the model.

    Returns:
        (model, info). The model is in `eval()` mode.

    Raises:
        FileNotFoundError: no such checkpoint.
        ValueError: the file is not a checkpoint this project wrote.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # `weights_only=True` is the safe unpickler and is the default from PyTorch
    # 2.6 onwards, so it goes first. It is tried, not assumed, because this
    # project's own checkpoints currently fail it: `save_checkpoint` stores
    # `asdict(TrainConfig)`, whose path fields are `pathlib.Path` objects, and the
    # restricted unpickler only allows a fixed allowlist of primitives. (Fixing
    # that means serialising the config with `default=str`, as the metrics.json
    # dump already does -- after which both this function and the reload in
    # `src/train.py` can go straight to `weights_only=True`.)
    #
    # The fallback only ever runs on a checkpoint this project wrote locally, so
    # the code execution that pickle implies is not a new risk -- but it is a real
    # one, hence a warning rather than a silent retry.
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception as safe_error:  # noqa: BLE001 - re-raised below if the retry fails
        logger.warning(
            "Could not load %s with weights_only=True (%s). Retrying with the "
            "unsafe unpickler, which is only acceptable for a checkpoint this "
            "project wrote itself.",
            path,
            type(safe_error).__name__,
        )
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except Exception:  # noqa: BLE001
            raise

    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        found = sorted(checkpoint) if isinstance(checkpoint, dict) else type(checkpoint)
        raise ValueError(
            f"{path} is not a checkpoint from src/train.py -- no 'state_dict' key. "
            f"Found: {found}"
        )

    arch = str(checkpoint.get("arch", ARCH_NAME))
    num_classes = int(checkpoint.get("num_classes", NUM_CLASSES))

    model = build_model(num_classes=num_classes, name=arch)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except RuntimeError as error:
        # The usual cause is an architecture whose constructor defaults changed
        # after training (BrainTumorCNN's `pool_to`, say) -- `save_checkpoint`
        # stores the arch *name* only, so a changed default cannot be detected
        # here and surfaces as a shape mismatch.
        raise RuntimeError(
            f"Could not load the weights for arch {arch!r} from {path}: {error}. "
            "If the architecture's constructor defaults changed after this "
            "checkpoint was trained, the state dict no longer matches -- retrain "
            "or pass the original keyword arguments."
        ) from error

    model.to(device)
    # `eval()` is not optional. It disables dropout, and it is the single most
    # common reason a number here disagrees with one computed elsewhere.
    model.eval()

    class_names = [str(name) for name in checkpoint.get("class_names", [])]
    if not class_names:
        # Nothing better to fall back on than the model itself: the output layer
        # says how many classes, not what they are called.
        class_names = [f"class_{i}" for i in range(num_classes)]
        logger.warning(
            "Checkpoint %s carries no class_names; labels will be shown as %s. "
            "The class order still matches the model, only the names are lost.",
            path,
            ", ".join(class_names),
        )

    info = CheckpointInfo(
        path=path,
        arch=arch,
        num_classes=num_classes,
        class_names=class_names,
        image_size=int(checkpoint.get("image_size", 0)),
        epoch=checkpoint.get("epoch"),
        train_metrics=dict(checkpoint.get("metrics", {}) or {}),
        config=dict(checkpoint.get("config", {}) or {}),
    )
    return model, info


def find_checkpoint(root: str | Path = "models") -> Path:
    """Return the most recently modified `best.pt` under `root`.

    So `python -m src.evaluate` with no arguments scores the newest run instead of
    erroring out about a missing flag. `models/` is a per-run tree
    (`models/<run>/best.pt`), which is why the glob is one level deep.

    Raises:
        FileNotFoundError: nothing under `root` looks like a checkpoint.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {root}. Train a model first with "
            "`python -m src.train`, or pass --checkpoint explicitly."
        )

    candidates = sorted(root.glob("*/best.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No 'best.pt' under {root}. Expected models/<run>/best.pt, written by "
            "src/train.py at the end of a run."
        )

    newest = candidates[-1]
    logger.info(
        "No --checkpoint given; using the most recent of %d checkpoint(s): %s",
        len(candidates),
        newest,
    )
    return newest


# ---------------------------------------------------------------------------
# Running the model
# ---------------------------------------------------------------------------


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    show_progress: bool = True,
    desc: str = "evaluate ",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over a loader and return what it said about every image.

    Args:
        model: already in `eval()` mode.
        loader: must be built with `shuffle=False`, so predictions line up with
            the dataset order and a saved CSV is reproducible.
        device: where inference runs.
        show_progress: draw a tqdm bar over the batches.
        desc: progress-bar caption.

    Returns:
        (y_true, y_pred, y_prob):
            y_true  int64  [N]         -- the real class index of each image
            y_pred  int64  [N]         -- the predicted class index
            y_prob  float32 [N, C]     -- softmax probabilities per class

    The probabilities are returned even though the metrics below do not need them,
    because they are what makes a later ROC-AUC or threshold sweep possible without
    re-running inference. `inference_mode()` is stronger than
    `set_grad_enabled(False)`: it also skips autograd bookkeeping for tensors that
    never escape the block.
    """
    was_training = model.training
    model.eval()

    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    y_prob: list[np.ndarray] = []

    with torch.inference_mode():
        bar = tqdm(
            loader, desc=desc, leave=False, disable=not show_progress, ncols=88
        )
        for images, labels in bar:
            logits = model(images.to(device, non_blocking=True))
            probs = logits.softmax(dim=1)
            # argmax on the probabilities and on the logits give the same answer --
            # softmax is monotonic, so it cannot reorder them -- but taking it here
            # keeps "predicted" and "probability" provably the same distribution.
            y_pred.append(probs.argmax(dim=1).cpu().numpy())
            y_prob.append(probs.cpu().numpy())
            y_true.append(labels.numpy())

    if was_training:
        # Leave the caller's model exactly as it was found.
        model.train()

    if not y_true:
        raise ValueError(
            "The DataLoader yielded no batches. Is the split folder empty?"
        )

    return (
        np.concatenate(y_true).astype(np.int64),
        np.concatenate(y_pred).astype(np.int64),
        np.concatenate(y_prob).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# The metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    class_names: Sequence[str],
) -> ClassificationMetrics:
    """Turn predictions into accuracy, precision, recall, F1, and a matrix.

    Pure numpy and scikit-learn: no torch, no model, no filesystem. The metric
    definitions and every averaging caveat are in the module docstring.

    A hand-checkable case, which is the fastest way to trust the output:

        y_true = [0, 0, 1, 1]   y_pred = [0, 1, 1, 1]   classes = [A, B]
        matrix  = [[1, 1],       rows = true, cols = predicted
                   [0, 2]]
        accuracy = (1 + 2) / 4                = 0.750
        A: TP 1, FP 0, FN 1  ->  precision 1/1 = 1.000, recall 1/2 = 0.500, F1 0.667
        B: TP 2, FP 1, FN 0  ->  precision 2/3 = 0.667, recall 2/2 = 1.000, F1 0.800
        macro F1 = (0.667 + 0.800) / 2        = 0.733

    Note A's precision: it is 1/1, not 1/2, because nothing was *wrongly* called A
    -- the one scan predicted as B is a false positive for B, not for A. That
    asymmetry between the two classes is the whole reason precision and recall are
    reported per class.

    Args:
        y_true: true class indices, one per image.
        y_pred: predicted class indices, same length and order.
        class_names: label per index. Its length fixes the matrix size, so a class
            absent from the split still gets its (empty) row instead of collapsing
            the matrix and shifting every other class's numbers.

    Returns:
        A populated `ClassificationMetrics`.

    Raises:
        ValueError: mismatched or empty inputs, or an index outside `class_names`.
    """
    class_names = [str(name) for name in class_names]
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()

    if y_true.size == 0:
        raise ValueError("No predictions to score: both arrays are empty.")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must be the same length, got {y_true.size} and "
            f"{y_pred.size}. This usually means the DataLoader was shuffled, so "
            "labels no longer correspond to predictions."
        )
    if not class_names:
        raise ValueError("class_names is empty, so there is nothing to report on.")

    n_classes = len(class_names)
    out_of_range = sorted(
        {
            int(i)
            for i in (*y_true, *y_pred)
            if not 0 <= int(i) < n_classes
        }
    )
    if out_of_range:
        # A -1 is the usual culprit: CrossEntropyLoss uses ignore_index=-1 for
        # padding, and a mask that survived into the labels would land here.
        raise ValueError(
            f"Label index out of range for {n_classes} classes {class_names}: "
            f"{out_of_range}"
        )

    # `labels=` is what keeps the matrix square and correctly ordered. Without it
    # a class that is never predicted silently drops a row *and* a column, and
    # every other class's numbers shift -- the easiest way to publish a wrong
    # confusion matrix without any error being raised.
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    accuracy = float(accuracy_score(y_true, y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(n_classes)),
        # See the module docstring: 0/0 is reported as 0.0 so an unpredicted class
        # shows up as a problem instead of as a perfect score.
        zero_division=0,
    )

    # macro: the unweighted mean over classes -- every class counts the same.
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    # weighted: each class weighted by its support, i.e. by how many scans it has.
    # `np.average` is skipped when the total support is 0 (an empty split is
    # rejected above, so this is unreachable, but a NaN in a report is worse than
    # an exception).
    weights = support.astype(np.float64)
    total = float(weights.sum())
    if total > 0:
        weighted_precision = float(np.average(precision, weights=weights))
        weighted_recall = float(np.average(recall, weights=weights))
        weighted_f1 = float(np.average(f1, weights=weights))
    else:  # pragma: no cover - defensive
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    per_class = [
        PerClassMetrics(
            name=name,
            precision=float(precision[i]),
            recall=float(recall[i]),
            f1=float(f1[i]),
            support=int(support[i]),
        )
        for i, name in enumerate(class_names)
    ]

    return ClassificationMetrics(
        class_names=class_names,
        per_class=per_class,
        confusion_matrix=matrix,
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_precision=weighted_precision,
        weighted_recall=weighted_recall,
        weighted_f1=weighted_f1,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _format_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    aligns: Sequence[str] = (),
) -> list[str]:
    """Render a fixed-width ASCII table.

    Column widths come from the content, so nothing is ever truncated -- a metric
    printed as `0.92673...` because the column was one character too narrow is a
    metric nobody can read.
    """
    if not headers:
        return []
    columns = len(headers)
    aligns = list(aligns) + ["<"] * (columns - len(aligns))
    widths = [
        max(len(str(headers[c])), *(len(str(row[c])) for row in rows)) if rows
        else len(str(headers[c]))
        for c in range(columns)
    ]

    def render(cells: Sequence[str]) -> str:
        return "  ".join(
            f"{str(cell):{aligns[c]}{widths[c]}}" for c, cell in enumerate(cells)
        )

    lines = [render(headers), render(["-" * width for width in widths])]
    lines.extend(render(row) for row in rows)
    return lines


def _number(value: float | None, digits: int = 4) -> str:
    """Format a metric, or `n/a` when it was never measurable."""
    return NOT_AVAILABLE if value is None else f"{value:.{digits}f}"


def render_report(
    metrics: ClassificationMetrics,
    *,
    header: Sequence[tuple[str, str]] = (),
) -> str:
    """Format the full result as a plain-text report.

    Args:
        metrics: the result to render.
        header: extra `("label", "value")` pairs printed above the tables, e.g.
            the split, the checkpoint path, the device.

    Returns:
        A multi-line string. Returned rather than logged so it can be printed to a
        file, pasted into a README, or asserted on in a test.
    """
    lines: list[str] = []

    if header:
        lines.append("Evaluation report")
        lines.append("=" * len(lines[0]))
        width = max(len(label) for label, _ in header)
        lines.extend(f"{label:<{width}}  {value}" for label, value in header)
        lines.append("")

    # --- per-class table ---------------------------------------------------
    lines.append("Per-class metrics")
    lines.append("-" * len(lines[-1]))
    lines.extend(
        _format_table(
            ["class", "precision", "recall", "f1", "support"],
            [
                [
                    row.name,
                    # A class with no examples in this split has 0/0 everywhere.
                    # Printing a number there would invent a measurement.
                    _number(row.precision) if row.is_measurable else NOT_AVAILABLE,
                    _number(row.recall) if row.is_measurable else NOT_AVAILABLE,
                    _number(row.f1) if row.is_measurable else NOT_AVAILABLE,
                    str(row.support),
                ]
                for row in metrics.per_class
            ],
            ["<", ">", ">", ">", ">"],
        )
    )
    lines.extend(
        _format_table(
            ["macro avg", "precision", "recall", "f1", "support"],
            [
                [
                    "",
                    _number(metrics.macro_precision),
                    _number(metrics.macro_recall),
                    _number(metrics.macro_f1),
                    str(metrics.n_samples),
                ]
            ],
            ["<", ">", ">", ">", ">"],
        )
    )
    lines.extend(
        _format_table(
            ["weighted avg", "precision", "recall", "f1", "support"],
            [
                [
                    "",
                    _number(metrics.weighted_precision),
                    _number(metrics.weighted_recall),
                    _number(metrics.weighted_f1),
                    str(metrics.n_samples),
                ]
            ],
            ["<", ">", ">", ">", ">"],
        )
    )
    lines.append("")

    # --- headline numbers --------------------------------------------------
    lines.append("Overall")
    lines.append("-" * len(lines[-1]))
    lines.append(
        f"accuracy {metrics.accuracy:.4f}   "
        f"({int(np.trace(metrics.confusion_matrix))}/{metrics.n_samples} scans "
        f"correct, counting only the matrix diagonal)"
    )
    lines.append(
        f"macro F1 {metrics.macro_f1:.4f}   "
        f"(mean of the per-class F1 column, all classes weighted equally)"
    )
    lines.append("")

    # --- confusion matrix --------------------------------------------------
    label_width = max(len(name) for name in metrics.class_names)
    lines.append("Confusion matrix -- rows = true class, columns = predicted class")
    lines.append("-" * len(lines[-1]))
    lines.extend(
        _format_table(
            ["true \\ pred", *metrics.class_names],
            [
                [f"{name:>{label_width}}", *(str(int(cell)) for cell in row)]
                for name, row in zip(metrics.class_names, metrics.confusion_matrix)
            ],
            [">"] * (1 + len(metrics.class_names)),
        )
    )
    lines.append("")

    # --- what the errors were ---------------------------------------------
    if metrics.errors:
        top = metrics.errors[:3]
        lines.append("Largest mistakes (true -> predicted, count)")
        lines.append("-" * len(lines[-1]))
        for true_name, pred_name, count in top:
            lines.append(f"  {count:>6}  {true_name} -> {pred_name}")
        remaining = sum(count for _, _, count in metrics.errors[3:])
        if remaining:
            smaller = len(metrics.errors) - 3
            lines.append(f"  {remaining:>6}  in {smaller} smaller pairs")
    else:
        lines.append("No misclassifications.")

    return "\n".join(lines)


def plot_confusion_matrix(
    metrics: ClassificationMetrics,
    path: str | Path,
    *,
    normalize: bool = False,
) -> Path:
    """Save the confusion matrix as a PNG.

    `normalize=True` divides each row by its own total, turning counts into
    per-class recall -- the cell in row `c` then reads directly as "of the real
    `c` scans, this fraction were called `j`", which is the view that makes a weak
    class obvious. The un-normalized view is the one to quote, because a rare class
    otherwise looks like a rounding error.

    matplotlib is imported inside the function: it is a heavy import that a
    text-only evaluation has no reason to pay for, and this keeps the module
    usable if it is ever optional.
    """
    import matplotlib

    # A file on disk is the only destination here, so pick the non-interactive
    # backend rather than letting a headless session fail at savefig time.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(metrics.confusion_matrix, dtype=np.float64)
    if normalize:
        row_totals = matrix.sum(axis=1, keepdims=True)
        # A class with no examples stays all-zero instead of becoming NaN.
        matrix = np.divide(
            matrix, row_totals, out=np.zeros_like(matrix), where=row_totals > 0
        )

    size = max(5.0, 0.9 * len(metrics.class_names) + 2.5)
    figure, axes = plt.subplots(figsize=(size, size))
    image = axes.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)

    ticks = range(len(metrics.class_names))
    axes.set_xticks(list(ticks), metrics.class_names, rotation=45, ha="right")
    axes.set_yticks(list(ticks), metrics.class_names)
    # These two are the labels that make the matrix readable; without them a
    # reader has to guess whether row or column is the truth.
    axes.set_xlabel("Predicted")
    axes.set_ylabel("True")
    axes.set_title(
        "Confusion matrix (rows = true, cols = predicted)"
        + (" [row-normalised]" if normalize else "")
    )

    # Cell annotations. The colour flips past 0.5 * max so dark cells get light
    # text and stay readable at a glance.
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for i, name_i in enumerate(metrics.class_names):
        for j, _ in enumerate(metrics.class_names):
            value = matrix[i, j]
            axes.text(
                j,
                i,
                f"{value:.2f}" if normalize else f"{int(value)}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=9,
            )

    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("Confusion matrix written to %s", path)
    return path


def save_report(metrics: ClassificationMetrics, path: str | Path) -> Path:
    """Write the metrics as JSON, next to the checkpoint for the record.

    The same object feeds the text report, the PNG, and this file, so the three
    cannot drift apart -- which is the whole reason the numbers live in one
    dataclass instead of being formatted twice.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = metrics.to_dict()
    payload["generated_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Metrics written to %s", path)
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_eval_loader(cfg: EvalConfig, info: CheckpointInfo) -> DataLoader:
    """Build the loader for the requested split, with the eval transform.

    Three details that are easy to get wrong:

      * The transform is `build_eval_transform`, never the augmented training
        pipeline. Random flips and rotations during evaluation make the score move
        on every run, which makes it impossible to tell an improvement from noise.
      * The class list comes from the *checkpoint*, not from the folders on disk.
        `BrainTumorDataset` then maps files to those labels, so a `test/` folder
        whose class set differs from training cannot silently shift every index.
      * `shuffle=False` everywhere, so predictions stay in dataset order.

    For `split="val"` the holdout is rebuilt with the *training* run's
    `val_fraction` and `seed`, read from the checkpoint's config dict, because
    `src/train.py` keeps no val index list on disk. If the config is missing the
    defaults are used and a warning says so -- that split will not be the one the
    model was selected on.
    """
    image_size = cfg.image_size or info.image_size or None
    if image_size is None:
        raise ValueError(
            f"{info.path} records no image_size and none was passed. The model was "
            "trained at a size that cannot be recovered, so the eval transform "
            "cannot be reproduced. Pass --image-size explicitly."
        )

    def make_dataset(split: str) -> BrainTumorDataset:
        return BrainTumorDataset(
            root=cfg.data_root,
            split=split,
            class_names=info.class_names,
            transform=build_eval_transform(image_size),
        )

    if cfg.split == "val":
        train_dataset = make_dataset("train")
        val_fraction = float(info.config.get("val_fraction", 0.2))
        seed = int(info.config.get("seed", 42))
        logger.info(
            "Rebuilding the validation holdout from the checkpoint: "
            "val_fraction=%.2f seed=%d",
            val_fraction,
            seed,
        )
        _, val_index = stratified_split(train_dataset, val_fraction, seed)
        dataset: BrainTumorDataset | Subset = Subset(train_dataset, val_index)
    elif cfg.split == "train":
        dataset = make_dataset("train")
    else:
        dataset = make_dataset("test")

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def evaluate(cfg: EvalConfig) -> tuple[ClassificationMetrics, CheckpointInfo]:
    """Score one checkpoint on one split. The module's one real entry point.

    Returns:
        (metrics, info) so a caller can reach both the numbers and the provenance.
    """
    cfg.validate()
    device = resolve_device(cfg.device)

    checkpoint_path = cfg.checkpoint or find_checkpoint()
    model, info = load_checkpoint(checkpoint_path, device)
    if cfg.num_classes is not None and cfg.num_classes != info.num_classes:
        logger.warning(
            "--num-classes %d does not match the checkpoint's %d; using the "
            "checkpoint.",
            cfg.num_classes,
            info.num_classes,
        )

    logger.info(
        "Evaluating %s (arch=%s, epoch=%s) on split '%s' of %s",
        info.path,
        info.arch,
        "?" if info.epoch is None else info.epoch,
        cfg.split,
        cfg.data_root,
    )

    loader = build_eval_loader(cfg, info)
    y_true, y_pred, _ = collect_predictions(
        model, loader, device, show_progress=cfg.show_progress, desc=f"[{cfg.split}]   "
    )

    metrics = compute_metrics(y_true, y_pred, info.class_names)
    logger.info(
        "Scored %d images -- accuracy %.4f, macro F1 %.4f",
        metrics.n_samples,
        metrics.accuracy,
        metrics.macro_f1,
    )
    return metrics, info


def parse_args(argv: list[str] | None = None) -> EvalConfig:
    """Build an EvalConfig from the command line, falling back to the defaults."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint: accuracy, precision, recall, "
        "F1, confusion matrix.",
    )
    defaults = EvalConfig()

    parser.add_argument("--checkpoint", type=Path, default=defaults.checkpoint,
                        help="path to a .pt from src/train.py; defaults to the "
                             "newest models/*/best.pt")
    parser.add_argument("--data-root", type=Path, default=defaults.data_root,
                        help="folder containing train/ and test/ (default: %(default)s)")
    parser.add_argument("--split", type=str, default=defaults.split,
                        choices=("test", "val", "train"),
                        help="which split to score (default: %(default)s). Only "
                             "'test' is a held-out number; 'train' is what the "
                             "model already fits")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers,
                        help="loader subprocesses; >0 requires the __main__ guard on Windows")
    parser.add_argument("--device", type=str, default=defaults.device,
                        help="'auto', 'cpu', 'cuda', or 'cuda:0'")
    parser.add_argument("--image-size", type=int, default=defaults.image_size,
                        help="override the checkpoint's image size")
    parser.add_argument("--num-classes", type=int, default=defaults.num_classes,
                        help="override the checkpoint's class count")
    parser.add_argument("--no-progress", dest="show_progress", action="store_false",
                        default=defaults.show_progress)
    parser.add_argument("--save-report", type=Path, default=defaults.report_path,
                        help="write the metrics as JSON to this path")
    parser.add_argument("--save-confusion-matrix", type=Path,
                        default=defaults.confusion_matrix_path,
                        help="render the confusion matrix to this PNG")
    parser.add_argument("--normalize-confusion-matrix", action="store_true",
                        default=defaults.normalize_confusion_matrix,
                        help="divide each matrix row by its total, i.e. show "
                             "per-class recall instead of raw counts")

    args = parser.parse_args(argv)
    return EvalConfig(
        checkpoint=args.checkpoint,
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        show_progress=args.show_progress,
        image_size=args.image_size,
        num_classes=args.num_classes,
        report_path=args.save_report,
        confusion_matrix_path=args.save_confusion_matrix,
        normalize_confusion_matrix=args.normalize_confusion_matrix,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = parse_args(argv)
    metrics, info = evaluate(cfg)

    if cfg.split == "train":
        logger.warning(
            "These numbers are on the split the model was trained on. They "
            "measure how well it memorised, not how well it generalises."
        )

    logger.info(
        "\n%s",
        render_report(
            metrics,
            header=[
                ("split", cfg.split),
                ("images", str(metrics.n_samples)),
                ("classes", ", ".join(info.class_names)),
                ("checkpoint", str(info.path)),
                ("arch", info.arch),
                ("data root", str(cfg.data_root)),
            ],
        ),
    )

    if cfg.report_path is not None:
        save_report(metrics, cfg.report_path)
    if cfg.confusion_matrix_path is not None:
        plot_confusion_matrix(
            metrics,
            cfg.confusion_matrix_path,
            normalize=cfg.normalize_confusion_matrix,
        )


# The __main__ guard is not optional on Windows: DataLoader workers are spawned as
# fresh processes that re-import this module, so without it every worker re-runs
# main(). Same reason as in src/train.py.
if __name__ == "__main__":
    main()
