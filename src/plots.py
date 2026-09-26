"""
plots.py -- render a finished training run as PNGs.

WHAT THIS FILE DOES
-------------------
`src/train.py` records one row per epoch in `history` and writes it to
`models/<run>/metrics.json`. This module turns those rows into the four figures you
actually look at when judging a run:

    train_loss.png        training loss per epoch
    val_loss.png          validation loss per epoch
    train_accuracy.png    training accuracy per epoch
    val_accuracy.png      validation accuracy per epoch

Two ways in, one code path:

    from .plots import plot_training_curves
    plot_training_curves(history, "outputs/runs/resnet50_20260926-101500")

    python -m src.plots --metrics models/resnet50_20260926-101500/metrics.json

`src/train.py` calls the function at the end of every run, so the PNGs appear without
anyone asking; the CLI regenerates them for an old run without retraining anything.
That second path matters more than it looks: curves are the evidence for a
hyperparameter decision, and the decision is often made weeks after the run, on a
machine that no longer has the checkpoint.

WHAT THE FOUR CURVES ARE FOR
----------------------------
Each one answers a different question, and reading only the training curve is the
classic mistake.

    train loss   -- did gradient descent do its job at all? Should fall
                   monotonically-ish. A curve that rises or flattens high means the
                   LR is wrong, the labels are shuffled, or the loss is a worse
                   signal than the metric (see `build_criterion` on why weighted
                   training loss is a poor progress readout).
    val loss     -- the number `EarlyStopping` watches, because it is the smoothest
                   signal available: it moves every epoch, where accuracy is a step
                   function that can sit flat for five epochs then jump. Its minimum
                   is the epoch that gets checkpointed, and it is marked on the plot.
    train acc    -- should stay below val acc early, then overtake it. Once it is
                   the higher of the two, the model is memorising.
    val acc      -- the generalisation number. The gap between the two accuracy
                   curves is the overfitting gap; a val curve that peaks and then
                   falls while train keeps climbing is the signature to stop at.

These four are deliberately separate files rather than one 2x2 figure: the comparison
that matters is train-vs-val *within* a metric, and two series on one axis is a
different, overlapping question. Open them side by side.

Notes on the rendering, since the defaults are chosen for reading rather than looks:

* Non-finite values (a NaN loss from a diverged run) are plotted as gaps, not
  dropped. Dropping a point would close the line across it and imply the run
  recovered; a gap shows the failure where it happened.
* Accuracy axes are not forced to 0..1. On a hard split the interesting variation is
  the last few percent, and a full-range axis flattens it into a straight line.
* Epoch axes get integer ticks only, so "epoch 3" cannot read as "epoch 3.7".
* matplotlib is imported lazily and forced to the `Agg` backend: these functions are
  called from the training script and from a terminal, and neither has a display.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

logger = logging.getLogger(__name__)

#: Where runs put their figures when the caller does not say. `outputs/` is the
#: project's disposable-artifact root (see outputs/README.md); nothing there is ever
#: read back at inference time.
DEFAULT_OUTPUT_ROOT = Path("outputs")

#: Per-run subfolder, so parallel experiments never overwrite each other's figures.
RUNS_SUBDIR = "runs"

_PYPLOT: ModuleType | None = None


def _pyplot() -> ModuleType:
    """Import pyplot once, on the non-interactive backend.

    Deferred to first use because matplotlib is a heavy import that training and
    inference have no reason to pay for -- `src.evaluate.plot_confusion_matrix` does
    the same thing. The caching matters: `matplotlib.use()` after pyplot has already
    been imported elsewhere is a no-op on some versions, so the backend is chosen
    once, here, before anything else can grab pyplot.
    """
    global _PYPLOT
    if _PYPLOT is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _PYPLOT = plt
    return _PYPLOT


# ---------------------------------------------------------------------------
# Curve definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Curve:
    """One figure: which history column to draw, and how to label it.

    A table of these instead of four hand-written plotting functions. The four
    figures are the same picture with different columns and titles, so duplicating the
    axis handling four times is four places for a tick or a title to go wrong -- and
    adding a fifth curve later becomes a copy-paste rather than a new row.
    """

    key: str
    """Column in each `history` record. These are the exact keys `src.train` writes
    (`train_loss`, `val_loss`, `train_acc`, `val_acc`) -- a mismatch here is silent,
    because the figure is simply never drawn."""

    filename: str
    ylabel: str
    title: str
    #: `"loss"` or `"accuracy"`. Only used to pick annotation text; the axis scaling
    #: is deliberately left to matplotlib in both cases.
    kind: str

    @property
    def is_loss(self) -> bool:
        return self.kind == "loss"


#: The four figures this module exists to produce, in the order they are written.
CURVES: tuple[Curve, ...] = (
    Curve(
        key="train_loss",
        filename="train_loss.png",
        ylabel="Cross-entropy loss",
        title="Training loss",
        kind="loss",
    ),
    Curve(
        key="val_loss",
        filename="val_loss.png",
        ylabel="Cross-entropy loss",
        title="Validation loss",
        kind="loss",
    ),
    Curve(
        key="train_acc",
        filename="train_accuracy.png",
        ylabel="Accuracy",
        title="Training accuracy",
        kind="accuracy",
    ),
    Curve(
        key="val_acc",
        filename="val_accuracy.png",
        ylabel="Accuracy",
        title="Validation accuracy",
        kind="accuracy",
    ),
)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _numeric_columns(
    history: Sequence[Mapping[str, float]],
) -> tuple[list[int], dict[str, list[float]]]:
    """Pull the epoch axis and every curve column out of the history records.

    Columns absent from the records are simply left out rather than raising: a
    metrics.json written by an older run may not carry every key, and refusing to
    draw the curves that *are* there helps nobody.
    """
    epochs: list[int] = []
    columns: dict[str, list[float]] = {}

    for position, record in enumerate(history):
        if "epoch" in record:
            epochs.append(int(record["epoch"]))
        else:
            # A history with no epoch column still has an order; use 1-based
            # positions so the axis is meaningful rather than dropped.
            epochs.append(position + 1)

        for key in (curve.key for curve in CURVES):
            if key not in record:
                continue
            value = record[key]
            columns.setdefault(key, []).append(float(value))

    return epochs, columns


def _best_epoch(epochs: Sequence[int], values: Sequence[float], curve: Curve) -> int | None:
    """The epoch this figure should point at.

    For a loss curve that is the minimum -- the epoch `EarlyStopping` monitors and
    checkpoints. For an accuracy curve it is the maximum. Non-finite values are
    skipped so a diverged epoch is never reported as "best".
    """
    usable = [
        (epoch, value)
        for epoch, value in zip(epochs, values)
        if value == value  # NaN != NaN; inf compares fine but is not a real metric
        and value not in (float("inf"), float("-inf"))
    ]
    if not usable:
        return None

    chooser = min if curve.is_loss else max
    return chooser(usable, key=lambda pair: pair[1])[0]


def _render_curve(
    curve: Curve,
    epochs: Sequence[int],
    values: Sequence[float],
    best: int | None,
    *,
    subtitle: str | None = None,
    dpi: int = 150,
) -> tuple[ModuleType, ModuleType]:
    """Draw one figure and return the (figure, axes) pair.

    Split out from the saving so the drawing is testable without touching disk.
    """
    plt = _pyplot()

    figure, axes = plt.subplots(figsize=(8.0, 4.5))

    # The series itself. `marker="o"` matters more than it looks: with 20-odd epochs
    # it shows exactly which points are real measurements, and it stays legible when
    # the curve is nearly flat.
    axes.plot(epochs, values, marker="o", markersize=4, linewidth=1.8, color="#3b6ea5")

    if best is not None and best in epochs:
        best_value = values[epochs.index(best)]
        marker, annotation = (
            ("best val loss" if curve.is_loss else "best val accuracy"),
            f"best: {best_value:.4f}",
        )
        # A vertical span plus a point on the curve. The span is what answers "which
        # epoch was selected", which is the question a checkpointed run is really
        # asking; the point answers "how good was it".
        axes.axvline(best, color="#b03a2e", linestyle="--", linewidth=1.0, alpha=0.7)
        axes.plot(
            [best], [best_value], marker="D", markersize=7, color="#b03a2e", zorder=5
        )
        axes.annotate(
            f"{marker} @ epoch {best}\n{annotation}",
            xy=(best, best_value),
            xytext=(6, 12),
            textcoords="offset points",
            fontsize=8,
            color="#b03a2e",
        )

    axes.set_title(curve.title if subtitle is None else f"{curve.title} -- {subtitle}")
    axes.set_xlabel("Epoch")
    axes.set_ylabel(curve.ylabel)
    axes.grid(True, alpha=0.3, linewidth=0.6)
    # Integer ticks only, so an epoch label can never be read as a fraction.
    axes.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes.set_xlim(min(epochs) - 0.5, max(epochs) + 0.5)

    figure.tight_layout()
    return figure, axes


def plot_training_curves(
    history: Sequence[Mapping[str, float]],
    output_dir: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    subtitle: str | None = None,
    dpi: int = 150,
) -> list[Path]:
    """Write the four training-curve PNGs into `output_dir`.

    Args:
        history: the per-epoch records from `src.train` -- the same objects written
            to `metrics.json`, each with at least `epoch` plus the four metric keys.
        output_dir: directory for the PNGs. Created if missing. A run's own
            convention is `outputs/runs/<experiment>`; passing `outputs` puts all
            four figures at the top level.
        subtitle: short run identifier appended to each title, e.g. `"resnet50_..."`,
            so a figure found loose in a folder still says what it belongs to.
        dpi: passed to `savefig`.

    Returns:
        The paths written, in `CURVES` order.

    Raises:
        ValueError: `history` is empty, or contains no recognised metric column.
    """
    history = list(history)
    if not history:
        raise ValueError(
            "Cannot plot an empty history -- no epoch completed. Nothing to draw."
        )

    plt = _pyplot()
    epochs, columns = _numeric_columns(history)

    drawable = [curve for curve in CURVES if columns.get(curve.key)]
    if not drawable:
        raise ValueError(
            f"None of the expected columns are present in the history records. "
            f"Expected any of: {', '.join(curve.key for curve in CURVES)}. "
            f"Found: {', '.join(sorted(history[0])) or '(no keys)'}"
        )
    if len(drawable) != len(CURVES):
        missing = [c.key for c in CURVES if c not in drawable]
        logger.warning(
            "History has no %s column(s); %d of %d curves will be written.",
            ", ".join(missing),
            len(drawable),
            len(CURVES),
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for curve in drawable:
        values = columns[curve.key]
        figure, _ = _render_curve(
            curve,
            epochs,
            values,
            _best_epoch(epochs, values, curve),
            subtitle=subtitle,
            dpi=dpi,
        )
        path = output_dir / curve.filename
        figure.savefig(path, dpi=dpi)
        plt.close(figure)
        written.append(path)
        logger.info("Wrote %s", path)

    return written


# ---------------------------------------------------------------------------
# Reading a finished run
# ---------------------------------------------------------------------------


def load_history(path: str | Path) -> list[dict[str, float]]:
    """Read the per-epoch history out of a run's `metrics.json`.

    Accepts either the real file -- `{"history": [...], ...}` -- or a bare list of
    records, so a history extracted from a notebook also works.

    Raises:
        FileNotFoundError: no such file.
        ValueError: the file exists but carries no usable history.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"metrics file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload.get("history") if isinstance(payload, Mapping) else payload

    if not isinstance(history, list) or not history:
        raise ValueError(
            f"{path} contains no 'history' list. Expected the metrics.json written "
            f"by src.train.py."
        )
    if not all(isinstance(record, Mapping) for record in history):
        raise ValueError(f"{path} has a 'history' that is not a list of records.")

    return [dict(record) for record in history]


def find_metrics(root: str | Path = "models") -> Path:
    """Return the most recently modified `*/metrics.json` under `root`.

    Mirrors `src.evaluate.find_checkpoint`, so `python -m src.plots` with no arguments
    plots the newest run instead of erroring on a missing flag.

    Raises:
        FileNotFoundError: nothing under `root` looks like a finished run.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {root}. Train a model first with "
            f"`python -m src.train`, or pass --metrics explicitly."
        )

    candidates = sorted(root.glob("*/metrics.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No metrics.json under {root}. Expected {root}/<run>/metrics.json, "
            f"written by src.train.py at the end of a run."
        )

    newest = candidates[-1]
    logger.info(
        "No --metrics given; using the most recent of %d run(s): %s",
        len(candidates),
        newest,
    )
    return newest


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="Plot a finished run's training/validation loss and accuracy "
        "curves as PNGs.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="a metrics.json from src.train.py; defaults to the newest "
             "models/*/metrics.json",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path("models"),
        help="where to look for runs when --metrics is omitted "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="where to write the PNGs (default: %(default)s)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="subtitle appended to each figure title, e.g. the run name",
    )
    parser.add_argument(
        "--dpi", type=int, default=150, help="savefig resolution (default: %(default)s)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args(argv)
    if args.dpi < 1:
        raise ValueError(f"dpi must be >= 1, got {args.dpi}")

    metrics_path = args.metrics or find_metrics(args.models_root)
    history = load_history(metrics_path)

    # Default the subtitle to the run directory's name, so figures are identifiable
    # after they have been copied out of their folder.
    subtitle = args.title or metrics_path.parent.name

    paths = plot_training_curves(
        history, args.output_dir, subtitle=subtitle, dpi=args.dpi
    )
    logger.info("Wrote %d figure(s) to %s", len(paths), args.output_dir)


# The guard is for the same reason as in src/train.py and src/evaluate.py: a
# DataLoader worker re-imports the module on spawn, and an unguarded main() would run
# the whole plotting pass once per core.
if __name__ == "__main__":
    main()
