"""
inference.py -- classify a single MRI scan with a trained checkpoint.

WHAT THIS FILE DOES
-------------------
`src/train.py` writes a checkpoint, `src/evaluate.py` scores it over a whole split.
This module is the third entry point: one image in, one class and a confidence out.

    from src.inference import predict_file

    prediction = predict_file("scan.jpg", checkpoint="models/run1/best.pt")
    print(prediction.label, f"{prediction.confidence:.1%}")

It is the whole inference path, and it is a module rather than a script because
`app.py` needs the same thing. The project's rule is that the UI never contains
model logic, so Streamlit calls `predict_file` and gets a `Prediction` back; the
`predict.py` CLI at the project root calls the same function and prints the
formatted report. Neither re-implements a line of it.

    predict_file(path)      one image, one Prediction
    predict_files(paths)    the same over a list, for batch scoring

THE ONE RULE THAT MATTERS: DO NOT GUESS THE PREPROCESSING
---------------------------------------------------------
A checkpoint carries its own `image_size`, `class_names`, `norm_mean` and
`norm_std` (see `models/README.md` and `src/train.py`'s `save_checkpoint`), and
every one of them is read from there. None is imported as a constant.

The reason is worth stating plainly, because it is the failure mode this project
is most exposed to. Getting the preprocessing wrong does not raise an error. The
forward pass still runs, still produces four numbers, and they still sum to one --
the model simply reads a shifted, wrongly-scaled image and reports a confident,
plausible, wrong answer. `src/models/classifier.py` says the same thing about
ImageNet normalisation: the failure is quiet, and training loss looking fine is
not evidence against it.

`class_names` in particular is the one that bites hardest, because the *order* is
the model's output order. A class list that has silently shifted by one position
turns every prediction into a different, equally confident, wrong label. So the
order comes from the checkpoint, never from the folder names on disk, and never
from `NUM_CLASSES` + alphabetical order.

WHAT "CONFIDENCE" ACTUALLY MEANS HERE
------------------------------------
`Prediction.confidence` is the softmax probability of the winning class, and it is
worth being precise about what that is: *the model's own belief*, measured on its
own training distribution. It is **not** a calibrated probability of the diagnosis
being correct, and on this project it is not, for two concrete reasons:

    1. A model trained with cross-entropy on a small, clean, pre-cropped dataset
       is systematically overconfident on data that is messier than its training
       set -- a differently-windowed scan, a different scanner, a slice with
       more surrounding tissue.
    2. A frozen ResNet-18 backbone with a 2,052-parameter linear head was never
       given enough capacity to represent a well-shaped probability, let alone
       temperature-calibrated one.

"94%" therefore means "of the four logits this model produced, one was 16x the
next" -- not "there is a 94% chance this is a glioma". Turning the first into the
second is what temperature scaling on a held-out calibration split is for, and it
is not implemented yet. The README's promise of "calibrated confidence" is
therefore not kept by this module, and `render_report` says so in its own output
so a number pasted into a slide cannot travel without the caveat.

A second consequence: the full per-class distribution is always returned, not
just the top class. The gap between the winner and the runner-up is the more
informative number when a scan is ambiguous, and it costs nothing to keep.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
    * No Grad-CAM. The README plans it and `app.py` expects it, but a heatmap is
      only as trustworthy as the explanation method validated against, and
      shipping an unvalidated one next to a class label invites reading it as
      evidence. It belongs here, as its own function, once it has been checked.
    * No ensemble, no test-time augmentation, no TTA voting. All defensible, all
      changes the number reported here away from the one `src/evaluate.py` scored.
      Inference that disagrees with evaluation is a bug, not a feature.
    * No batch path that quietly takes a mean of the confidences. `predict_files`
      returns one `Prediction` per image, because an average confidence over a set
      of scans means nothing.
    * No `no_tumor` special-casing. A "scan looks normal" result is a prediction
      like any other and gets no separate threshold; the honest way to express
      "this model is not sure there is even a tumour here" is the confidence
      number, not a hand-tuned cutoff hidden inside the inference path.

WHY THE TRANSFORM IS BUILT HERE RATHER THAN REUSED FROM THE DATASET
-------------------------------------------------------------------
`src/data/transforms.build_eval_transform` is used directly, and that reuse is the
point: the pipeline is defined once, in one file, for training and for scoring.
`BrainTumorDataset` is not used, because it wants a `split/class/` directory tree
and a label, and this module has a path to one file and no label. The three lines
of `Image.open(...).convert("RGB")` before the transform are the only
preprocessing duplicated from `dataset.py`, and the `.convert("RGB")` in it is
not optional -- see `dataset.py`'s comment at `__getitem__`.

`convert("RGB")` also does something slightly surprising for a grayscale scan: it
replicates the single channel into all three, so the normalised input is R == G == B
rather than a colour image. That is intended. The ImageNet stem the pretrained
backbone was trained on expects three channels, and a gray pixel is exactly the
ImageNet mean, so this is the transformation the weights were designed for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn

from .data.dataset import IMAGE_EXTENSIONS
from .data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_eval_transform,
)
from .evaluate import (
    CheckpointInfo,
    display_label,
    find_checkpoint,
    load_checkpoint,
)
from .train import resolve_device

# `display_label` is imported, not defined, here. It lives in `evaluate` because
# that module already needs it for the confusion figure, and the import graph runs
# `inference -> evaluate`; re-exporting keeps `predict.py` and `app.py` reading
# `from src.inference import display_label` -- the name belongs with the prediction
# they are formatting -- without a second implementation to drift.

# Library code must never call print() (see src/README.md) -- a Prediction is
# returned to the caller, and `format_report` builds a string it can print, log, or
# write wherever it likes. The printing belongs to the entry points.
logger = logging.getLogger(__name__)

#: Width of the probability bar in `format_report`, in characters. Fixed rather
#: than terminal-derived so the output is byte-identical when redirected to a file
#: or asserted on in a test, and wide enough that a 40% probability still shows
#: visible length.
BAR_WIDTH = 24

#: Printed at the end of every report. The project is explicitly not a medical
#: device (README), and a prediction copied out of a terminal into a patient note
#: should carry that with it.
NOT_A_DEVICE = "Research output only -- not for diagnosis or treatment decisions."


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prediction:
    """One image's classification: the winning class, and how sure the model is.

    Frozen, and every field is a plain scalar or an immutable mapping, because a
    prediction is a *reading*. Nothing downstream -- a report, a CSV, a Streamlit
    panel -- should be able to edit the number after the fact.

    Attributes:
        label: the predicted class, in the checkpoint's own spelling
            (`no_tumor`, not `No Tumor`). Formatting for display is
            `format_report`'s job, so the stored value stays the key the
            checkpoint indexes by.
        class_index: position of `label` in the model's output, which is the only
            thing the index ever means.
        confidence: softmax probability of `label`, in [0, 1]. See the module
            docstring: this is the model's belief, not a calibrated probability.
        probabilities: every class mapped to its probability, in the checkpoint's
            own order. Kept whole because the runner-up is what makes an ambiguous
            scan legible, and discarding it loses that for nothing.
        image_size: side length the image was resized to before the forward pass.
            On the record, because it is the one input to the number that a
            reader cannot recover from the image itself.
    """

    label: str
    class_index: int
    confidence: float
    probabilities: dict[str, float]
    image_size: int

    @property
    def runner_up(self) -> tuple[str, float] | None:
        """The second-highest class and its probability, or None if there is none.

        Comparing it with `confidence` is the cheapest available read on whether
        a prediction is worth trusting: a 0.94 against a 0.01 runner-up is a
        different claim from 0.94 against 0.44, and a single number cannot show
        that.
        """
        others = [
            (name, prob)
            for name, prob in self.probabilities.items()
            if name != self.label
        ]
        return max(others, key=lambda item: item[1]) if others else None

    @property
    def margin(self) -> float:
        """Confidence minus the runner-up's probability.

        Roughly "how far ahead of the field" the winner is, in probability
        points. 1.0 means the model put everything on one class and 0.0 means it
        could not tell the top two apart. Negative is impossible by construction,
        since `confidence` is the maximum.
        """
        second = self.runner_up
        return self.confidence - (second[1] if second else 0.0)

    def to_dict(self) -> dict[str, object]:
        """JSON-ready view, for writing a prediction to disk or an API response."""
        return {
            "label": self.label,
            "class_index": self.class_index,
            "confidence": self.confidence,
            "confidence_percent": round(self.confidence * 100, 2),
            "margin": self.margin,
            "image_size": self.image_size,
            "probabilities": dict(self.probabilities),
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def resolve_image_size(
    info: CheckpointInfo,
    override: int | None = None,
) -> int:
    """Work out what size to resize to, refusing to guess.

    `override` (a `--image-size` flag) wins, then the checkpoint's own
    `image_size`, and if neither is available this raises rather than falling back
    to `transforms.IMAGE_SIZE`.

    The refusal is the interesting part. Falling back to the module constant
    looks harmless -- the constant is 224 and most checkpoints are 224 -- but it
    is a guess dressed as a default, and a guess here is the silent-wrong-answer
    failure described in the module docstring. `src/evaluate.py` makes the same
    call for the same reason.

    Raises:
        ValueError: neither source knows the size.
    """
    if override is not None:
        if override < 1:
            raise ValueError(f"image_size must be >= 1, got {override}")
        return override

    if info.image_size >= 1:
        return info.image_size

    raise ValueError(
        f"{info.path} records no image_size and none was passed, so the "
        "preprocessing cannot be reproduced and this image cannot be classified "
        "honestly. Pass image_size= (or --image-size) explicitly if you know what "
        "the model was trained at."
    )


def check_preprocessing(info: CheckpointInfo) -> None:
    """Warn when the checkpoint's normalisation is not the one we would apply.

    `build_eval_transform` normalises with `transforms.IMAGENET_MEAN` / `_STD`,
    because those are the constants a checkpoint written by this project records.
    So in the normal case the two agree and there is nothing to do.

    They stop agreeing if a checkpoint was trained with different statistics --
    a per-dataset mean, say, computed over the training split. Then this module
    would apply the wrong ones *silently*: the transform pipeline has no way to be
    told otherwise, and a model fed ImageNet-normalised input when it was trained
    on dataset-normalised input returns a confident wrong answer rather than an
    error. A mismatch here is not a cosmetic difference, so it is logged loudly
    instead.

    An absent `norm_mean`/`norm_std` is a warning too, but a milder one: it means
    the checkpoint cannot vouch for its own preprocessing, not that the two
    provably differ.
    """
    for name, recorded in (
        ("norm_mean", info.norm_mean),
        ("norm_std", info.norm_std),
    ):
        if recorded is None:
            logger.warning(
                "Checkpoint %s records no %s. Falling back to the constants in "
                "src/data/transforms.py, which is a guess: if the model was "
                "trained on different statistics the confidence below is "
                "meaningless.",
                info.path,
                name,
            )
            continue

        current = list(IMAGENET_MEAN if name == "norm_mean" else IMAGENET_STD)
        if len(recorded) != len(current) or any(
            not _close(a, b) for a, b in zip(recorded, current)
        ):
            logger.warning(
                "Checkpoint %s was trained with %s=%s, but src/data/transforms.py "
                "applies %s. Every prediction from this checkpoint is on "
                "mismatched preprocessing; fix the constants or rebuild the "
                "transform from the checkpoint before trusting the output.",
                info.path,
                name,
                recorded,
                current,
            )


def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    """Float equality with a tolerance.

    The constants are written as decimal literals and round-tripped through
    torch.save, so exact `==` would work -- but a tolerance is the right default
    for a comparison deciding whether to raise a warning, and a spurious warning
    here trains people to ignore warnings.
    """
    return abs(a - b) <= tol


def load_image(
    path: str | Path,
    image_size: int,
) -> torch.Tensor:
    """Read one image from disk into the exact tensor the model expects.

    Args:
        path: a JPEG, PNG, or anything else in `data.dataset.IMAGE_EXTENSIONS`.
        image_size: square side length to resize to, from `resolve_image_size`.

    Returns:
        float32 tensor of shape [1, 3, image_size, image_size], already resized
        and normalised. The leading 1 is the batch dimension: a ResNet expects a
        batch, and a single 3-D tensor `[3, H, W]` would be read as a batch of 3
        unbatched images and then fail deep inside the first convolution with a
        shape error naming a tensor the caller never mentioned.

    Raises:
        FileNotFoundError: no such file.
        ValueError: the path is a directory, has an extension that is not an
            image, or is not decodable as one. Checked here rather than left to
            PIL so a typo'd path gives a sentence instead of a traceback.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.is_dir():
        raise ValueError(
            f"{path} is a directory. This function takes a single image file; "
            "use predict_files() for a list of paths."
        )
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        # A wrong extension is far more often a typo'd flag than a genuinely
        # exotic format, and a wrong-but-decodable extension is a silent
        # correctness trap -- so it is checked, not tolerated.
        raise ValueError(
            f"{path} does not look like an image (extension "
            f"{path.suffix or 'none'!r}). Expected one of: "
            f"{', '.join(IMAGE_EXTENSIONS)}."
        )

    try:
        with Image.open(path) as handle:
            # Grayscale -> 3 identical channels. See the module docstring: the
            # ImageNet stem the backbone came from needs three, and a gray pixel
            # is exactly the ImageNet mean, so this is lossless in the sense that
            # matters.
            image = handle.convert("RGB")
    except UnidentifiedImageError as error:
        raise ValueError(
            f"Could not decode {path} as an image. The extension says it is one, "
            "so the file is probably corrupt or renamed from something else."
        ) from error

    # One pipeline, shared with training and with src/evaluate.py. Reached only
    # after the file has been decoded, so the Image.open inside Resize is on a
    # real PIL image and cannot fail late.
    tensor = build_eval_transform(image_size)(image)

    if tensor.shape[0] != 3:
        # Cannot happen via convert("RGB"), but the model cannot accept it and a
        # silent wrong-channel input would be a far worse failure than this.
        raise ValueError(
            f"Expected a 3-channel tensor for {path}, got {tensor.shape[0]} "
            f"channels."
        )

    # unsqueeze(0): [3, H, W] -> [1, 3, H, W]. See the Returns note above.
    return tensor.unsqueeze(0)


# ---------------------------------------------------------------------------
# Predicting
# ---------------------------------------------------------------------------


def predict_tensor(
    model: nn.Module,
    tensor: torch.Tensor,
    class_names: Sequence[str],
    device: torch.device | str = "cpu",
) -> Prediction:
    """Classify one already-preprocessed tensor.

    The batch-size-1 case of `src/evaluate.collect_predictions`, kept separate
    because there is no DataLoader here and no progress bar worth drawing. The
    softmax-then-argmax order is copied from that function on purpose: softmax is
    monotonic, so the two orders agree, but computing both from one distribution
    makes it impossible for the reported "predicted" and the reported
    "confidence" to come from different distributions.

    Args:
        model: in `eval()` mode, on `device`. Built by `load_checkpoint`, which
            sets both.
        tensor: [1, 3, H, W] float32, normalised.
        class_names: label per output index, in the model's output order. This
            *must* be the checkpoint's list; see the module docstring.
        device: where to run the forward pass.

    Returns:
        The `Prediction`.

    Raises:
        ValueError: `class_names` does not match what the model outputs, or the
            tensor is not a single 3-channel image. Both are silent-wrong-answer
            bugs if allowed through, so both raise.
    """
    model.eval()

    # Checked here rather than left to the first convolution. A 3-D [3, H, W]
    # tensor -- which is exactly what `BrainTumorDataset.__getitem__` returns,
    # and the obvious thing for a caller to pass -- reaches BatchNorm2d as if it
    # were a batch of three unbatched images and fails six frames deep with
    # "expected 4D input (got 3D input)", naming a tensor the caller never
    # mentioned. This names the fix instead.
    if tensor.dim() == 3:
        raise ValueError(
            f"predict_tensor expects a batched tensor [1, 3, H, W], got [3, H, W] "
            f"({tuple(tensor.shape)}). Add the batch dimension with "
            f"`.unsqueeze(0)`, or use predict_file()/load_image(), which do it "
            f"for you."
        )
    if tensor.dim() != 4:
        raise ValueError(
            f"predict_tensor expects a 4-D tensor [1, 3, H, W], got "
            f"{tensor.dim()}-D {tuple(tensor.shape)}."
        )
    if tensor.shape[0] != 1:
        raise ValueError(
            f"predict_tensor classifies one image, so the batch dimension must be "
            f"1, got {tensor.shape[0]}. For several images, loop and call this "
            f"per image, or use src/evaluate.collect_predictions over a DataLoader."
        )
    if tensor.shape[1] != 3:
        raise ValueError(
            f"Expected 3 channels (grayscale scans are replicated to RGB), got "
            f"{tensor.shape[1]} in {tuple(tensor.shape)}."
        )

    with torch.inference_mode():
        logits = model(tensor.to(device))
        probabilities = logits.softmax(dim=1)

    # [1, C] -> [C]. Squeeze the batch dimension: one image in means one row out,
    # and leaving a stray leading 1 here is how `probabilities[0]` ends up being a
    # tensor rather than a float somewhere downstream.
    distribution = probabilities.squeeze(0)

    if distribution.numel() != len(class_names):
        raise ValueError(
            f"Model produced {distribution.numel()} scores but {len(class_names)} "
            f"class names were given ({', '.join(class_names)}). The label list "
            "does not match this model, so every predicted class would be wrong. "
            "Use the class_names stored in the checkpoint."
        )

    # float() on each, so the dataclass holds plain Python numbers and the tensor
    # can be freed rather than pinning a CUDA allocation for the object's life.
    values = [float(value) for value in distribution]
    scores = dict(zip(class_names, values, strict=True))

    # Strictly greater-than everywhere, so argmax returns the *first* maximum on
    # a tie. torch.argmax already does this, so a tie breaks identically here and
    # in src/evaluate.py -- which is what keeps a tied prediction from being
    # scored one way and reported another.
    best = int(distribution.argmax().item())

    return Prediction(
        label=class_names[best],
        class_index=best,
        confidence=values[best],
        probabilities=scores,
        image_size=int(tensor.shape[-1]),
    )


def predict_image(
    path: str | Path,
    model: nn.Module,
    info: CheckpointInfo,
    device: torch.device | str = "cpu",
    *,
    image_size: int | None = None,
) -> Prediction:
    """Load one image and classify it, given an already-loaded model.

    The two halves of inference that always have to happen together: the file is
    read with the checkpoint's own preprocessing, and the checkpoint's own class
    names label the output.

    Args:
        path: the image file.
        model: from `load_checkpoint`, already in `eval()` mode.
        info: the same checkpoint's metadata. Supplies the class names and,
            unless `image_size` overrides it, the resize size.
        device: where to run the forward pass.
        image_size: override the checkpoint's recorded size.

    Returns:
        The `Prediction`.
    """
    size = resolve_image_size(info, image_size)
    tensor = load_image(path, size)
    return predict_tensor(model, tensor, info.class_names, device)


def predict_file(
    path: str | Path,
    checkpoint: str | Path | None = None,
    *,
    device: str = "auto",
    image_size: int | None = None,
) -> Prediction:
    """Classify one image, loading the checkpoint first. The one-call path.

    This is what `predict.py` and `app.py` both use, and it is the function to
    reach for from a REPL:

        from src.inference import predict_file
        predict_file("scan.jpg", checkpoint="models/run1/best.pt")

    Args:
        path: the image file.
        checkpoint: a `.pt` from `src/train.py`. Defaults to the newest
            `models/*/best.pt`, via `evaluate.find_checkpoint`.
        device: `'auto'`, `'cpu'`, `'cuda'`, `'cuda:0'`, `'mps'`.
        image_size: override the checkpoint's recorded size. Only needed for a
            checkpoint that does not carry one.

    Returns:
        The `Prediction`.

    Raises:
        FileNotFoundError: no image, or no checkpoint to use.
        ValueError: the image cannot be read, or its preprocessing is unknown.

    Loading the model per call is deliberate at this scale -- an 11M-parameter
    ResNet is a fraction of a second on a CPU, and the alternative is a module
    level global that makes the function untestable and unsafe to call from two
    threads with two different checkpoints. If this ever runs in a loop, build the
    model once and call `predict_image` instead; that is the whole reason it is
    separate.
    """
    resolved = resolve_device(device)
    checkpoint_path = Path(checkpoint) if checkpoint is not None else find_checkpoint()
    model, info = load_checkpoint(checkpoint_path, resolved)

    logger.info(
        "Loaded %s (arch=%s, epoch=%s) on %s",
        info.path,
        info.arch,
        "?" if info.epoch is None else info.epoch,
        resolved,
    )
    check_preprocessing(info)

    return predict_image(path, model, info, resolved, image_size=image_size)


def predict_files(
    paths: Sequence[str | Path],
    checkpoint: str | Path | None = None,
    *,
    device: str = "auto",
    image_size: int | None = None,
) -> list[Prediction]:
    """Classify several images with one model load.

    Exists so the model is loaded once. It deliberately does not aggregate: there
    is no "average confidence" over a set of scans, and inventing one would give
    a caller a number that means nothing. Returns one `Prediction` per input, in
    the order given.
    """
    resolved = resolve_device(device)
    checkpoint_path = Path(checkpoint) if checkpoint is not None else find_checkpoint()
    model, info = load_checkpoint(checkpoint_path, resolved)

    logger.info(
        "Loaded %s (arch=%s, epoch=%s) on %s",
        info.path,
        info.arch,
        "?" if info.epoch is None else info.epoch,
        resolved,
    )
    check_preprocessing(info)

    return [
        predict_image(path, model, info, resolved, image_size=image_size)
        for path in paths
    ]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _bar(fraction: float, width: int = BAR_WIDTH) -> str:
    """A fixed-width bar for one probability.

    Rounded, not floored: a 0.041 probability at width 24 is 0.98 characters, and
    flooring it to 0 prints an empty bar next to "4.1%", which reads as a bug in
    the number rather than as a rounding artefact. `max(1, ...)` for anything
    visible, so a small-but-nonzero class never disappears from the table -- the
    whole point of listing all four is seeing the small ones.

    Block characters rather than `|`, which double the width for the same
    information.
    """
    filled = int(round(max(0.0, min(1.0, fraction)) * width))
    filled = max(1, filled) if fraction > 0 else 0
    return "#" * filled


def format_report(
    prediction: Prediction,
    *,
    image: str | Path | None = None,
    checkpoint: str | Path | None = None,
    arch: str | None = None,
    epoch: int | None = None,
    show_all: bool = True,
) -> str:
    """Render one prediction as a plain-text report.

    Returned rather than logged or printed, for the same reason
    `src/evaluate.render_report` is: the same string can be printed to a terminal,
    written to a file, or asserted on in a test, and only the caller decides
    which.

    The layout is the one a terminal is good at -- a heading, the two numbers that
    were asked for, then the full distribution underneath. The distribution is not
    optional decoration: a reader who sees only "glioma, 94%" cannot tell a
    confident correct call from a model that is 94% sure and wrong, which is the
    distinction this project most needs to make visible.

    Args:
        prediction: what to render.
        image: the file it came from, for the header.
        checkpoint: the `.pt` used, for the header.
        arch: architecture name, for the header.
        epoch: which epoch the checkpoint is, for the header.
        show_all: include the per-class table. `False` trims the report to the
            headline two lines.

    Returns:
        A multi-line string.
    """
    lines: list[str] = ["Prediction", "=" * len("Prediction"), ""]

    # --- provenance: what answered, and on what ----------------------------
    facts: list[tuple[str, str]] = []
    if image is not None:
        facts.append(("image", str(image)))
    if checkpoint is not None:
        facts.append(("checkpoint", str(checkpoint)))
    if arch is not None:
        epoch_text = "?" if epoch is None else str(epoch)
        facts.append(("model", f"{arch} (epoch {epoch_text})"))
    if facts:
        width = max(len(label) for label, _ in facts)
        lines.extend(f"  {label:<{width}}   {value}" for label, value in facts)
        lines.append("")

    # --- the two numbers the caller asked for ------------------------------
    lines.append(f"  Predicted class   {display_label(prediction.label)}")
    lines.append(f"  Confidence        {prediction.confidence:.2%}")
    lines.append("")

    runner_up = prediction.runner_up
    if runner_up is not None:
        lines.append(
            f"  Next candidate    {display_label(runner_up[0])} at "
            f"{runner_up[1]:.2%}  (margin {prediction.margin:.2%})"
        )
        lines.append("")

    if not show_all:
        lines.append(NOT_A_DEVICE)
        return "\n".join(lines)

    # --- the whole distribution --------------------------------------------
    lines.append("Class probabilities")
    lines.append("-" * len("Class probabilities"))

    ordered = sorted(
        prediction.probabilities.items(), key=lambda item: item[1], reverse=True
    )
    label_width = max((len(display_label(name)) for name, _ in ordered), default=0)

    for name, probability in ordered:
        display = display_label(name)
        # The winner is flagged with a "*" so the eye lands on it while scanning;
        # the ordering alone would otherwise have to be trusted for that.
        marker = "*" if name == prediction.label else " "
        lines.append(
            f" {marker} {display:<{label_width}}  {probability:>7.2%}  "
            f"{_bar(probability)}"
        )
    lines.append("")
    lines.append(f"  * predicted class.  Image resized to {prediction.image_size}x"
                 f"{prediction.image_size}.")
    lines.append("")

    # --- the caveat, attached to the number ---------------------------------
    lines.append(NOT_A_DEVICE)
    lines.append(
        "Confidence is the model's softmax score, not a calibrated probability of "
        "a correct diagnosis."
    )

    return "\n".join(lines)
