"""
app.py -- the Streamlit inference UI.

WHAT THIS FILE IS
-----------------
A browser front end over `src/inference.py`. Upload one MRI scan, see the scan,
see what the model called it and how sure it is.

    python -m src.train          # you need a checkpoint first -- see below
    streamlit run app.py

The single rule this file obeys: **it contains no model logic.** Every number on
screen was computed by `src/inference.py`, and this file only decides where to
put it. The reason is stated in this file's original scaffold docstring and
repeated in `src/README.md`: the UI is the layer most likely to be hand-edited,
and a hand-edited preprocessing line here makes the app disagree with
`python -m src.evaluate` about the same image with nothing to catch it. So there
is no `transforms.Resize` below, no `softmax`, no ImageNet mean, no `argmax`. The
distribution comes from `Prediction.probabilities` and the class name from
`Prediction.label`, both already resolved against the checkpoint.

`predict.py` at the project root is this UI with a terminal attached, over the
same functions. Neither is privileged.

WHY THERE IS NO PREDICT BUTTON
------------------------------
Prediction runs on upload. A "Classify" button is a second click for no benefit
here: inference on one 224x224 image is a single ResNet-18 forward pass, well
under a second on a CPU, so gating it behind a button only adds latency and a way
to look broken while the model loads. The model is loaded once and cached (see
`_load_model`), so later uploads are immediate.

WHY THE UPLOAD GOES TO A TEMPORARY FILE
---------------------------------------
`src/inference.load_image` takes a *path*, not a byte string, and that is
deliberate -- it is also the entry point `predict.py` and any future REST API
would use. Rather than widen it to accept file objects and give one function two
ways in, the upload is written to a temp file and the ordinary path-based call
runs on it. The benefit is that the browser, the CLI, and a batch script are all
provably on the same code path, so a prediction shown here is the same
computation as one shown in a terminal.

The temp file's extension is copied from the uploaded name and checked against
`IMAGE_EXTENSIONS` *before* the write. That check is also what makes the write
safe: the destination filename is built here rather than taken from the upload,
and only a suffix already on the allowlist is ever appended to it, so a hostile
upload name cannot escape the temp directory.

WHY SO MUCH EXPLAINING ABOUT CONFIDENCE
---------------------------------------
Because the number on screen invites a conclusion it cannot support. A softmax
maximum is the model's belief on its own training distribution, not a calibrated
probability of a diagnosis. On a frozen-backbone ResNet-18 with a linear head
trained on a few thousand slices it is systematically overconfident, and on a
scan from a scanner it has never seen it is overconfident *and* wrong. So the UI
shows the full distribution next to the headline number -- 94% against a 1%
runner-up and 54% against a 47% runner-up are the same number and completely
different stories -- and the caveat is rendered on the page rather than left in
this docstring. The README promises calibrated confidence; this model does not
deliver it, and a page showing 94.3% with no qualification would imply otherwise.

This is also why there is no `no_tumor` special case, no hiding of low-confidence
results, and no refusal to answer. The model is allowed to say "I am not sure",
and an uncertain answer shown plainly is more useful than a confident one shown
with the caveat in small print.

WHAT HAPPENS WITH NO CHECKPOINT
-------------------------------
`models/` is empty until `python -m src.train` has been run, so the most likely
first experience with this app is having nothing to load. That state gets a real
panel explaining what to run, not a stack trace. `src/inference.py` also raises
rather than guessing when a checkpoint does not record its image size, and that
error is surfaced verbatim, because the message names the fix.

FRAMEWORK NOTE
--------------
Streamlit, because it is the fastest path to a working inference UI for an image
model. If a REST API is needed instead, swap this file for FastAPI + uvicorn and
leave `src/inference.py` untouched: everything here is an `st.*` call on a value
`src/inference.py` returned, so the port is this file rewritten and nothing else.
"""

from __future__ import annotations

import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
import torch

from src.data.dataset import IMAGE_EXTENSIONS
from src.evaluate import CheckpointInfo, list_checkpoints
from src.inference import (
    NOT_A_DEVICE,
    Prediction,
    check_preprocessing,
    display_label,
    load_checkpoint,
    predict_image,
    resolve_image_size,
)
from src.train import resolve_device

#: Where checkpoints are looked for. Relative to the project root, per the
#: project rule that no path in this repo is absolute or machine-specific.
MODELS_DIR = Path("models")

#: Ceiling on an upload. Nothing in this project needs more, and Streamlit's own
#: 200MB default is generous enough that one file could stall the server. Checked
#: on the byte length, before anything is decoded.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

#: Below this gap between the top two classes, the result is flagged as a near
#: tie. A judgement call, stated as a constant so it can be argued with; the two
#: numbers that triggered it are always on screen next to the message.
TIE_MARGIN = 0.20

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading, cached across Streamlit reruns
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _load_model(
    checkpoint: str,
    device_spec: str,
) -> tuple[torch.nn.Module, CheckpointInfo, torch.device]:
    """Load a checkpoint once and keep it for the life of the server process.

    Streamlit re-executes the whole script on every widget interaction, so
    without this every click on the device dropdown would re-read the checkpoint
    off disk and re-allocate 11M parameters. `cache_resource` rather than
    `cache_data` because the return value is an `nn.Module`: it is kept by
    identity and never pickled, and shared across sessions, which is what makes a
    second browser tab instant.

    Args are `str` because `cache_resource` hashes its arguments.

    Returns:
        (model, info, device) -- everything `predict_image` needs, plus the device
        the model was placed on.

    The returned model is shared mutable state. Nothing here mutates it: it is
    already in `eval()` mode from `load_checkpoint`, and `predict_tensor` calls
    `model.eval()` again defensively, so one session cannot put the shared model
    into training mode out from under another.
    """
    device = resolve_device(device_spec)
    model, info = load_checkpoint(Path(checkpoint), device)
    # Warned once per cache fill rather than once per rerun. That is the right
    # trade: a preprocessing mismatch is a property of the checkpoint, not of the
    # view, and a warning repeated on every interaction is a warning nobody reads.
    check_preprocessing(info)
    return model, info, device


def _available_devices() -> list[str]:
    """Device choices, restricted to backends this machine actually has.

    Offering a device that is not installed would put a value in the picker that
    raises when picked, which reads as a bug in the app rather than a missing
    CUDA build.
    """
    choices = ["auto", "cpu"]
    if torch.cuda.is_available():
        choices.append("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        choices.append("mps")
    return choices


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------


def inject_styles() -> None:
    """Apply the small stylesheet that makes the page read as one thing.

    Deliberately short. Streamlit's DOM is an implementation detail, so heavy CSS
    is a liability: it breaks on upgrade, and the usual failure is silently hiding
    the content a user needs. Everything structural here is done with native
    widgets (`layout="wide"`, `st.metric(border=True)`), and this only adjusts
    spacing, type scale, and the metric card -- all of which degrade to "slightly
    plainer" rather than to "broken" when a selector stops matching.
    """
    st.markdown(
        """
        <style>
            /* The default measure is too narrow for a side-by-side scan and
               prediction; the page layout is already "wide". */
            .block-container { padding-top: 2.2rem; max-width: 1400px; }

            /* Metrics hold the headline numbers, so give them presence. The
               default has no border; a soft one separates them from the page
               without needing a container wrapped round each. */
            [data-testid="stMetric"] {
                border: 1px solid rgba(128, 128, 128, 0.25);
                border-radius: 0.6rem;
                padding: 0.7rem 0.9rem 0.5rem 0.9rem;
            }
            [data-testid="stMetricLabel"] { font-size: 0.78rem; opacity: 0.7; }
            [data-testid="stMetricValue"] { font-size: 1.7rem; font-weight: 650; }

            /* Progress bars carry the per-class text; tighten the stack so four
               of them read as one list instead of four separate widgets. */
            [data-testid="stProgress"] { margin-bottom: 0.1rem; }

            /* Tabular figures keep the percentage columns aligned. */
            [data-testid="stMetricValue"], [data-testid="stProgressText"] {
                font-variant-numeric: tabular-nums;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _header() -> None:
    """The page title and its one-line subtitle."""
    st.markdown(
        """
        <div style="font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em;">
          Brain Tumor MRI Classifier
        </div>
        <div style="color: rgba(135,135,135,1); font-size: 0.98rem;">
          Upload one scan. The model returns a predicted class, a confidence, and
          the full distribution behind them.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _timestamp(path: Path) -> str:
    """Short timestamp for the checkpoint picker."""
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return "?"


def render_sidebar() -> tuple[Path, str, int | None] | None:
    """Build the sidebar and return the chosen settings, or None if unusable.

    Returns None both when there is no checkpoint to choose from (the empty
    state, which the caller renders as a panel) and when the advanced override is
    not a usable number (an input error, which the sidebar has already reported).
    Both are "cannot classify yet", and both are better as a message than as an
    exception.
    """
    st.sidebar.markdown("### Settings")

    checkpoints = list_checkpoints(MODELS_DIR)
    if not checkpoints:
        st.sidebar.error(
            f"No checkpoint in `{MODELS_DIR}/`. Train one: `python -m src.train`"
        )
        return None

    # Newest first in the picker, so the default is the run most likely to be the
    # current one. `list_checkpoints` returns oldest-first, so this reverses
    # rather than re-sorting: the display order is a UI choice, not the loader's.
    ordered = list(reversed(checkpoints))
    labels = [f"{path.parent.name}  ({_timestamp(path)})" for path in ordered]
    index = st.sidebar.selectbox(
        "Checkpoint",
        options=range(len(ordered)),
        format_func=lambda i: labels[i],
        index=0,
        help="Which trained run to classify with, newest first. Each is a "
        "models/<run>/best.pt written by src/train.py.",
    )

    device = st.sidebar.selectbox(
        "Device",
        options=_available_devices(),
        index=0,
        help="'auto' uses a GPU when one is available and falls back to CPU.",
    )

    image_size: int | None = None
    with st.sidebar.expander("Advanced", expanded=False):
        raw = st.text_input(
            "Image size override",
            value="",
            placeholder="from the checkpoint",
            help="Leave empty to use the image size stored in the checkpoint, "
            "which is the correct answer. Set a number only for a checkpoint "
            "that does not record one. The size actually used is shown under the "
            "prediction.",
        )
        if raw.strip():
            try:
                image_size = int(raw)
            except ValueError:
                st.sidebar.error(f"'{raw}' is not a whole number.")
                return None
            if image_size < 1:
                st.sidebar.error("Image size must be at least 1.")
                return None

    st.sidebar.divider()
    st.sidebar.caption(NOT_A_DEVICE)
    return ordered[index], device, image_size


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------


def render_no_checkpoint() -> None:
    """Explain, in the page body, that there is nothing to classify with yet."""
    st.markdown("### No model loaded")
    st.markdown(
        f"The app needs a trained checkpoint, and `{MODELS_DIR}/` has none yet. "
        "This project ships no weights: train one first, then reload this page."
    )
    st.code("python -m src.train", language="bash")
    st.markdown(
        "That writes `models/<run>/best.pt`, which this app then offers in the "
        "sidebar. `python -m src.train --arch custom_cnn` trains the from-scratch "
        "baseline instead, if you want to compare the two."
    )
    st.info(
        "Nothing about the model is guessed in the meantime. In particular the app "
        "will not fall back to a default image size or a default class list: a "
        "wrong guess there produces a confident, wrong answer rather than an "
        "error."
    )


# ---------------------------------------------------------------------------
# The upload
# ---------------------------------------------------------------------------


def save_upload(data: bytes, name: str, directory: Path) -> Path:
    """Write an uploaded scan into `directory` and return its path.

    The destination filename is built here rather than taken from the upload, so
    a client-supplied name cannot influence the path beyond the already-validated
    extension. The extension is preserved because `src/inference.load_image`
    checks it: a PNG that arrived as a JPEG is still readable by PIL, but dropping
    the suffix would make the library reject a perfectly good image.

    Args:
        data: the uploaded bytes.
        name: the original client-supplied filename, used only for its suffix.
        directory: an existing, empty directory owned by the caller.

    Returns:
        The path written.
    """
    suffix = Path(name).suffix.lower()
    destination = directory / f"upload{suffix}"
    destination.write_bytes(data)
    return destination


def render_scan(data: bytes, name: str) -> None:
    """Show the scan in the left column."""
    st.markdown("### Scan")
    st.image(data, caption=name, width="stretch")


# ---------------------------------------------------------------------------
# The prediction
# ---------------------------------------------------------------------------


def render_prediction(prediction: Prediction, info: CheckpointInfo) -> None:
    """Render the class, the confidence, and the distribution behind them.

    Args:
        prediction: what `src/inference` returned. Nothing here computes a
            number; every value is read off the object.
        info: the checkpoint's metadata, for the provenance line.
    """
    st.markdown("### Prediction")

    # The two headline numbers, as bordered metrics so they read as a summary
    # rather than as body text. `border=True` is the native card, so it survives a
    # change to the stylesheet.
    left, right = st.columns(2, gap="small")
    with left:
        st.metric("Predicted class", display_label(prediction.label), border=True)
    with right:
        st.metric(
            "Confidence",
            f"{prediction.confidence:.1%}",
            border=True,
            help="The model's softmax score for the winning class. Not a "
            "calibrated probability of a diagnosis.",
        )

    # A bar for the winner alone, so the headline number has a visual anchor
    # above the list of alternatives it is competing with.
    st.progress(
        float(prediction.confidence),
        text=f"{display_label(prediction.label)}  {prediction.confidence:.1%}",
    )
    st.caption(
        "A softmax score, not a calibrated probability of a diagnosis. Read it "
        "against the distribution below before reading anything into it."
    )

    # --- the whole distribution --------------------------------------------
    st.markdown("**Class probabilities**")
    ordered = sorted(
        prediction.probabilities.items(), key=lambda item: item[1], reverse=True
    )
    for name, probability in ordered:
        # The winner is marked with the same "*" the CLI report uses, rather than
        # by colour alone, so it is still identifiable in a monochrome theme or
        # for a reader who cannot distinguish the hues.
        marker = "*" if name == prediction.label else " "
        st.progress(
            float(probability),
            text=f"{marker} {display_label(name)}   {probability:.1%}",
        )

    runner_up = prediction.runner_up
    if runner_up is not None and prediction.margin < TIE_MARGIN:
        # A near-tie is a genuinely different situation from a clear win, and on a
        # medical classifier it is the case worth surfacing rather than leaving in
        # the bar chart. Both numbers are on screen above this message.
        st.warning(
            f"The top two classes are close: {display_label(prediction.label)} at "
            f"{prediction.confidence:.1%} against {display_label(runner_up[0])} at "
            f"{runner_up[1]:.1%}. A scan the model cannot separate is not a "
            "diagnosis, and this is the case where that matters most."
        )

    _provenance(prediction, info)


def _provenance(prediction: Prediction, info: CheckpointInfo) -> None:
    """Which model, which run, and how the image was preprocessed.

    On the page rather than only in the logs, because a screenshot of a
    prediction is the artifact that travels, and a prediction without its model
    attached is a number nobody can check.
    """
    st.divider()
    st.caption(
        f"Model `{info.arch}` · epoch "
        f"{info.epoch if info.epoch is not None else '?'} · "
        f"{len(info.class_names)} classes · resized to "
        f"{prediction.image_size}×{prediction.image_size} · checkpoint "
        f"`{info.path.parent.name}/best.pt`"
    )
    st.caption(NOT_A_DEVICE)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def classify(upload: Any, checkpoint: Path, device_spec: str, image_size: int | None) -> Prediction:
    """Classify one upload. Thin wrapper over the cached model plus the library.

    Separated from `main` so the whole upload -> predict path is one call that a
    test can drive without a Streamlit runtime.
    """
    model, info, device = _load_model(str(checkpoint), device_spec)
    size = resolve_image_size(info, image_size)
    with tempfile.TemporaryDirectory() as directory:
        path = save_upload(upload["data"], upload["name"], Path(directory))
        return predict_image(path, model, info, device, image_size=size)


def main() -> None:
    """Run the Streamlit app."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Must be the first Streamlit call in the script: everything else depends on
    # the page config being set before the first element is created.
    st.set_page_config(
        page_title="Brain Tumor MRI Classifier",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    settings = render_sidebar()
    if settings is None:
        _header()
        render_no_checkpoint()
        return
    checkpoint, device_spec, image_size_override = settings

    _header()

    uploaded = st.file_uploader(
        "Upload an MRI scan",
        type=[suffix.lstrip(".") for suffix in IMAGE_EXTENSIONS],
        help="One image per scan: JPEG, PNG, BMP, TIFF, or WebP. DICOM and NIfTI "
        "are not supported -- export a slice to PNG first.",
    )
    if uploaded is None:
        st.caption(
            "Waiting for a scan. The model runs automatically once one is "
            "uploaded, so there is no button to press."
        )
        return

    # Read the bytes once. `getvalue()` is not free, and three separate readers
    # of it (the size check, the preview, and the write) would be three copies.
    data = uploaded.getvalue()
    name = uploaded.name

    if not data:
        st.error("That file is empty.")
        return
    if len(data) > MAX_UPLOAD_BYTES:
        st.error(
            f"That file is {len(data) / 1024 / 1024:.1f} MB, over the "
            f"{MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB limit. A single MRI slice "
            "should be far smaller, so this is probably not a slice."
        )
        return

    scan_column, prediction_column = st.columns([1, 1], gap="large")

    with st.spinner("Loading the model and classifying..."):
        try:
            prediction = classify(
                {"data": data, "name": name},
                checkpoint,
                device_spec,
                image_size_override,
            )
            _, info, _ = _load_model(str(checkpoint), device_spec)
        except FileNotFoundError as error:
            scan_column.empty()
            st.error(str(error))
            return
        except ValueError as error:
            # The library's messages name the fix ("pass an image size", "this
            # file could not be decoded"), so they are shown rather than replaced
            # with something vaguer.
            scan_column.empty()
            st.error(str(error))
            return
        except RuntimeError as error:
            scan_column.empty()
            st.error(f"The model could not be loaded: {error}")
            return

    with scan_column:
        render_scan(data, name)
    with prediction_column:
        render_prediction(prediction, info)


if __name__ == "__main__":
    main()
