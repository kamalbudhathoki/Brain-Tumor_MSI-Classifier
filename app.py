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

Counting the model's parameters is the one thing below that touches the network
at all, and it is a read of `model.parameters()` rather than a second opinion
about anything the prediction depends on: `count_parameters` is about how big
the model is, not about what it says.

`predict.py` at the project root is this UI with a terminal attached, over the
same functions. Neither is privileged.

WHAT IS ON THE PAGE
-------------------
Top to bottom, and why in that order:

    Header              what this is, and the not-a-device line, before anything
                        that could be mistaken for a result.
    Sidebar             which checkpoint, which device, the override, what is
                        loaded, and what this session has seen.
    Upload              one scan, in a card, with its accepted formats.
    Result              scan on the left, prediction on the right, so the thing
                        being judged and the judgement are read together.
    Model information   the checkpoint's own metadata: architecture, parameter
                        counts, provenance, and the preprocessing it dictates.
    Prediction history  what this session has already classified, newest first.

Model information comes *before* history because it is a property of the model
and does not change between scans, while the history grows with every upload: a
fixed fact above a growing list is the order a reader can predict.

WHY THE MODEL IS LOADED BEFORE ANYTHING IS UPLOADED
----------------------------------------------------
The original version loaded the checkpoint lazily, on the first upload, and put
the provenance in a caption under the result. That makes the page useless until
the reader already has a scan in hand: the first question anyone opens a model
UI with is "what am I about to trust?", and it had no answer. Loading first costs
the same wall-clock time -- it is paid either way -- and buys a page that
describes itself before it is used. It also means a checkpoint that cannot be
loaded fails in a panel that says so, rather than after someone has picked a
scan.

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

WHY THE HISTORY IS SESSION-ONLY AND SAYS SO
-------------------------------------------
`st.session_state` is per browser tab and dies with it. That is the right scope
for a convenience list and the wrong scope for a record, so the page says so
next to the table rather than letting a screenshot imply that the rows are a log
of anything. Nothing is written to disk; there is no SQLite file, no append-only
JSONL, and no way to get a scan out of the browser except the CSV button, which
carries its own provenance columns for exactly that reason.

The stored rows deliberately keep no image bytes. A session that classified 25
scans would otherwise hold 25 decoded images in server memory for the life of
the tab, and the history is the one part of the page that grows without limit,
so it is the part that has to stay small. The filename, the label, the
confidence, the whole distribution and the checkpoint name are the numbers; the
scan itself is still on the reader's disk.

The history also has no mean confidence. `src/inference.py` deliberately refuses
to average confidences across images -- an average over a set of scans means
nothing about any of them -- and adding one here would put the number the module
argues against into the one place a reader is most likely to quote it. What it
counts instead is near-ties, which is a count of *decisions* rather than an
average of beliefs, and the per-class breakdown, which is a census of what the
session's scans were called.

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

WHY THE STYLESHEET IS STILL SHORT
---------------------------------
The styling work here is the part of this file most likely to rot, so it stays
small and everything structural is done with widgets that survive it. Layout is
`layout="wide"` and `st.columns`; the cards are `st.container(border=True)` and
`st.metric(border=True)`; the responsive behaviour is Streamlit's own column
wrapping, which the CSS does not touch. The stylesheet sets spacing, a type
scale, and the metric card, and every one of those degrades to "slightly
plainer" when a selector stops matching -- which it will, at some upgrade,
because these are not stable API.

Every selector that is used was checked against the installed Streamlit rather
than copied from an older app: `[data-testid="stProgressText"]`, which the
previous version styled, no longer exists in 1.64, so that rule was dropped
instead of being carried forward as dead CSS. Raw elements (`hr`) are used where
a component has no testid to target. Section headings are rendered as `<h3>` with
a class rather than as `###` markdown, because Streamlit's heading margins are
set in JavaScript and the vertical rhythm here needs to be one value in one
place.

Responsive layout is mostly not this file's job. `st.columns` wraps to full-width
rows at narrow viewports on its own, the sidebar becomes an overlay, and
`st.dataframe` scrolls rather than overflowing -- so the page is usable on a phone
because of the framework, and the media query at the bottom only shrinks the
type scale and the page padding so the phone is not left with desktop-sized
headings.

FRAMEWORK NOTE
--------------
Streamlit, because it is the fastest path to a working inference UI for an image
model. If a REST API is needed instead, swap this file for FastAPI + uvicorn and
leave `src/inference.py` untouched: everything here is an `st.*` call on a value
`src/inference.py` returned, so the port is this file rewritten and nothing else.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import logging
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn

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
#: 200MB default is generous enough that one file could stall the server. Passed
#: to the uploader as well as checked here after the read: the widget's limit
#: rejects the file in the browser, and the check below covers a caller that
#: hands bytes in directly (`classify` is testable that way).
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

#: Below this gap between the top two classes, the result is flagged as a near
#: tie. A judgement call, stated as a constant so it can be argued with; the two
#: numbers that triggered it are always on screen next to the message. The
#: history marks its rows with the same constant through `_is_tie`, so the flag
#: in the table and the warning above it cannot disagree.
TIE_MARGIN = 0.20

#: How many predictions one session keeps. The list is a convenience, not a
#: record (see the docstring), and a table that grows without bound is the part
#: of a page that pushes everything else off the screen. The count of everything
#: classified is kept separately, so a truncated table can still say what it is
#: not showing.
HISTORY_LIMIT = 25

#: `st.session_state` keys. Named constants because they are written in one
#: function and read in another, and a typo in a session-state key does not
#: raise -- it silently creates a second list.
HISTORY_KEY = "prediction_history"
HISTORY_TOTAL_KEY = "prediction_history_total"

#: Explicit keys for the two clear buttons. They run the same callback and would
#: otherwise be the same widget drawn twice, which Streamlit rejects. Also the
#: uploader's, so the widget's state can be addressed by name.
UPLOAD_KEY = "scan_upload"
CLEAR_KEY = "clear_history_sidebar"
CLEAR_KEY_BODY = "clear_history_body"

#: Viewport width, in CSS pixels, below which the page stops being a two-column
#: desktop layout. Not a magic number of our own: Streamlit's own column CSS
#: wraps at its `breakpoints.columns` theme value, and this is the same
#: threshold, so the type scale changes over at the same moment the columns
#: reflow. `st.get_option("theme.base")` is deliberately not consulted -- the
#: light and dark themes do not change the layout, and a dark-mode branch here
#: would be one more thing to keep in sync.
#:
#: Substituted into `STYLESHEET` rather than typed into it, so the number the CSS
#: uses and the number documented above cannot drift apart.
WIDE_LAYOUT_BREAKPOINT = 640

#: `st.container(key=...)` puts an `st-key-<key>` class on the container's DOM
#: element, which is the one supported way for a stylesheet to target a specific
#: piece of this page instead of every element of its kind. The scan is the only
#: image the app shows, so this is belt-and-braces -- but a selector scoped to
#: this container cannot accidentally catch a future one.
SCAN_CONTAINER_KEY = "scan"

logger = logging.getLogger(__name__)


#: Attached under the history in both its states, including the empty one, because
#: the scope of the list is the thing a reader is most likely to over-read: a
#: table of predictions is exactly what a log of diagnoses looks like. Defined
#: next to the other page-wide strings rather than beside `render_history`,
#: because `render_sidebar_session` says a shorter version of the same thing.
SCOPE_NOTE = (
    "Session-only. These predictions are not written to disk and are gone when "
    "this tab closes; nothing here is a record of a patient, and a row is only "
    "as good as the scan it came from. There is deliberately no average "
    "confidence: an average over a set of scans says nothing about any of them."
)


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

    Raises:
        Exception: whatever reading the checkpoint raises, deliberately unwrapped.
            The file lives in `models/`, so it is a file the user put there, and
            what a bad one raises depends on how it is bad: a truncated archive
        raises from `pickle`, a `state_dict` that no longer fits its architecture
            raises from `torch.nn`, an unexpected dtype raises from `torch.load`.
            `main` reports all of them the same way, and this function has nothing
            useful to add to the message.

    The returned model is shared mutable state. Nothing here mutates it: it is
    already in `eval()` mode from `load_checkpoint`, and `predict_tensor` calls
    `model.eval()` again defensively, so one session cannot put the shared model
    into training mode out from under another.

    An exception is not cached, so fixing or deleting the file and rerunning the
    app is enough to recover; there is no poisoned cache entry to clear.
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
    raises when picked, which reads as a bug in the app rather than as a missing
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

#: The page's stylesheet, with `@breakpoint@` standing in for
#: `WIDE_LAYOUT_BREAKPOINT`. A plain string rather than an f-string because CSS is
#: almost entirely braces, and doubling every one of them to satisfy `str.format`
#: would make the sheet harder to edit than the single `.replace()` is.
STYLESHEET = """
<style>
    /* -- page ------------------------------------------------------ */
    /* The default measure is too narrow for a side-by-side scan and
       prediction; the page layout is already "wide". Generous bottom
       padding so the last section is not flush against the viewport
       edge. */
    .block-container {
        padding-top: 2.4rem;
        padding-bottom: 4rem;
        max-width: 1500px;
    }

    /* -- headings -------------------------------------------------- */
    /* Streamlit sets heading margins in JavaScript, so the section rhythm
       is a class on an <h3> rendered by `_section` instead. The element
       selector is scoped to the block container so the sidebar's own
       markdown headings pick up the tracking without losing their
       spacing. */
    .block-container h3 {
        letter-spacing: -0.015em;
        margin-top: 0;
        margin-bottom: 0.15rem;
    }
    .section-subtitle {
        color: rgba(135, 135, 135, 1);
        font-size: 0.9rem;
        margin: 0 0 0.9rem 0;
        line-height: 1.45;
    }

    /* -- sections -------------------------------------------------- */
    /* Space between the page's top-level sections, which no widget owns:
       `st.container(gap=...)` only knows about the inside of a card. Scoped
       to the main block container's direct children, and to all but the last
       of them, so it adds a rhythm between sections without also inflating
       the gap inside every bordered card.

       Note the `.block-container ` prefix on every framework selector below.
       Streamlit styles its components with emotion, i.e. with a generated
       single class, so a bare `[data-testid="x"]` rule and an emotion rule
       have *equal* specificity and the winner is decided by which one the
       browser happens to see last. Prefixing with a class makes these rules
       win deterministically, which matters more than it sounds: the same
       prefixing is what stops a decoration here from quietly becoming a
       layout bug after an upgrade moves the style tags around. */
    .block-container > [data-testid="stVerticalBlock"]
        > [data-testid="stElementContainer"]:not(:last-child) {
        margin-bottom: 1.4rem;
    }

    /* -- metrics --------------------------------------------------- */
    /* Metrics hold the headline numbers, so give them presence. The
       default has no border; a soft one separates them from the page
       without needing a container wrapped round each. `border=True` on
       st.metric is the native equivalent and is used as well, so the
       cards survive this rule being dropped. */
    .block-container [data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 0.6rem;
        padding: 0.7rem 0.9rem 0.6rem 0.9rem;
    }
    .block-container [data-testid="stMetricLabel"] {
        font-size: 0.78rem;
        letter-spacing: 0.01em;
        opacity: 0.7;
    }
    .block-container [data-testid="stMetricValue"] {
        font-size: 1.7rem;
        font-weight: 650;
        font-variant-numeric: tabular-nums;
    }
    /* The delta line is used as a caption on a few cards ("2,052
       trainable"), so it is set at caption size rather than as a change
       indicator. */
    .block-container [data-testid="stMetricDelta"] {
        font-size: 0.78rem;
    }

    /* -- the scan -------------------------------------------------- */
    /* A portrait slice rendered at full column height is tall enough to
       push the prediction it explains below the fold. The class comes
       from st.container(key=SCAN_CONTAINER_KEY); `object-fit` keeps the
       aspect ratio rather than stretching it. */
    .st-key-scan img {
        max-height: 460px;
        width: 100%;
        object-fit: contain;
        background: rgba(128, 128, 128, 0.06);
        border-radius: 0.5rem;
    }
    .block-container [data-testid="stImageCaption"] {
        font-size: 0.78rem;
    }

    /* -- progress bars --------------------------------------------- */
    /* The bars carry the per-class text; tighten the stack so four of
       them read as one list instead of four separate widgets. (The old
       `stProgressText` rule went when the testid did; the percentage is
       inside the bar, not in a separate element to size.) */
    .block-container [data-testid="stProgress"] { margin-bottom: 0.1rem; }
    .block-container [data-testid="stProgressBarTrack"] { padding: 0.1rem 0; }

    /* -- rules ----------------------------------------------------- */
    /* `hr` rather than a testid: st.divider() has carried a different one
       across Streamlit versions, and a plain element selector is the only
       one that has not. Best-effort -- if Streamlit's own divider margins
       win, the rule is simply ignored. */
    .block-container hr {
        margin: 1.2rem 0;
        border: 0;
        border-top: 1px solid rgba(128, 128, 128, 0.25);
    }

    /* -- fact tables ----------------------------------------------- */
    /* The key/value lists in the sidebar and in the model card. A grid
       rather than a <table> so the key column sizes to the longest label
       and the value column takes what is left, wrapping instead of
       overflowing on a narrow sidebar. */
    .facts {
        display: grid;
        grid-template-columns: max-content minmax(0, 1fr);
        column-gap: 0.75rem;
        row-gap: 0.2rem;
        font-size: 0.82rem;
        align-items: baseline;
    }
    .facts dt { opacity: 0.65; white-space: nowrap; }
    .facts dd {
        margin: 0;
        font-variant-numeric: tabular-nums;
        overflow-wrap: anywhere;
    }

    /* -- narrow viewports ------------------------------------------ */
    /* Streamlit's own columns wrap at its `breakpoints.columns` theme
       value; this only brings the type scale down to meet them, so the
       page does not show desktop-sized headings above a single-column
       stack. Everything structural above is already width-agnostic. */
    @media (max-width: @breakpoint@px) {
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
        }
        .block-container [data-testid="stMetricValue"] { font-size: 1.4rem; }
        .block-container [data-testid="stMetric"] {
            padding: 0.55rem 0.7rem 0.5rem 0.7rem;
        }
        .section-subtitle { font-size: 0.85rem; }
        .st-key-scan img { max-height: 320px; }
    }
</style>
"""


def inject_styles() -> None:
    """Apply the small stylesheet that makes the page read as one thing.

    Three jobs, and nothing else: vertical rhythm, a type scale for the metric
    cards, and a mobile pass. Structure is left entirely to widgets -- the
    columns, the bordered containers and the card borders are all native, so if
    every selector in here stops matching tomorrow the page is plainer and still
    correct, which is the only acceptable way for CSS in this project to fail.
    """
    st.markdown(
        STYLESHEET.replace("@breakpoint@", str(WIDE_LAYOUT_BREAKPOINT)),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Headings and fact tables
# ---------------------------------------------------------------------------


class SectionTitle(NamedTuple):
    """A section heading and the id that goes with it.

    The id exists for the history table's per-row links, which point at
    `#section-<slug>` in this same page. It is derived rather than hand-written
    so a heading cannot get an id that no longer matches its text.
    """

    full: str
    anchor: str


def _slug(text: str) -> str:
    """A stable, URL-safe fragment id for a heading."""
    cleaned = "".join(character.lower() if character.isalnum() else "-" for character in text)
    return "section-" + "-".join(part for part in cleaned.split("-") if part)


def _section(title: str, subtitle: str) -> SectionTitle:
    """Render a section heading: the title, then one line saying what it is for.

    The subtitle is not decoration. Every panel on this page is showing a number
    that means something narrower than it looks, and a one-line statement of what
    the panel is for is what stops the next panel's number being read as an
    answer to the wrong question.
    """
    section = SectionTitle(title, _slug(title))
    st.markdown(
        f'<h3 class="section-title" id="{section.anchor}">{html.escape(title)}</h3>'
        f'<p class="section-subtitle">{html.escape(subtitle)}</p>',
        unsafe_allow_html=True,
    )
    return section


def _fact_rows(rows: Sequence[tuple[str, str]]) -> str:
    """A key/value list as a definition grid, ready for `st.markdown`.

    A grid rather than a markdown table or a `st.dataframe`: the values are
    static text, and the two structured widgets would add a scrollbar or a
    sortable header to a list that has neither problem. Every value is escaped,
    because these strings come out of a checkpoint -- a class name containing an
    angle bracket would otherwise break the layout of everything below it.
    """
    cells = []
    for label, value in rows:
        cells.append(f"<dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd>")
    return f'<dl class="facts">{"".join(cells)}</dl>'


def _record(value: object) -> str:
    """Format an optional checkpoint field, or say plainly that it is absent.

    "not recorded" rather than a dash, because a dash next to a real value is
    read as "the same, and zero" and a missing epoch is not that.
    """
    return "not recorded" if value is None else str(value)


def _file_size(path: Path) -> str:
    """A file's size in the largest unit that leaves a number above 1."""
    try:
        size = path.stat().st_size
    except OSError:
        # The checkpoint was listed a moment ago and the picker still offers it,
        # so this is a race or a permissions problem -- either way, a missing
        # size must not take the model panel down with it.
        return "?"
    if size < 1024:
        return f"{size} B"
    if size < 1024**2:
        return f"{size / 1024:.1f} KB"
    return f"{size / 1024**2:.1f} MB"


def _count_parameters(model: nn.Module) -> tuple[int, int]:
    """(total, trainable) parameter counts.

    Summed from `model.parameters()` rather than read from the architecture's own
    `num_parameters` method, which both of this project's models have: a
    checkpoint can name any architecture, and `load_checkpoint` returns an
    `nn.Module` whose class this file has no reason to know about. Calling the
    method where it exists would mean either a cast or a `hasattr` branch for a
    number that costs two lines to count directly.

    The split is worth showing separately because it is the whole story about
    this model: the ImageNet backbone carries the representation and the 2,052
    trainable weights in the head are the entire learned contribution of the
    fine-tune. A single "11,178,564 parameters" would read as a model that trained
    all of it.
    """
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def _header() -> None:
    """The page title, its one-line subtitle, and the not-a-device line.

    The disclaimer is above the first upload control rather than at the bottom of
    the page. On a medical classifier the caveat has to be on screen at the moment
    the number is, and the result panel repeats it under the confidence for the
    screenshot case.
    """
    st.markdown(
        """
        <div style="font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em;">
          Brain Tumor MRI Classifier
        </div>
        <div style="color: rgba(135,135,135,1); font-size: 0.98rem; margin-top: 0.2rem;">
          Upload one scan. The model returns a predicted class, a confidence, and
          the full distribution behind them.
        </div>
        <div style="color: rgba(135,135,135,1); font-size: 0.82rem; margin-top: 0.7rem;">
          Research output only -- not for diagnosis or treatment decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """What the sidebar chose: the checkpoint to classify with, and how to run it.

    A dataclass rather than the positional triple this function used to return:
    four call sites had to remember the order, and a swapped `device` and
    `image_size` is a mistake that still runs. Frozen, like everything else here
    that a reader might be tempted to edit in place.
    """

    checkpoint: Path
    device: str
    image_size: int | None


def _timestamp(path: Path) -> str:
    """Modification time of a checkpoint, for the picker and the model card."""
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return "?"


def render_sidebar() -> Settings | None:
    """Build the sidebar's controls and return the chosen settings, or None.

    Returns None both when there is no checkpoint to choose from (the empty
    state, which the caller renders as a panel) and when the advanced override is
    not a usable number (an input error, which the sidebar has already reported).
    Both are "cannot classify yet", and both are better as a message than as an
    exception.

    Only the *controls* live here. The loaded-model summary and the session
    counts are rendered by separate calls afterwards, because both need the
    checkpoint to be loaded first and Streamlit draws the sidebar in call order.
    """
    st.sidebar.caption(NOT_A_DEVICE)
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
    st.sidebar.divider()
    with st.sidebar.expander("Advanced", expanded=False):
        raw = st.text_input(
            "Image size override",
            value="",
            placeholder="from the checkpoint",
            help="Leave empty to use the image size stored in the checkpoint, "
            "which is the correct answer. Set a number only for a checkpoint "
            "that does not record one. The size actually used is shown in the "
            "model information panel.",
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

    return Settings(ordered[index], device, image_size)


def render_sidebar_model(
    model: nn.Module,
    info: CheckpointInfo,
    device: torch.device,
    settings: Settings,
) -> None:
    """The sidebar's one-screen answer to "what is loaded?".

    Deliberately a summary and not the full model card: the sidebar is narrow, and
    the model information section in the page body has the parameter counts, the
    class list and the preprocessing. What belongs up here is the thing a reader
    wants without scrolling -- which run, how big, on what, at what input size --
    because that is what they check before trusting a number, and it is what they
    would otherwise have to look up in the picker they just used.
    """
    total, trainable = _count_parameters(model)
    st.sidebar.divider()
    st.sidebar.markdown("### Loaded model")
    st.sidebar.caption(f"`{info.path.parent.name}/best.pt`")
    st.sidebar.markdown(
        _fact_rows(
            [
                ("architecture", info.arch),
                ("parameters", f"{total:,} ({trainable:,} trainable)"),
                ("classes", str(info.num_classes)),
                ("input size", _input_size_text(info, settings)),
                ("checkpoint", _file_size(info.path)),
                ("trained", _timestamp(info.path)),
                ("device", str(device)),
                ("mode", "eval" if not model.training else "TRAINING"),
            ]
        )
    )
    if model.training:
        # Should be unreachable -- `load_checkpoint` calls `eval()` and nothing
        # here sets it back -- but a model in training mode is a different number
        # entirely, and it is worth saying so on the page rather than only in a
        # log line nobody opens.
        st.sidebar.error("The model is in training mode. Predictions are unreliable.")


def render_sidebar_session() -> None:
    """What this session has classified so far, and a way to forget it.

    In the sidebar rather than only under the history table because it is the
    number that answers "how much have I actually done?", which is asked before
    the reader has scrolled to the bottom of the page. The clear button appears
    in both places on purpose: the same callback, two keys, so the control is
    under the cursor of whoever decided they wanted it.
    """
    state = read_history()
    st.sidebar.divider()
    st.sidebar.markdown("### This session")
    if state.entries:
        st.sidebar.markdown(
            _fact_rows(
                [
                    ("scans", str(state.total)),
                    ("near-ties", f"{state.near_ties} of {len(state.entries)}"),
                    ("classes", ", ".join(display_label(n) for n in state.distinct_labels)),
                ]
            )
        )
    else:
        st.sidebar.caption("Nothing classified yet.")
    st.sidebar.button(
        "Clear history",
        on_click=clear_history,
        key=CLEAR_KEY,
        disabled=not state.entries,
        width="stretch",
    )
    st.sidebar.caption(
        "Session-only: these predictions live in this browser tab, are not "
        "written to disk, and are gone when the tab is closed."
    )


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


def render_load_error(error: Exception) -> None:
    """Report a checkpoint that could not be loaded, without a traceback.

    The model is loaded before any upload, so this is where a checkpoint that is
    unreadable, is not one this project wrote, or whose state dict does not fit
    its architecture is reported -- and it is reported on a page the reader can
    still act on, rather than as the whole app failing.

    Only the first line of the message is shown. `torch.load` failures arrive
    wrapped in paragraphs explaining that `weights_only` changed in PyTorch 2.6
    and inviting a bug report, none of which helps someone whose `best.pt` was
    truncated; the full text is in the terminal running `streamlit run`, which
    `main` logs it to.
    """
    detail = " ".join(str(error).split())
    if len(detail) > 200:
        detail = detail[:200].rstrip() + "..."
    if not detail:
        detail = type(error).__name__
    st.error(f"The model could not be loaded: {detail}")
    st.markdown(
        "The checkpoint is listed in the sidebar but could not be rebuilt. The "
        "message above is what `src/evaluate.load_checkpoint` raised; the usual "
        "causes are a half-written `best.pt` from an interrupted run, or an "
        "architecture whose constructor defaults changed after the run. Retrain, "
        "or delete the run directory, then reload this page."
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


def upload_problem(data: bytes) -> str | None:
    """Why these bytes cannot be classified, or None if they can be.

    Checked on the byte length, before anything is decoded, so a 400MB file costs
    a comparison rather than a PIL error. The uploader is given the same limit as
    `max_upload_size`, so this is the second of two checks rather than the only
    one -- the widget's limit rejects in the browser, and this covers a caller
    that hands bytes in directly.

    A pure function of the bytes, so the size rule is testable without a
    Streamlit runtime.
    """
    if not data:
        return "That file is empty."
    if len(data) > MAX_UPLOAD_BYTES:
        return (
            f"That file is {len(data) / 1024 / 1024:.1f} MB, over the "
            f"{MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB limit. A single MRI slice "
            "should be far smaller, so this is probably not a slice."
        )
    return None


def render_uploader() -> Any:
    """The upload control, in a card, with what it accepts written under it."""
    with st.container(border=True):
        uploaded = st.file_uploader(
            "Upload an MRI scan",
            type=[suffix.lstrip(".") for suffix in IMAGE_EXTENSIONS],
            key=UPLOAD_KEY,
            max_upload_size=MAX_UPLOAD_BYTES,
            help="One image per scan: JPEG, PNG, BMP, TIFF, or WebP. DICOM and NIfTI "
            "are not supported -- export a slice to PNG first.",
        )
        st.caption(
            f"{', '.join(sorted(suffix.lstrip('.').upper() for suffix in IMAGE_EXTENSIONS))}"
            f" · up to {MAX_UPLOAD_BYTES // (1024 * 1024)} MB · the model runs as "
            "soon as a file is chosen, so there is no button to press."
        )
    return uploaded


def render_waiting() -> None:
    """The page's state before there is anything to classify."""
    st.info(
        "Waiting for a scan. Everything else on this page -- the model "
        "information below, the sidebar's summary -- is already available; only "
        "the result and the history need an image."
    )


def upload_facts(data: bytes) -> str:
    """A one-line description of what was actually uploaded: size, format, pixels.

    Worth the two lines because it answers the question the resize raises. A
    reader who sees the scan described as 512×512 and the result described as
    "resized to 224×224" can see what the model was given; without it the resize
    looks like a change to the image rather than a change to its size.

    `Image.open` reads the header and stops -- the pixels are not decoded, and the
    file is not rewritten -- so this costs a few hundred bytes of I/O. A failure
    here is not a problem: the uploader already restricted the extension, and
    `src/inference` reports a real decode failure properly, so the worst case is
    a caption with the byte size alone.
    """
    size = f"{len(data) / 1024:.1f} KB"
    try:
        with Image.open(io.BytesIO(data)) as handle:
            width, height = handle.size
            image_format = handle.format
    except (UnidentifiedImageError, OSError, ValueError):
        return size
    return f"{width}×{height} · {image_format} · {size}"


def render_scan(data: bytes, name: str) -> None:
    """Show the scan in the left column, at its own size.

    The `key` is what the stylesheet's `.st-key-scan` selector hangs off, and
    nothing else: the height cap on the image is the reason this container
    exists rather than a bare `st.image`.
    """
    st.markdown("### Scan")
    with st.container(key=SCAN_CONTAINER_KEY):
        st.image(data, caption=name, width="stretch")
    st.caption(upload_facts(data))


# ---------------------------------------------------------------------------
# The prediction
# ---------------------------------------------------------------------------


def _is_tie(prediction: Prediction) -> bool:
    """True when the top two classes are too close to call.

    One rule, used both by the warning under the distribution and by the history
    table's flag column, so the two can never disagree about which rows were
    ambiguous. The threshold is `TIE_MARGIN` and the two numbers that triggered it
    are always on screen next to the message.
    """
    return prediction.runner_up is not None and prediction.margin < TIE_MARGIN


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
    if runner_up is not None and _is_tie(prediction):
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
    attached is a number nobody can check. The full version of this -- parameter
    counts, the class list, the normalisation -- is in the model information
    panel below; this is the one line that has to survive being cropped.
    """
    st.divider()
    st.caption(
        f"Model `{info.arch}` · epoch "
        f"{info.epoch if info.epoch is not None else '?'} · "
        f"{len(info.class_names)} classes · resized to "
        f"{prediction.image_size}×{prediction.image_size} · checkpoint "
        f"`{info.path.parent.name}/best.pt`"
    )


# ---------------------------------------------------------------------------
# Model information
# ---------------------------------------------------------------------------


def _input_size_text(info: CheckpointInfo, settings: Settings) -> str:
    """The input size actually in force, and where it came from.

    "not recorded" is a real state, not an error to hide: a checkpoint that does
    not carry an image size cannot have its preprocessing reproduced, and
    `src.inference.resolve_image_size` raises rather than guessing. The sidebar
    override is shown as the source when it is the reason, so a size that came
    from a text box does not look like one that came from the checkpoint.
    """
    if settings.image_size is not None:
        return f"{settings.image_size}×{settings.image_size} (overridden)"
    if info.image_size >= 1:
        return f"{info.image_size}×{info.image_size}"
    return "not recorded"


def _mean_std(values: Sequence[float] | None) -> str:
    """A normalisation constant list, or a plain statement that it is missing."""
    if not values:
        return "not recorded"
    return ", ".join(f"{value:g}" for value in values)


def _metric_text(values: Mapping[str, Any]) -> str:
    """The training metrics a checkpoint recorded, in one line.

    These are the numbers from the best epoch of a *training* run, kept because
    they are the only indication of how the run went that the checkpoint carries.
    Labelled as such on the page: they are not a test score, and quoting them as
    one is how a training number ends up in a paper.
    """
    if not values:
        return "not recorded"
    parts = []
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            parts.append(f"{key} {value}")
        else:
            parts.append(f"{key} {value:.4f}")
    return ", ".join(parts)


def render_model_info(
    model: nn.Module,
    info: CheckpointInfo,
    device: torch.device,
    settings: Settings,
) -> None:
    """Everything the checkpoint says about itself, in one card.

    Not a model card -- there is no trained checkpoint in this repository to write
    one for -- but the part of one that is a matter of record: what the model is,
    how big it is, where it came from, and what preprocessing it insists on. Every
    value is read out of the checkpoint or off the loaded module. Nothing here is
    asserted about the model's quality, because nothing on this page has measured
    it: the only number that would be is the test score from
    `python -m src.evaluate`, and the caption says so rather than implying this
    panel supplies one.
    """
    total, trainable = _count_parameters(model)
    share = (trainable / total * 100) if total else 0.0

    with st.container(border=True, gap="medium"):
        _section(
            "Model information",
            "The checkpoint this page classifies with, as it describes itself. "
            "Read the preprocessing rows before trusting any number above.",
        )

        top = st.columns(4, gap="medium")
        with top[0]:
            st.metric(
                "Architecture",
                info.arch,
                border=True,
                help="The registry key saved in the checkpoint, as written by "
                "src/train.py's save_checkpoint.",
            )
        with top[1]:
            st.metric(
                "Parameters",
                f"{total:,}",
                delta=f"{trainable:,} trainable ({share:.1f}%)",
                delta_color="off",
                border=True,
                help="The whole network, and how much of it this fine-tune "
                "actually moved. The rest came from the ImageNet backbone and "
                "is frozen.",
            )
        with top[2]:
            st.metric("Classes", str(info.num_classes), border=True)
        with top[3]:
            st.metric(
                "Input",
                f"{info.image_size}×{info.image_size}"
                if info.image_size >= 1
                else "not recorded",
                border=True,
                help="Square side length the image is resized to before the "
                "forward pass, from the checkpoint.",
            )

        body = st.columns([1, 1], gap="large")
        with body[0]:
            st.markdown("**Checkpoint**")
            st.markdown(
                _fact_rows(
                    [
                        ("run", info.path.parent.name),
                        ("path", str(info.path)),
                        ("size on disk", _file_size(info.path)),
                        ("last written", _timestamp(info.path)),
                        ("epoch", _record(info.epoch)),
                        ("seed", _record(info.config.get("seed"))),
                        (
                            "augmentation",
                            _record(info.config.get("augment")),
                        ),
                        (
                            "best-epoch metrics",
                            _metric_text(info.train_metrics),
                        ),
                    ]
                )
            )
            st.caption(
                "`best-epoch metrics` is what the training run recorded for the "
                "epoch this checkpoint came from. It is not a test score."
            )
        with body[1]:
            st.markdown("**Preprocessing**")
            st.markdown(
                _fact_rows(
                    [
                        ("input size used", _input_size_text(info, settings)),
                        ("norm mean", _mean_std(info.norm_mean)),
                        ("norm std", _mean_std(info.norm_std)),
                        ("device", str(device)),
                        (
                            "mode",
                            "eval()" if not model.training else "TRAINING",
                        ),
                        ("output classes", str(len(info.class_names))),
                    ]
                )
            )
            st.caption(
                "The mean and std are the ones the checkpoint says it was trained "
                "with. `src/inference.check_preprocessing` compares them against "
                "the constants in `src/data/transforms.py` and logs a warning if "
                "they disagree -- it does not substitute them, because applying "
                "the wrong ones produces a confident wrong answer rather than an "
                "error."
            )

        st.markdown("**Class names, in the order the model outputs them**")
        st.code("  ·  ".join(info.class_names), language="text")
        st.caption(
            "The order is the model's output order and it comes from the "
            "checkpoint, never from the folder names on disk. A list shifted by "
            "one position turns every prediction into a different, equally "
            "confident, wrong label."
        )
        st.caption(
            f"To score this checkpoint honestly: `python -m src.evaluate "
            f"--checkpoint {info.path}`"
        )


# ---------------------------------------------------------------------------
# Prediction history, stored
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoryEntry:
    """One classified scan, as kept in `st.session_state`.

    The numbers a `Prediction` produced, plus the two things that make a row
    checkable later -- which file, and which checkpoint. No image bytes: see the
    module docstring for why the list that grows without bound is also the one
    that must stay small.

    `probabilities` is the one mutable field, and it is the same trade
    `Prediction` makes. It is copied out of the frozen prediction rather than
    referenced, so a later prediction cannot reach back and edit a recorded row.

    Fields, and what each is for on screen:
        at:             ISO timestamp, truncated to seconds. Rendered as a clock
                        time -- a session cannot outlive the day it started, so a
                        date column would be a constant column.
        name:           the uploaded filename, as a label. Never used as a path.
        label:          the predicted class, in the checkpoint's own spelling.
        confidence:     softmax probability of `label`. See the docstring on what
                        that is and is not.
        margin:         confidence minus the runner-up, which is the number that
                        separates a clear call from an ambiguous one.
        near_tie:       `margin < TIE_MARGIN`, from the same rule that raises the
                        warning under the result.
        image_size:     the side length actually used, for the same provenance
                        reason the result panel repeats it.
        checkpoint:     the run that produced it, so a table spanning a change of
                        model is not read as one model's output.
        probabilities:  the whole distribution, kept so the CSV export carries
                        what `src/inference.format_report` prints.
        fingerprint:    identity of "these bytes, this checkpoint", for the
                        duplicate check in `record_prediction`.
    """

    at: str
    name: str
    label: str
    confidence: float
    margin: float
    near_tie: bool
    image_size: int
    checkpoint: str
    probabilities: dict[str, float]
    fingerprint: str


@dataclass(frozen=True)
class HistoryState:
    """What one session has classified, and what it is not showing.

    `total` counts everything ever recorded in the session; `entries` is the tail
    of that, capped at `HISTORY_LIMIT`. Keeping them apart is what lets the table
    say how many older rows it dropped instead of quietly looking like the whole
    session.
    """

    entries: list[HistoryEntry]
    total: int

    @property
    def truncated(self) -> bool:
        """True when older entries exist that are not in `entries`."""
        return self.total > len(self.entries)

    @property
    def near_ties(self) -> int:
        """How many of the shown rows were ambiguous.

        A count of decisions, not an average of beliefs: `src/inference.py`
        refuses to average confidences across images, and "4 of these 25 were
        near-ties" is a statement about the model being undecided rather than a
        number that invites a reader to compute a mean from it.
        """
        return sum(1 for entry in self.entries if entry.near_tie)

    @property
    def distinct_labels(self) -> list[str]:
        """Classes predicted in the shown rows, most recent first.

        Order is first-seen rather than alphabetical so the list reads as "what
        has this session been calling things" in the order it happened.
        """
        seen: dict[str, None] = {}
        for entry in self.entries:
            seen.setdefault(entry.label, None)
        return list(seen)


def _fingerprint(data: bytes, checkpoint: Path) -> str:
    """A short hash identifying "these upload bytes, classified by this checkpoint".

    blake2b rather than md5 or sha1: this is an identity check, not a security
    control, and blake2b is the fastest of the three at this size. The checkpoint
    path is folded in so the same scan classified by a different run is a
    different row, which is what actually happened.
    """
    digest = hashlib.blake2b(digest_size=12)
    digest.update(data)
    digest.update(str(checkpoint).encode("utf-8"))
    return digest.hexdigest()


def read_history() -> HistoryState:
    """Whatever this session has recorded so far.

    Read out of `st.session_state` on every call rather than cached in a module
    global: a module global is shared by every browser session the same process
    serves, so one reader's scans would appear in another reader's history. The
    dataclass round-trip through `asdict` is what keeps `HistoryEntry` a real
    dataclass here instead of a dict with attribute access by hand.
    """
    stored = st.session_state.get(HISTORY_KEY, [])
    entries = [HistoryEntry(**row) for row in stored]
    total = int(st.session_state.get(HISTORY_TOTAL_KEY, len(entries)))
    return HistoryState(entries=entries, total=max(total, len(entries)))


def record_prediction(
    prediction: Prediction,
    checkpoint: Path,
    name: str,
    data: bytes,
) -> bool:
    """Add one prediction to the session history, newest first.

    Returns whether it was added, which is False when this exact result was
    already recorded.

    The duplicate check is the load-bearing part of this function. Streamlit
    re-executes the whole script on every interaction, so the result panel is
    recomputed each time a sidebar control moves, a clear button is pressed, or
    the history table is sorted -- and without a check every one of those
    appends another copy of whatever is on screen. Comparing the *content* of the
    upload and the checkpoint, rather than the filename, is what makes a
    second scan that happens to share a name and a byte length with the first one
    still get recorded: dropping it silently would be worse than a duplicate.

    Re-uploading the same file against the same model is a no-op rather than an
    error, because the answer is deterministic -- `predict_tensor` runs the model
    in `eval()` mode, and dropout is off.
    """
    fingerprint = _fingerprint(data, checkpoint)
    state = read_history()
    if any(entry.fingerprint == fingerprint for entry in state.entries):
        logger.debug("Already in history: %s", name)
        return False

    entry = HistoryEntry(
        at=datetime.now().isoformat(timespec="seconds"),
        name=name,
        label=prediction.label,
        confidence=prediction.confidence,
        margin=prediction.margin,
        near_tie=_is_tie(prediction),
        image_size=prediction.image_size,
        checkpoint=str(checkpoint),
        probabilities=dict(prediction.probabilities),
        fingerprint=fingerprint,
    )
    # Newest first, so the most recent result is the one a reader sees without
    # scrolling -- which is also the order the CSV is reversed back into.
    kept = [entry, *state.entries][:HISTORY_LIMIT]
    st.session_state[HISTORY_KEY] = [asdict(item) for item in kept]
    st.session_state[HISTORY_TOTAL_KEY] = state.total + 1
    logger.info(
        "Recorded %s -> %s at %.1f%% (%s)",
        name,
        prediction.label,
        prediction.confidence * 100,
        checkpoint,
    )
    return True


def clear_history() -> None:
    """Forget this session's predictions. Wired to both clear buttons.

    A callback rather than an `if` in the body, because the button's return value
    is only True on the run that pressed it, and the history is drawn *before* the
    reader reaches the buttons that would change it.
    """
    st.session_state.pop(HISTORY_KEY, None)
    st.session_state.pop(HISTORY_TOTAL_KEY, None)
    logger.info("Prediction history cleared")


# ---------------------------------------------------------------------------
# Prediction history, rendered
# ---------------------------------------------------------------------------


def _clock(at: str) -> str:
    """The time of day from an ISO timestamp, for the history table.

    Falls back to the raw string rather than raising: a row whose timestamp could
    not be parsed is a row with a bad time on it, which is a cosmetic problem, and
    a table that refuses to draw because of one is a worse one.
    """
    try:
        return datetime.fromisoformat(at).strftime("%H:%M:%S")
    except ValueError:
        return at


def _next_candidate(entry: HistoryEntry) -> str:
    """The runner-up as "Class 3.1%", or an em dash when there is only one class.

    Recomputed from the stored distribution rather than kept as its own field:
    `margin` already fixes the value, and a second copy of the same number is a
    second thing that can disagree with the first.
    """
    others = {
        name: probability
        for name, probability in entry.probabilities.items()
        if name != entry.label
    }
    if not others:
        return "—"
    name, probability = max(others.items(), key=lambda item: item[1])
    return f"{display_label(name)}  {probability:.1%}"


def history_rows(entries: Sequence[HistoryEntry]) -> list[dict[str, object]]:
    """The history as display rows, newest first.

    Percentages are given to `st.dataframe` as numbers in [0, 100] with a
    `column_config` format, not as pre-formatted strings, so the columns stay
    sortable and right-aligned. A history a reader cannot sort is much less use
    than a plain table: sorting by confidence is how a run of ambiguous scans
    becomes visible.
    """
    return [
        {
            "Time": _clock(entry.at),
            "Scan": entry.name,
            "Predicted class": display_label(entry.label),
            "Confidence": entry.confidence * 100,
            "Margin": entry.margin * 100,
            "Next candidate": _next_candidate(entry),
            "Model": entry.checkpoint,
            "Flag": "near-tie" if entry.near_tie else "",
        }
        for entry in entries
    ]


def history_csv(entries: Sequence[HistoryEntry]) -> str:
    """The shown history as CSV: one row per scan, oldest first.

    Chronological, the reverse of the table, because a CSV is a log rather than a
    panel: rows in the order they happened append cleanly to a file that already
    has rows, and sorting either way in a spreadsheet is one click.

    The columns are the prediction plus its provenance -- the image, the
    checkpoint, the input size, and one column per class holding the whole
    distribution. A CSV of class names alone would be exactly the artifact this
    project argues against: a number that cannot be traced back to the model and
    the image that produced it. Only the rows still in `entries` are exported, so
    the file and the table always agree, and the caption above the button says so
    when the table is truncated.
    """
    class_names: list[str] = []
    for entry in entries:
        for name in entry.probabilities:
            if name not in class_names:
                class_names.append(name)

    header = [
        "time",
        "image",
        "label",
        "confidence",
        "margin",
        "near_tie",
        "image_size",
        "checkpoint",
        *(f"prob_{name}" for name in class_names),
    ]

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    for entry in reversed(entries):
        writer.writerow(
            [
                entry.at,
                entry.name,
                entry.label,
                f"{entry.confidence:.6f}",
                f"{entry.margin:.6f}",
                "true" if entry.near_tie else "false",
                entry.image_size,
                entry.checkpoint,
                *(f"{entry.probabilities.get(name, 0.0):.6f}" for name in class_names),
            ]
        )
    return buffer.getvalue()


def render_history() -> None:
    """The session's predictions, newest first, with the caveats attached.

    A bordered card, a three-number summary and a table, in that order: the
    numbers answer "is this session going normally" and the table answers "which
    scan", and the caveat about scope sits under both because it applies to both.
    """
    state = read_history()

    with st.container(border=True, gap="medium"):
        _section(
            "Prediction history",
            "Every scan classified in this browser session, newest first.",
        )

        if not state.entries:
            st.markdown(
                "Nothing yet. Each upload is classified automatically and appears "
                "here, so the same scan can be compared against the last few "
                "without re-running the model."
            )
            st.caption(SCOPE_NOTE)
            return

        # `vertical_alignment="bottom"` so the clear button sits on the same line
        # as the bottom of the metric cards rather than floating at the top of a
        # taller cell.
        top = st.columns([1, 1, 1, 1], gap="medium", vertical_alignment="bottom")
        with top[0]:
            st.metric("Scans this session", str(state.total), border=True)
        with top[1]:
            st.metric(
                "Near-ties",
                str(state.near_ties),
                border=True,
                help=f"Rows whose top two classes were within {TIE_MARGIN:.0%} of "
                "each other -- the ambiguous results, marked in the Flag column.",
            )
        with top[2]:
            st.metric(
                "Classes predicted",
                str(len(state.distinct_labels)),
                border=True,
                help=", ".join(display_label(n) for n in state.distinct_labels),
            )
        with top[3]:
            # Same callback as the sidebar's button, different key. Both are here
            # because the reader who wants to clear a list is looking at the list.
            st.button(
                "Clear history",
                on_click=clear_history,
                key=CLEAR_KEY_BODY,
                width="stretch",
            )

        st.dataframe(
            history_rows(state.entries),
            width="stretch",
            height=_table_height(len(state.entries)),
            hide_index=True,
            column_order=[
                "Time",
                "Scan",
                "Predicted class",
                "Confidence",
                "Margin",
                "Next candidate",
                "Model",
                "Flag",
            ],
            column_config={
                "Time": st.column_config.TextColumn(
                    "Time", width="small", help="Local time the scan was classified."
                ),
                "Scan": st.column_config.TextColumn(
                    "Scan",
                    width="medium",
                    help="The uploaded filename, as a label. Never used as a path.",
                ),
                "Predicted class": st.column_config.TextColumn(
                    "Predicted class",
                    help="In the checkpoint's own spelling, title-cased for "
                    "display by src/evaluate.display_label.",
                ),
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    format="%.1f%%",
                    min_value=0.0,
                    max_value=100.0,
                    help="The model's softmax score, not a calibrated probability. "
                    "Sort by this column to find the scans it was least sure of.",
                ),
                "Margin": st.column_config.NumberColumn(
                    "Margin",
                    format="%.1f%%",
                    help="Confidence minus the runner-up. Below "
                    f"{TIE_MARGIN:.0%} the row is flagged as a near-tie.",
                ),
                "Next candidate": st.column_config.TextColumn(
                    "Next candidate",
                    help="The second-highest class and its probability -- the "
                    "number that says whether a confident answer was easy.",
                ),
                "Model": st.column_config.TextColumn(
                    "Model",
                    width="medium",
                    help="The checkpoint that produced the row, so a table "
                    "spanning a change of model is not read as one model's output.",
                ),
                "Flag": st.column_config.TextColumn(
                    "Flag",
                    width="small",
                    help="`near-tie` when the top two classes were within "
                    f"{TIE_MARGIN:.0%}.",
                ),
            },
        )

        if state.truncated:
            st.caption(
                f"Showing the last {len(state.entries)} of {state.total} scans "
                "classified in this session. Older rows are dropped to keep the "
                "page responsive, and the CSV below exports only what is shown."
            )

        st.download_button(
            "Download history as CSV",
            data=history_csv(state.entries),
            file_name=f"predictions_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
            help="The rows above, oldest first, with the full per-class "
            "distribution and the checkpoint each came from.",
        )
        st.caption(SCOPE_NOTE)


def _table_height(rows: int) -> int:
    """A dataframe tall enough for its rows, capped so it scrolls.

    Streamlit's own `height="auto"` sizes to the content, which for 25 rows is
    taller than most screens -- and the page below it is the part with the
    download button. A capped height turns a long history into a scroll region
    inside the card instead of a page the reader has to scroll back up.
    """
    return min(420, 38 * (rows + 1) + 42)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def classify(
    upload: Mapping[str, Any],
    model: nn.Module,
    info: CheckpointInfo,
    device: torch.device,
    image_size: int | None,
) -> Prediction:
    """Classify one upload. Thin wrapper over the library, with the model in hand.

    Takes an already-loaded model rather than loading one, which is what lets the
    page load the checkpoint once, describe itself with it, and then classify --
    and what lets a test drive the whole upload -> predict path without a
    Streamlit runtime or a second read of a 45MB file.

    Args:
        upload: a mapping with `data` (the bytes) and `name` (the filename). A
            dict rather than an `UploadedFile` so the caller -- and a test -- does
            not need a browser to hand one over.
        model: from `_load_model`.
        info: the same checkpoint's metadata.
        device: where the forward pass runs.
        image_size: the sidebar's override, or None to use the checkpoint's.

    Returns:
        The `Prediction`.
    """
    size = resolve_image_size(info, image_size)
    with tempfile.TemporaryDirectory() as directory:
        path = save_upload(upload["data"], upload["name"], Path(directory))
        return predict_image(path, model, info, device, image_size=size)


def render_result(
    uploaded: Any,
    model: nn.Module,
    info: CheckpointInfo,
    device: torch.device,
    settings: Settings,
) -> None:
    """Classify the upload and draw it beside the scan.

    The bytes are read once: `getvalue()` is not free, and three separate readers
    of it (the size check, the preview, and the write) would be three copies.

    The prediction is computed *before* the two columns are created, rather than
    inside them. The previous version opened the columns first and then called
    `.empty()` on the left one when inference failed, which left an error message
    in a column that had been laid out for an image; here a failure is a message
    in the flow of the page and no half-drawn layout is left behind.

    The history is recorded from the same `Prediction` that the panel below draws,
    immediately before the panel is built. Recording it after the render would
    work just as well and was the previous order, but keeping the two adjacent
    means there is one place where a result becomes a row, so the table and the
    result on screen cannot come from two different calls.
    """
    data = uploaded.getvalue()
    name = uploaded.name

    problem = upload_problem(data)
    if problem is not None:
        st.error(problem)
        return

    with st.spinner("Classifying..."):
        try:
            prediction = classify(
                {"data": data, "name": name},
                model,
                info,
                device,
                settings.image_size,
            )
        except (FileNotFoundError, ValueError) as error:
            # The library's messages name the fix ("pass an image size", "this
            # file could not be decoded"), so they are shown rather than replaced
            # with something vaguer.
            st.error(str(error))
            return
        except Exception as error:  # noqa: BLE001 - see the comment below
            # Broad for the same reason as in `main`, and it costs the same when it
            # is wrong: which exceptions a load of someone else's image can raise
            # is not knowable in advance, and the cases that are easy to forget are
            # not rare ones -- PIL raises `DecompressionBombError` for a crafted
            # header, and that is not an `OSError`. The traceback would land on top
            # of a page whose model information and history are still perfectly
            # good, hiding them behind a stack trace about a file the reader only
            # just chose. It goes to the terminal instead.
            logger.exception("Could not classify %s", name)
            st.error(f"The scan could not be classified: {error}")
            return

    record_prediction(prediction, settings.checkpoint, name, data)

    scan_column, prediction_column = st.columns([1, 1], gap="large")
    with scan_column:
        render_scan(data, name)
    with prediction_column:
        render_prediction(prediction, info)


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

    _header()

    # Loaded before the uploader is even drawn. See the docstring: the page
    # describes what it is about to use rather than waiting to be asked.
    #
    # The `except` is deliberately `Exception` rather than a list of the types a
    # bad checkpoint is known to raise. That list is a guess, and a guess here is
    # expensive: an unlisted type escapes as a red traceback that replaces the
    # whole page, which is the one outcome `render_load_error` exists to prevent.
    # (A corrupt `best.pt` raises `_pickle.UnpicklingError`, which is neither
    # `OSError` nor `RuntimeError`, so the narrower version of this handler was
    # not hypothetical.) The traceback still reaches the terminal, so nothing is
    # hidden from whoever is developing the app; only the page gets the short
    # version. `KeyboardInterrupt` and `SystemExit` are not `Exception` and still
    # stop the server, as they should.
    try:
        with st.spinner("Loading the model..."):
            model, info, device = _load_model(str(settings.checkpoint), settings.device)
    except Exception as error:
        logger.exception("Could not load the checkpoint %s", settings.checkpoint)
        render_load_error(error)
        return

    # The rest of the sidebar, now that there is something to describe.
    render_sidebar_model(model, info, device, settings)
    render_sidebar_session()

    uploaded = render_uploader()
    if uploaded is None:
        render_waiting()
    else:
        render_result(uploaded, model, info, device, settings)

    render_model_info(model, info, device, settings)
    render_history()


if __name__ == "__main__":
    main()
