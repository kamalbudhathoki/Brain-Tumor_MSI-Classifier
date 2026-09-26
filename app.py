"""
Brain Tumor MRI Classifier — application entry point.

STATUS: scaffold only. The model, preprocessing, and prediction logic are
deliberately NOT implemented yet. This file defines the contract the rest of the
project will fill in, so the interface is settled before the model exists.

Planned responsibilities:
  1. Upload a single MRI scan (or a batch) from the browser.
  2. Run the exact preprocessing the checkpoint was trained with.
  3. Return the predicted tumor class, per-class probabilities, and
     Grad-CAM heatmaps for the non-technical user to review.

Framework note: this scaffold targets Streamlit (fastest path to a working
inference UI for image models). If a REST API is needed instead, swap the
Streamlit layer for FastAPI + uvicorn and keep src/inference.py untouched —
the UI must never contain model logic.
"""

from __future__ import annotations


def main() -> None:
    """Run the inference application."""
    raise NotImplementedError(
        "Model not implemented yet. Planned steps: "
        "1) src/inference.py (preprocessing + predict), "
        "2) this UI layer, "
        "3) wire in a checkpoint from models/ via models/registry.json."
    )


if __name__ == "__main__":
    main()
