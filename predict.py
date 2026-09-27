"""
predict.py -- classify a single MRI scan from the command line.

    python predict.py data/processed/test/glioma/glioma (1).jpg
    python predict.py scan.png --checkpoint models/run1/best.pt
    python predict.py scan.png --device cpu --image-size 224
    python predict.py scan.png --brief

INPUT
    One image file. JPEG, PNG, BMP, TIFF, or WebP -- whatever
    `src/data/dataset.py` recognises as an image.

OUTPUT
    The predicted class and a confidence percentage, printed as a plain report,
    followed by the full per-class distribution:

        Prediction
        ==========

          image        scan.jpg
          checkpoint   models\\run1\\best.pt

          Predicted class   Glioma
          Confidence        94.31%

          Next candidate    No Tumor at 3.11%  (margin 91.20%)

        Class probabilities
        --------------------
        * Glioma          94.31%  ######################
          No Tumor         3.11%  #
          Meningioma       2.44%
          Pituitary        0.14%

          * predicted class.  Image resized to 224x224.

        Research output only -- not for diagnosis or treatment decisions.
        Confidence is the model's softmax score, not a calibrated probability of
        a correct diagnosis.

    Add --json to get the same numbers as a machine-readable object instead, for
    scripting or piping into something else, and --brief to drop the distribution
    and keep the two headline lines.

    The architecture and epoch are on stderr rather than in the report, in the
    "Loaded models/run1/best.pt (arch=resnet18, epoch=17)" log line -- the
    checkpoint path already names the run, so repeating it in the header would
    add a line to every report and identify nothing new. Stdout is kept to the
    report alone so it can be redirected without dragging log lines along.

WHY THIS FILE IS ALMOST EMPTY
----------------------------
It parses arguments, prints a report, and nothing else. All of the model work is
in `src/inference.py`, which `app.py` also uses. The project rule is that the UI
must never contain model logic, and a script that a person reads first is
exactly where that rule is most likely to be quietly broken -- one hand-edited
preprocessing line here and the CLI starts disagreeing with the Streamlit app
about the same image, with nothing to catch it.

That also means the CLI cannot drift from the evaluation numbers. It calls the
same checkpoint loader and the same eval transform that `src/evaluate.py` scores
with, so a prediction here is the same number that appears in a confusion matrix
there, and any change to the pipeline shows up in both at once.

EXIT CODES
----------
    0   the image was classified
    1   the image could not be classified (bad path, unreadable file, no
        checkpoint, unknown preprocessing)

`--fail-under` additionally exits 1 when the confidence is below the given
percentage, which is what makes this usable as a gate in a script:

    python predict.py scan.jpg --fail-under 80 || echo "needs review"

WARNINGS, NOT ERRORS
--------------------
A low confidence is *not* a failure. The model is allowed to say "I am not sure",
and that is a real answer, not an error condition -- so by default a 12%-confidence
prediction still exits 0 and still prints. Only `--fail-under` turns uncertainty
into a non-zero exit, and it has to be asked for.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.inference import format_report, predict_file

#: Diagnostics go to stderr through the logger, so stdout carries the report and
#: nothing else. That is what lets `python predict.py scan.jpg --json > out.json`
#: produce a file that is valid JSON rather than one with a report wrapped round
#: it. The logger format matches src/train.py and src/evaluate.py.
LOG_FORMAT = "%(asctime)s  %(levelname)-7s %(name)s: %(message)s"
LOG_DATEFMT = "%H:%M:%S"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    The image is positional and required. Making it a flag (`--image`) would
    allow it to be omitted and then fail later, in torch, on a tensor the user
    never named; argparse can require it up front and say what is missing.
    """
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description="Classify a single brain MRI scan: prints the predicted "
        "class and a confidence percentage.",
        epilog="Not a medical device. Research/educational use only.",
    )

    parser.add_argument(
        "image",
        type=Path,
        help="path to one MRI image (jpg, jpeg, png, bmp, tif, tiff, webp)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        metavar="PATH",
        help="a .pt from src/train.py; defaults to the newest models/*/best.pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        metavar="DEV",
        help="'auto' (GPU when present), 'cpu', 'cuda', 'cuda:0', 'mps' "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        metavar="N",
        help="override the checkpoint's recorded image size. Only needed for a "
        "checkpoint that does not record one",
    )
    parser.add_argument(
        "--brief",
        action="store_true",
        help="print only the predicted class and the confidence, without the "
        "full per-class distribution",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="print the result as JSON instead of the text report",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        metavar="PCT",
        help="exit 1 if the confidence is below this percentage",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="silence the log output on stderr, warnings included. It hides the "
        "preprocessing-mismatch warning from src/inference.py as well as the "
        "progress lines, so do not use it when a checkpoint's preprocessing is "
        "in doubt",
    )

    args = parser.parse_args(argv)

    if args.fail_under is not None and not 0.0 <= args.fail_under <= 100.0:
        # Checked here rather than being left to the comparison below, where
        # 120 would silently never trigger and 101 would be indistinguishable
        # from 100.
        parser.error(f"--fail-under must be between 0 and 100, got {args.fail_under}")

    return args


def main(argv: list[str] | None = None) -> int:
    """Run one prediction. Returns the process exit code."""
    args = parse_args(argv)

    logging.basicConfig(
        # ERROR, not WARNING, when quiet: the shared resolve_device() in
        # src/train.py logs its "no GPU detected" fallback at WARNING, so a
        # WARNING-level quiet would still leave that line on stderr.
        level=logging.ERROR if args.quiet else logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
    )

    if not args.image.exists():
        # Checked before the model loads, so a typo'd path costs a millisecond
        # rather than a checkpoint read, and the error names the actual problem.
        print(f"error: no such image: {args.image}", file=sys.stderr)
        return 1
    if not args.image.is_file():
        # Distinguished from "missing" because the two have different fixes: a
        # directory is a wrong argument (this CLI takes one file), not a typo.
        print(
            f"error: {args.image} is not a file. This CLI takes a single image; "
            "point it at one file, or use src.inference.predict_files() for a "
            "list of paths.",
            file=sys.stderr,
        )
        return 1

    try:
        prediction = predict_file(
            args.image,
            checkpoint=args.checkpoint,
            device=args.device,
            image_size=args.image_size,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        # These three cover every failure this path can produce on purpose: a
        # missing file, an undecodable image or unknown preprocessing, and a
        # state dict that will not fit the architecture. Each already carries a
        # message written to be read on its own, so it is printed as-is rather
        # than being re-wrapped in a traceback that hides the useful sentence.
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.as_json:
        payload = prediction.to_dict()
        payload["image"] = str(args.image)
        payload["checkpoint"] = str(args.checkpoint) if args.checkpoint else None
        print(json.dumps(payload, indent=2))
    else:
        print(
            format_report(
                prediction,
                image=args.image,
                checkpoint=args.checkpoint,
                show_all=not args.brief,
            )
        )

    if args.fail_under is not None:
        threshold = args.fail_under / 100.0
        if prediction.confidence < threshold:
            print(
                f"error: confidence {prediction.confidence:.2%} is below the "
                f"{args.fail_under:.2f}% threshold",
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == "__main__":
    # A guarded main that returns an int, so the exit code is explicit rather
    # than an exception traceback doubling as a failure signal.
    sys.exit(main())
