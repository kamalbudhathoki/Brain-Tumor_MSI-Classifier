# Brain Tumor MRI Classifier

Production-oriented project scaffold for classifying brain tumor MRI scans with PyTorch.

> **Status: scaffold only — no model, training, or inference logic is implemented yet.**
> This repository currently defines the structure, conventions, and contracts that the
> implementation will follow.

---

## Scope

Classify a brain MRI scan into one of the standard classes (e.g. `glioma`,
`meningioma`, `pituitary`, `no_tumor`) and surface the prediction with calibrated
confidence and a visual explanation.

> ⚠️ **Not a medical device.** This is a research/educational project. No output of
> this software may be used for diagnosis or treatment decisions.

---

## Project structure

```
brain-tumor-classifier/
├── data/           # Raw + processed images and split manifests. Git-ignored (large,
│                   #   license-restricted). Every folder has its own README.
├── notebooks/      # EDA, baselines, ablations. Exploration only — never imported
│                   #   by app code.
├── src/            # All importable production code: config, dataset, transforms,
│                   #   model factory, train, evaluate, inference.
├── models/         # Trained checkpoints + registry.json. Git-ignored (binaries).
│                   #   Each checkpoint is self-describing.
├── outputs/        # Derived run artifacts: logs, metrics, curves, figures,
│                   #   TensorBoard events. Git-ignored (regenerable).
├── app.py          # Application entry point (inference UI). Streamlit.
├── predict.py      # Classify one scan from the CLI: predicted class + confidence
├── requirements.txt
├── README.md
└── .gitignore
```

The split is deliberate: **`data/` and `models/` are inputs, `src/` is logic, `outputs/`
is disposable.** Nothing in `outputs/` is ever read at inference time, and no logic
lives in `notebooks/`.

`app.py` and `predict.py` are both entry points over `src/inference.py`, and
neither contains model logic:

```powershell
streamlit run app.py                              # browser UI
python predict.py path\to\scan.jpg                # one scan, terminal
python predict.py path\to\scan.jpg --json         # machine-readable
```

---

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

For CPU-only machines, install torch from the CPU wheel index first (see the comment at
the top of `requirements.txt`).

---

## Planned implementation order

1. **Config** — `src/config.py`: dataclass config + YAML, one source of truth for
   paths, image size, normalization stats, and hyperparameters.
2. **Data** — download, build patient-level splits, write `manifest.csv`, implement
   `Dataset` + train/val transform pipelines.
3. **Model** — `build_model()` factory starting from an ImageNet-pretrained backbone
   (ResNet-50 / EfficientNet), with the classifier head sized to `num_classes`.
4. **Training** — AMP mixed precision, cosine LR schedule, early stopping, best/last
   checkpointing, TensorBoard logging, deterministic seeding.
5. **Evaluation** — accuracy, macro precision/recall/F1, per-class ROC-AUC, confusion
   matrix, and error analysis on the held-out test set.
6. **Inference** — `src/inference.py` returns label + probabilities; single scan is
   done and shared by `predict.py` and `app.py`. Still to do: Grad-CAM heatmaps for
   explainability, and a batch mode.
7. **Application** — `app.py` is done: upload one scan, get the predicted class,
   confidence, and the full class distribution, with the checkpoint's own
   preprocessing applied automatically.
8. **Hardening** — tests, ONNX export, containerization, model card.

---

## Engineering conventions

| Concern | Rule |
| --- | --- |
| Data leakage | Split by **patient**, never by slice. |
| Reproducibility | Seed Python/NumPy/PyTorch; save `config.yaml` + seed with every run. |
| Checkpoints | Self-describing: arch, class names, image size, normalization, epoch, metrics. |
| Preprocessing | Always read it from the checkpoint — never hardcode it in the UI. |
| Logging | Logger in `src/`, `print` only in notebooks. |
| Paths | Relative to project root; no absolute or machine-specific paths. |
| Artifacts | Git-ignore data, weights, and outputs; commit code, configs, and cards. |

---

## Datasets

See `data/README.md` for the expected layout and the public sources this project targets
(BraTS 2020/2021, Kaggle "Brain Tumor MRI Dataset"). Record license and provenance in
`data/dataset_card.md` before use.

---

## License

Add a license before publishing (e.g. MIT for code, plus the upstream dataset license).
