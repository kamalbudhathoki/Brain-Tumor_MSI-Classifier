# `src/`

All importable production code. This is the only directory the app and the training
entry points import from.

## Planned modules

```
src/
├── __init__.py
├── config.py        # dataclasses + YAML loading for every hyperparameter
├── data/
│   ├── dataset.py   # torch Dataset: file list -> augmented tensor
│   └── transforms.py# train/val transform pipelines (resize, normalize, flips)
├── models/
│   └── classifier.py# model factory (build_model(name, num_classes))
├── train.py         # training loop, checkpointing, early stopping
├── evaluate.py      # metrics: accuracy, precision/recall/F1, ROC-AUC, confusion matrix
├── inference.py     # single-scan and batch prediction, returns probabilities + label
├── registry.py      # model/checkpoint bookkeeping (what exists, what is best)
└── utils/
    ├── seed.py      # deterministic seeding across python/numpy/torch
    └── logging.py   # structured run logging
```

## Rules

- No `print` in library code — use the logger.
- No absolute paths; all paths resolve relative to the project root.
- Type hints and docstrings on public functions.
- Keep notebooks and `src/` in sync: a fix discovered in a notebook gets ported here.
