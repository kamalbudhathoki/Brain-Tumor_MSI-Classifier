# `src/`

All importable production code. This is the only directory the app and the training
entry points import from.

## Modules

```
src/
├── __init__.py
├── data/
│   ├── dataset.py   # torch Dataset: file list -> augmented tensor
│   └── transforms.py# train/val transform pipelines (resize, normalize, flips)
├── models/
│   └── classifier.py# model factory (build_model(name, num_classes))
├── train.py         # training loop, checkpointing, early stopping
├── evaluate.py      # metrics: accuracy, precision/recall/F1, ROC-AUC, confusion
│                    #   matrix, and the count/recall/precision error-analysis figure
│                    #   plus checkpoint loading/discovery (list_checkpoints,
│                    #   find_checkpoint) -- the "registry" role
├── inference.py     # single-scan prediction, returns probabilities + label
├── plots.py         # the confusion-matrix / error-analysis figure code
└── data, models     # __init__.py re-exports the public names
```

`config.py` is not a separate module: the run configuration lives in `train.py` as
`TrainConfig` and in `evaluate.py` as `EvalConfig`, each a dataclass whose defaults
are declared once and validated up front, so a run and its evaluation are described
by the same objects that consume them.

`inference.py` is implemented. It is the only place model logic lives at prediction
time, and both entry points over it — `../predict.py` (CLI) and `../app.py` (UI) —
call it rather than reimplementing any of it.

Import direction is one-way: `inference` → `evaluate` → `train` → `data`/`models`.
Anything needed by more than one layer is defined at the top of that chain and
imported from above, never reimplemented below it — `display_label` is the worked
example, and it lives in `evaluate` because the confusion figure needs it too.

## Rules

- No `print` in library code — use the logger.
- No absolute paths; all paths resolve relative to the project root.
- Type hints and docstrings on public functions.
- Keep notebooks and `src/` in sync: a fix discovered in a notebook gets ported here.
