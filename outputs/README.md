# `outputs/`

Everything a run produces that is *derived* from data and code. **Not tracked by git** —
it can always be regenerated. Only curated figures in the README are committed.

## Layout

```
outputs/
├── runs/<experiment_name>/
│   ├── config.yaml            # resolved config for the run
│   ├── train.log              # stdout + logger output
│   ├── metrics.csv            # per-epoch train/val loss and metrics
│   ├── metrics.json           # best + final metrics summary
│   ├── train_loss.png         # training loss per epoch
│   ├── val_loss.png           # validation loss per epoch (best epoch marked)
│   ├── train_accuracy.png     # training accuracy per epoch
│   ├── val_accuracy.png       # validation accuracy per epoch (best epoch marked)
│   └── confusion_matrix.png
├── eval/                      # `python -m src.evaluate` writes here
│   ├── test_report.json       # every metric, machine-readable
│   ├── confusion_matrix.png   # counts, or per-class recall with
│   │                          #   --normalize-confusion-matrix
│   └── error_analysis.png     # counts + recall + precision side by side, with
│                              #   the wrong predictions called out
├── predictions/               # per-image prediction CSVs for later analysis
├── figures/                   # EDA plots, Grad-CAM heatmaps, paper-quality charts
└── tensorboard/               # `tensorboard --logdir outputs/tensorboard`
```

## Rules

- Every directory name is the experiment name plus a short git SHA, e.g.
  `resnet50_a7f3c1d/`, so parallel experiments never overwrite each other.
- Never read anything from `outputs/` at inference time — it is not an input surface.
- Everything here regenerates. `src/train.py` writes the four curves at the end of
  every run; `python -m src.plots --metrics models/<run>/metrics.json` rebuilds them
  for an older run without retraining.
- The figures in `eval/` come from one `python -m src.evaluate` run, and their titles
  carry the run name and split (`resnet50_a7f3c1d · split=test`) so a PNG stays
  identifiable after it has been copied out of `outputs/`.
