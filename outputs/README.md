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
│   ├── curves.png             # loss / accuracy curves
│   └── confusion_matrix.png
├── predictions/               # per-image prediction CSVs for later analysis
├── figures/                   # EDA plots, Grad-CAM heatmaps, paper-quality charts
└── tensorboard/               # `tensorboard --logdir outputs/tensorboard`
```

## Rules

- Every directory name is the experiment name plus a short git SHA, e.g.
  `resnet50_a7f3c1d/`, so parallel experiments never overwrite each other.
- Never read anything from `outputs/` at inference time — it is not an input surface.
