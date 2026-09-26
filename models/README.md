# `models/`

Trained model artifacts. **Not tracked by git** (weights are large binaries) except this
README and the small `registry.json` manifest.

## Layout

```
models/
├── registry.json              # tracked: metadata for every checkpoint on disk
├── resnet50_brain_tumor/
│   ├── best.pt                # best validation checkpoint (weights + optimizer state)
│   ├── last.pt                # final epoch, used to resume interrupted runs
│   ├── config.yaml            # exact hyperparameter set used for this run
│   └── metrics.json           # train/val curves, best epoch, final metrics
```

## Checkpoint contents

A `.pt` file should hold more than raw weights:

```python
{
    "state_dict": ...,
    "arch": "resnet50",
    "num_classes": 4,
    "class_names": ["glioma", "meningioma", "pituitary", "no_tumor"],
    "image_size": 224,
    "norm_mean": [...], "norm_std": [...],
    "epoch": 17, "val_acc": 0.973, "seed": 42,
}
```

This makes a checkpoint self-describing: inference never has to guess the preprocessing.

## Rules

- A run is reproducible only if its `config.yaml` and `registry.json` entry are committed.
- Never overwrite `best.pt`; use a new directory per experiment.
- Store the SHA-256 of each `.pt` in `registry.json` to detect corrupted or swapped files.
