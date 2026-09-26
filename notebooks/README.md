# `notebooks/`

Exploration and research notebooks. **Not used in production** — anything needed at
training or inference time belongs in `src/` so it can be imported and tested.

## Naming

Use a numbered, ordered prefix so the intended execution order is obvious:

```
01_data_exploration.ipynb     # class balance, intensity distributions, slice sanity checks
02_baseline_model.ipynb      # simple transfer-learning CNN, first end-to-end number
03_augmentation_ablation.ipynb
04_error_analysis.ipynb      # inspect false positives / false negatives
05_inference_demo.ipynb      # run a single scan through a trained checkpoint
```

## Conventions

- Clear all outputs before committing (`jupyter nbconvert --clear-output --inplace`).
- Keep secrets out of notebooks; load config from `src/` or environment variables.
- Pin the kernel: the project virtualenv, not the system Python.
- Record the random seed and library versions in the first markdown cell.
