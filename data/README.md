# `data/`

Raw and processed image data. **Not tracked by git** (see the root `.gitignore`) because
medical imaging datasets are large and often license-restricted.

## Expected layout

```
data/
├── raw/            # original archives exactly as downloaded (e.g. figshare *.zip)
├── interim/        # partially cleaned data (de-identified, re-encoded, deduped)
├── processed/      # model-ready tensors: resized, normalized, train/val/test split
└── splits/         # *.csv or *.json manifests: filename -> split, label, patient_id
```

## Rules

- **One row per patient per split.** All slices from a single patient must land in exactly
  one of `train/`, `val/`, `test/` to prevent data leakage (the most common failure mode in
  medical imaging projects).
- Store a `manifest.csv` with columns: `path, label, patient_id, split, source`.
- Record provenance (dataset name, version, license, download date) in
  `data/dataset_card.md`.
- Prefer DICOM/NIfTI readers (`pydicom`, `SimpleITK`) over raw JPEG/PNG if the source
  allows it, so that acquisition metadata is not lost.

## Common public sources for brain tumor MRI

- BraTS 2020/2021 (multi-institution glioma segmentation; 4 sequences: FLAIR, T1, T1-CE, T2)
- Kaggle "Brain Tumor MRI Dataset" (glioma / meningioma / pituitary / no-tumor)
