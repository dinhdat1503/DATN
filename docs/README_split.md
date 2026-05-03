# Patient-level split for ODIR-style data

This folder contains a utility to create train/val/test splits without patient leakage.

## Why this script

In ODIR, one patient can contribute both left and right eye images. If you split by image only, the same patient can appear in both train and validation/test sets, which inflates metrics.

## Run

From workspace root:

```powershell
.\.venv\Scripts\python.exe scripts\build_patient_splits.py
```

Optional custom ratios:

```powershell
.\.venv\Scripts\python.exe scripts\build_patient_splits.py --val-size 0.15 --test-size 0.15 --seed 42
```

## Inputs

- `archive/full_df.csv`
- `archive/preprocessed_images`

## Outputs

- `archive/splits/train.csv`
- `archive/splits/val.csv`
- `archive/splits/test.csv`
- `archive/splits/summary.json`

Each output CSV includes a normalized `filepath` column that points to the image location in the workspace.
