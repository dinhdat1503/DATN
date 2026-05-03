from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

LABEL_COLUMNS = ["N", "D", "G", "C", "A", "H", "M", "O"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create patient-level train/val/test splits for ODIR-style datasets."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("archive/full_df.csv"),
        help="Input metadata CSV path.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("archive/preprocessed_images"),
        help="Directory containing image files referenced by the filename column.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("archive/splits"),
        help="Output directory for split CSV files and summary JSON.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def validate_schema(df: pd.DataFrame) -> None:
    required = {"ID", "filename", "Patient Age", *LABEL_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def infer_patient_strata(df: pd.DataFrame) -> pd.Series:
    patient_labels = df.groupby("ID", as_index=True)[LABEL_COLUMNS].sum()
    dominant_idx = patient_labels.to_numpy().argmax(axis=1)
    strata = pd.Series(
        [LABEL_COLUMNS[idx] for idx in dominant_idx],
        index=patient_labels.index,
        name="stratum",
    )
    return strata


def split_ids_by_stratum(
    strata: pd.Series,
    val_size: float,
    test_size: float,
    seed: int,
) -> Dict[str, set]:
    if val_size <= 0 or test_size <= 0 or (val_size + test_size) >= 1:
        raise ValueError("val_size and test_size must be > 0 and val_size + test_size < 1")

    rng = np.random.default_rng(seed)
    train_ids: set = set()
    val_ids: set = set()
    test_ids: set = set()

    for stratum, series in strata.groupby(strata):
        ids = series.index.to_numpy(copy=True)
        rng.shuffle(ids)
        n = len(ids)

        n_test = int(round(n * test_size))
        n_val = int(round(n * val_size))

        if n >= 3:
            n_test = max(1, min(n_test, n - 2))
            n_val = max(1, min(n_val, n - n_test - 1))
        elif n == 2:
            n_test = 1
            n_val = 0
        else:
            n_test = 0
            n_val = 0

        test_chunk = ids[:n_test]
        val_chunk = ids[n_test : n_test + n_val]
        train_chunk = ids[n_test + n_val :]

        if len(train_chunk) == 0:
            # Guard to keep each stratum represented in train when possible.
            if len(val_chunk) > 0:
                train_chunk = val_chunk[:1]
                val_chunk = val_chunk[1:]
            elif len(test_chunk) > 0:
                train_chunk = test_chunk[:1]
                test_chunk = test_chunk[1:]

        train_ids.update(train_chunk.tolist())
        val_ids.update(val_chunk.tolist())
        test_ids.update(test_chunk.tolist())

    overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    if overlap:
        raise RuntimeError(f"Patient leakage detected across splits: {len(overlap)} IDs")

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def prepare_dataframe(df: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    df = df.copy()
    df["filepath"] = df["filename"].astype(str).map(lambda x: str((image_dir / x).as_posix()))
    df["image_exists"] = df["filename"].astype(str).map(lambda x: (image_dir / x).exists())
    missing_images = int((~df["image_exists"]).sum())
    if missing_images > 0:
        print(f"Warning: {missing_images} rows reference missing image files. They will be dropped.")
        df = df[df["image_exists"]].copy()
    return df.drop(columns=["image_exists"])


def summarize_split(df: pd.DataFrame) -> Dict[str, object]:
    label_counts = {label: int(df[label].sum()) for label in LABEL_COLUMNS}
    return {
        "rows": int(len(df)),
        "patients": int(df["ID"].nunique()),
        "age_mean": float(df["Patient Age"].mean()),
        "age_std": float(df["Patient Age"].std()),
        "label_counts": label_counts,
    }


def print_summary(name: str, stats: Dict[str, object]) -> None:
    print(
        f"{name:>5} | rows={stats['rows']:4d} | patients={stats['patients']:4d} "
        f"| age={stats['age_mean']:.2f}+-{stats['age_std']:.2f}"
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    validate_schema(df)
    df = prepare_dataframe(df, args.image_dir)

    strata = infer_patient_strata(df)
    id_splits = split_ids_by_stratum(strata, args.val_size, args.test_size, args.seed)

    split_frames: Dict[str, pd.DataFrame] = {}
    for split_name, ids in id_splits.items():
        split_df = df[df["ID"].isin(ids)].copy()
        split_df = split_df.sort_values(["ID", "filename"]).reset_index(drop=True)
        split_df.to_csv(args.out_dir / f"{split_name}.csv", index=False)
        split_frames[split_name] = split_df

    summary = {split: summarize_split(frame) for split, frame in split_frames.items()}

    all_ids = {split: set(frame["ID"].unique()) for split, frame in split_frames.items()}
    assert all_ids["train"].isdisjoint(all_ids["val"])
    assert all_ids["train"].isdisjoint(all_ids["test"])
    assert all_ids["val"].isdisjoint(all_ids["test"])

    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Patient-level split created successfully")
    print(f"CSV source: {args.csv.as_posix()}")
    print(f"Image dir:   {args.image_dir.as_posix()}")
    print(f"Output dir:  {args.out_dir.as_posix()}")
    print(f"Ratios: train={1-args.val_size-args.test_size:.2f}, val={args.val_size:.2f}, test={args.test_size:.2f}")
    print("-" * 72)
    for split_name in ["train", "val", "test"]:
        print_summary(split_name, summary[split_name])


if __name__ == "__main__":
    main()
