"""
Làm sạch dữ liệu và rebuild patient-level splits.

Các bước:
  1. Loại bỏ 28 mẫu có Patient Age = 1  (lỗi dữ liệu gốc ODIR-5K)
  2. Cập nhật filepath trỏ tới enhanced_images/
  3. Rebuild train/val/test splits (patient-level, stratified)
  4. Tính class weights (pos_weight cho BCEWithLogitsLoss)
  5. Tính age statistics (mean, std) cho chuẩn hóa tuổi

Sử dụng:
    python scripts/clean_and_rebuild.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["N", "D", "G", "C", "A", "H", "M", "O"]
RANDOM_SEED = 42


def remove_age_outliers(df: pd.DataFrame, min_age: int = 5) -> pd.DataFrame:
    """Loại bỏ bệnh nhân có tuổi bất thường (< min_age)."""
    bad_ids = df[df["Patient Age"] < min_age]["ID"].unique()
    n_before = len(df)
    df_clean = df[~df["ID"].isin(bad_ids)].copy()
    n_after = len(df_clean)
    print(f"Loai bo {len(bad_ids)} benh nhan ({n_before - n_after} dong) "
          f"co Patient Age < {min_age}")
    print(f"  Truoc: {n_before} dong | Sau: {n_after} dong")
    return df_clean


def update_filepath(
    df: pd.DataFrame, img_dir: str = "archive/enhanced_images"
) -> pd.DataFrame:
    """Cập nhật cột filepath trỏ tới thư mục ảnh enhanced."""
    df["filepath"] = df["filename"].apply(lambda fn: f"{img_dir}/{fn}")
    return df


def create_patient_level_splits(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tạo train/val/test splits theo patient-level, stratified.

    Stratify dựa trên nhãn phổ biến nhất của mỗi bệnh nhân.
    """
    # Lấy thông tin mỗi bệnh nhân
    patient_df = df.groupby("ID").agg({
        "Patient Age": "first",
        "Patient Sex": "first",
        **{label: "max" for label in LABELS},  # lấy max để có nhãn bệnh
    }).reset_index()

    # Tạo stratify key = nhãn đơn phổ biến nhất
    patient_df["strat_key"] = patient_df[LABELS].values.argmax(axis=1)

    patient_ids = patient_df["ID"].values
    strat_keys = patient_df["strat_key"].values

    # Split lần 1: train vs (val + test)
    hold_out_ratio = val_ratio + test_ratio
    train_ids, hold_ids, _, hold_strat = train_test_split(
        patient_ids, strat_keys,
        test_size=hold_out_ratio,
        random_state=seed,
        stratify=strat_keys,
    )

    # Split lần 2: val vs test
    val_test_ratio = test_ratio / hold_out_ratio
    val_ids, test_ids = train_test_split(
        hold_ids,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=hold_strat,
    )

    train_df = df[df["ID"].isin(train_ids)].copy()
    val_df = df[df["ID"].isin(val_ids)].copy()
    test_df = df[df["ID"].isin(test_ids)].copy()

    return train_df, val_df, test_df


def compute_class_weights(
    train_df: pd.DataFrame,
) -> dict[str, float]:
    """Tính pos_weight cho BCEWithLogitsLoss.

    pos_weight[i] = num_negative[i] / num_positive[i]
    Giúp xử lý class imbalance cho multi-label classification.
    """
    weights = {}
    n_total = len(train_df)
    for label in LABELS:
        n_pos = int(train_df[label].sum())
        n_neg = n_total - n_pos
        if n_pos > 0:
            weights[label] = round(n_neg / n_pos, 4)
        else:
            weights[label] = 1.0
    return weights


def compute_age_stats(train_df: pd.DataFrame) -> dict[str, float]:
    """Tính mean và std tuổi từ training set cho chuẩn hóa."""
    ages = train_df["Patient Age"].values
    return {
        "mean": round(float(ages.mean()), 4),
        "std": round(float(ages.std()), 4),
        "min": int(ages.min()),
        "max": int(ages.max()),
    }


def print_split_summary(
    name: str, df: pd.DataFrame
) -> None:
    """In thống kê tóm tắt cho 1 split."""
    n_patients = df["ID"].nunique()
    print(f"\n  {name:>5}: {len(df):>5} dong, {n_patients:>4} benh nhan")
    label_str = "    "
    for label in LABELS:
        count = int(df[label].sum())
        pct = count / len(df) * 100
        label_str += f"{label}:{count}({pct:.1f}%) "
    print(label_str)
    print(f"    Tuoi: mean={df['Patient Age'].mean():.1f}, "
          f"std={df['Patient Age'].std():.1f}, "
          f"min={df['Patient Age'].min()}, max={df['Patient Age'].max()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lam sach du lieu va rebuild splits ODIR-5K"
    )
    parser.add_argument(
        "--csv", type=str, default="archive/full_df.csv",
        help="File CSV goc",
    )
    parser.add_argument(
        "--out-dir", type=str, default="archive/splits_clean",
        help="Thu muc luu splits moi",
    )
    parser.add_argument(
        "--img-dir", type=str, default="archive/enhanced_images",
        help="Thu muc anh enhanced",
    )
    parser.add_argument(
        "--min-age", type=int, default=5,
        help="Tuoi toi thieu (loai bo benh nhan < min_age)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Ty le validation set",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.15,
        help="Ty le test set",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Đọc CSV gốc
    print("=" * 60)
    print("1. DOC DU LIEU GOC")
    print("=" * 60)
    df = pd.read_csv(args.csv)
    print(f"Tong dong: {len(df)}, So benh nhan: {df['ID'].nunique()}")

    # 2. Loại bỏ tuổi bất thường
    print()
    print("=" * 60)
    print("2. LOAI BO TUOI BAT THUONG")
    print("=" * 60)
    df_clean = remove_age_outliers(df, min_age=args.min_age)

    # 3. Cập nhật filepath
    print()
    print("=" * 60)
    print("3. CAP NHAT FILEPATH")
    print("=" * 60)
    df_clean = update_filepath(df_clean, img_dir=args.img_dir)
    print(f"Filepath mau: {df_clean['filepath'].iloc[0]}")

    # 4. Lưu CSV sạch
    clean_csv = out_dir / "full_df_clean.csv"
    df_clean.to_csv(clean_csv, index=False)
    print(f"Luu CSV sach: {clean_csv}")

    # 5. Tạo splits mới
    print()
    print("=" * 60)
    print("4. TAO SPLITS MOI")
    print("=" * 60)
    train_df, val_df, test_df = create_patient_level_splits(
        df_clean, val_ratio=args.val_ratio, test_ratio=args.test_ratio
    )

    # Kiểm tra leakage
    train_ids = set(train_df["ID"].unique())
    val_ids = set(val_df["ID"].unique())
    test_ids = set(test_df["ID"].unique())
    assert len(train_ids & val_ids) == 0, "Patient leakage Train-Val!"
    assert len(train_ids & test_ids) == 0, "Patient leakage Train-Test!"
    assert len(val_ids & test_ids) == 0, "Patient leakage Val-Test!"
    print("[OK] Khong co patient leakage")

    print_split_summary("Train", train_df)
    print_split_summary("Val", val_df)
    print_split_summary("Test", test_df)

    # Lưu splits
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    print(f"\nLuu splits vao {out_dir}/")

    # 6. Tính class weights
    print()
    print("=" * 60)
    print("5. TINH CLASS WEIGHTS")
    print("=" * 60)
    weights = compute_class_weights(train_df)
    for label, w in weights.items():
        print(f"  {label}: pos_weight = {w:.4f}")

    # 7. Tính age stats
    print()
    print("=" * 60)
    print("6. TINH AGE STATISTICS")
    print("=" * 60)
    age_stats = compute_age_stats(train_df)
    for k, v in age_stats.items():
        print(f"  {k}: {v}")

    # 8. Lưu metadata
    metadata = {
        "labels": LABELS,
        "class_weights": weights,
        "age_stats": age_stats,
        "splits": {
            "train": {"rows": len(train_df), "patients": int(train_df["ID"].nunique())},
            "val": {"rows": len(val_df), "patients": int(val_df["ID"].nunique())},
            "test": {"rows": len(test_df), "patients": int(test_df["ID"].nunique())},
        },
        "total_rows": len(df_clean),
        "total_patients": int(df_clean["ID"].nunique()),
        "removed_patients": int(len(df) - len(df_clean)),
    }

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nLuu metadata: {meta_path}")

    print()
    print("=" * 60)
    print("HOAN TAT!")
    print("=" * 60)


if __name__ == "__main__":
    main()
