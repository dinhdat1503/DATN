"""Script kiem tra chat luong tien xu ly du lieu."""
from __future__ import annotations
import os
import pandas as pd
import numpy as np
from pathlib import Path

LABELS = ["N", "D", "G", "C", "A", "H", "M", "O"]
CSV_PATH = "archive/full_df.csv"
PREPROCESSED_DIR = "archive/preprocessed_images"
SPLITS_DIR = "archive/splits"


def check_csv(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("1. KIEM TRA full_df.csv")
    print("=" * 60)
    print(f"   Tong so dong: {len(df)}")
    print(f"   Cac cot: {list(df.columns)}")
    print(f"   So benh nhan (ID unique): {df['ID'].nunique()}")
    print(f"   Gioi tinh: {df['Patient Sex'].value_counts().to_dict()}")
    print()


def check_age(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("2. KIEM TRA TUOI BENH NHAN")
    print("=" * 60)
    ages = df["Patient Age"]
    print(f"   Min: {ages.min()}, Max: {ages.max()}, Mean: {ages.mean():.2f}, Std: {ages.std():.2f}")

    outliers = df[(ages < 5) | (ages > 100)]
    if len(outliers) > 0:
        print(f"   [CANH BAO] {len(outliers)} dong tuoi bat thuong (<5 hoac >100):")
        for _, row in outliers.iterrows():
            print(f"      ID={row['ID']}, Age={row['Patient Age']}, file={row['filename']}")
    else:
        print("   [OK] Khong co tuoi bat thuong")

    # Kiem tra tuoi null / NaN
    null_age = ages.isna().sum()
    if null_age > 0:
        print(f"   [CANH BAO] {null_age} dong tuoi bi NULL")
    else:
        print("   [OK] Khong co tuoi NULL")
    print()


def check_labels(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("3. PHAN BO NHAN BENH")
    print("=" * 60)
    for label in LABELS:
        count = int(df[label].sum())
        pct = count / len(df) * 100
        print(f"   {label}: {count:>5} ({pct:.1f}%)")

    total = df[LABELS].sum().sum()
    print(f"   Tong cong nhan: {int(total)}")

    # Dong khong co nhan
    row_sums = df[LABELS].sum(axis=1)
    no_label = (row_sums == 0).sum()
    print(f"   [CHECK] Dong khong co nhan nao (tat ca 0): {no_label}")
    if no_label > 0:
        bad = df[row_sums == 0]
        for _, row in bad.head(5).iterrows():
            print(f"      ID={row['ID']}, file={row['filename']}")

    # Multi-label
    multi = (row_sums > 1).sum()
    print(f"   [CHECK] Dong multi-label (>1 nhan): {multi}")

    # Label duplicates
    dup_ids = df[df.duplicated(subset=["ID"], keep=False)]
    dup_count = dup_ids["ID"].nunique()
    print(f"   [CHECK] So benh nhan co nhieu dong (left+right): {dup_count}")
    print()


def check_images(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("4. KIEM TRA ANH PREPROCESSED")
    print("=" * 60)
    preprocessed = Path(PREPROCESSED_DIR)
    disk_files = set(f for f in os.listdir(preprocessed) if f.endswith(".jpg"))
    csv_files = set(df["filename"].values)

    print(f"   Anh tren disk: {len(disk_files)}")
    print(f"   Anh trong CSV: {len(csv_files)}")

    missing_on_disk = csv_files - disk_files
    orphan_on_disk = disk_files - csv_files

    print(f"   Anh trong CSV nhung THIEU tren disk: {len(missing_on_disk)}")
    if len(missing_on_disk) > 0:
        for f in list(missing_on_disk)[:10]:
            print(f"      - {f}")

    print(f"   Anh tren disk nhung THIEU trong CSV: {len(orphan_on_disk)}")
    if len(orphan_on_disk) > 0:
        for f in list(orphan_on_disk)[:10]:
            print(f"      - {f}")
    print()


def check_image_sizes(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("5. KIEM TRA KICH THUOC ANH")
    print("=" * 60)
    try:
        from PIL import Image
    except ImportError:
        print("   [SKIP] Can cai PIL/Pillow de kiem tra kich thuoc anh")
        print()
        return

    preprocessed = Path(PREPROCESSED_DIR)
    sizes = []
    file_sizes = []
    sample_files = list(preprocessed.glob("*.jpg"))[:200]

    for img_path in sample_files:
        try:
            with Image.open(img_path) as img:
                sizes.append(img.size)
            file_sizes.append(img_path.stat().st_size)
        except Exception as e:
            print(f"   [LOI] Khong doc duoc: {img_path.name}: {e}")

    if sizes:
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        print(f"   Kiem tra {len(sizes)} anh mau (tren tong {len(list(preprocessed.glob('*.jpg')))})")
        print(f"   Width  - min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.0f}")
        print(f"   Height - min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.0f}")
        unique_sizes = set(sizes)
        print(f"   So kich thuoc khac nhau: {len(unique_sizes)}")
        if len(unique_sizes) <= 5:
            for s in unique_sizes:
                c = sizes.count(s)
                print(f"      {s[0]}x{s[1]}: {c} anh")
        else:
            print(f"   [CANH BAO] Anh chua duoc resize ve cung kich thuoc!")

        # Kich thuoc file
        print(f"   File size - min: {min(file_sizes)/1024:.1f} KB, max: {max(file_sizes)/1024:.1f} KB, mean: {np.mean(file_sizes)/1024:.1f} KB")

        # Kiem tra anh qua nho (co the bi loi)
        tiny = [s for s in file_sizes if s < 5000]
        if tiny:
            print(f"   [CANH BAO] {len(tiny)} anh co kich thuoc file < 5KB (co the bi loi)")
    print()


def check_splits() -> None:
    print("=" * 60)
    print("6. KIEM TRA SPLITS (Train/Val/Test)")
    print("=" * 60)
    splits_dir = Path(SPLITS_DIR)

    train = pd.read_csv(splits_dir / "train.csv")
    val = pd.read_csv(splits_dir / "val.csv")
    test = pd.read_csv(splits_dir / "test.csv")

    print(f"   Train: {len(train)} dong, {train['ID'].nunique()} benh nhan")
    print(f"   Val:   {len(val)} dong, {val['ID'].nunique()} benh nhan")
    print(f"   Test:  {len(test)} dong, {test['ID'].nunique()} benh nhan")
    total = len(train) + len(val) + len(test)
    print(f"   Tong:  {total} dong")

    # Kiem tra patient leakage
    train_ids = set(train["ID"].unique())
    val_ids = set(val["ID"].unique())
    test_ids = set(test["ID"].unique())

    leak_tv = train_ids & val_ids
    leak_tt = train_ids & test_ids
    leak_vt = val_ids & test_ids

    if leak_tv:
        print(f"   [LOI NGHIEM TRONG] Patient leakage Train-Val: {len(leak_tv)} benh nhan!")
    else:
        print("   [OK] Khong co patient leakage Train-Val")

    if leak_tt:
        print(f"   [LOI NGHIEM TRONG] Patient leakage Train-Test: {len(leak_tt)} benh nhan!")
    else:
        print("   [OK] Khong co patient leakage Train-Test")

    if leak_vt:
        print(f"   [LOI NGHIEM TRONG] Patient leakage Val-Test: {len(leak_vt)} benh nhan!")
    else:
        print("   [OK] Khong co patient leakage Val-Test")

    # Kiem tra phan bo nhan cua moi split
    print()
    print("   Phan bo nhan theo split:")
    header = f"   {'Label':>6}"
    for s in ["Train", "Val", "Test"]:
        header += f" | {s:>12}"
    print(header)
    print("   " + "-" * 50)

    for label in LABELS:
        row = f"   {label:>6}"
        for split_df in [train, val, test]:
            c = int(split_df[label].sum())
            pct = c / len(split_df) * 100
            row += f" | {c:>5} ({pct:>4.1f}%)"
        print(row)

    # Kiem tra phan bo tuoi
    print()
    print("   Phan bo tuoi theo split:")
    for name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
        ages = split_df["Patient Age"]
        print(f"   {name:>5}: mean={ages.mean():.1f}, std={ages.std():.1f}, "
              f"min={ages.min()}, max={ages.max()}")

    # Kiem tra filepath trong split co tro dung den anh
    print()
    preprocessed = Path(PREPROCESSED_DIR)
    for name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
        if "filename" in split_df.columns:
            missing = 0
            for fn in split_df["filename"]:
                if not (preprocessed / fn).exists():
                    missing += 1
            if missing > 0:
                print(f"   [CANH BAO] {name}: {missing} anh trong split nhung KHONG co tren disk")
            else:
                print(f"   [OK] {name}: Tat ca anh deu ton tai tren disk")
    print()


def check_duplicates(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("7. KIEM TRA TRUNG LAP")
    print("=" * 60)
    # Kiem tra filename trung lap
    dup_files = df[df.duplicated(subset=["filename"], keep=False)]
    if len(dup_files) > 0:
        print(f"   [CANH BAO] {len(dup_files)} dong co filename trung lap!")
        for fn in dup_files["filename"].unique()[:5]:
            print(f"      - {fn}")
    else:
        print("   [OK] Khong co filename trung lap")

    # Kiem tra dong hoan toan trung lap
    full_dup = df[df.duplicated(keep=False)]
    if len(full_dup) > 0:
        print(f"   [CANH BAO] {len(full_dup)} dong hoan toan trung lap!")
    else:
        print("   [OK] Khong co dong hoan toan trung lap")
    print()


def check_left_right_consistency(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("8. KIEM TRA TINH NHAT QUAN LEFT/RIGHT")
    print("=" * 60)
    # Kiem tra moi benh nhan co bao nhieu anh
    imgs_per_patient = df.groupby("ID")["filename"].count()
    print(f"   So anh trung binh/benh nhan: {imgs_per_patient.mean():.2f}")
    print(f"   Benh nhan chi co 1 anh: {(imgs_per_patient == 1).sum()}")
    print(f"   Benh nhan co 2 anh: {(imgs_per_patient == 2).sum()}")
    print(f"   Benh nhan co >2 anh: {(imgs_per_patient > 2).sum()}")

    # Kiem tra left/right
    left_count = df["filename"].str.contains("_left").sum()
    right_count = df["filename"].str.contains("_right").sum()
    print(f"   Anh mat trai (left): {left_count}")
    print(f"   Anh mat phai (right): {right_count}")
    print()


def main() -> None:
    print()
    print("*" * 60)
    print("  KIEM TRA TIEN XU LY DU LIEU - ODIR-5K")
    print("*" * 60)
    print()

    df = pd.read_csv(CSV_PATH)

    check_csv(df)
    check_age(df)
    check_labels(df)
    check_duplicates(df)
    check_left_right_consistency(df)
    check_images(df)
    check_image_sizes(df)
    check_splits()

    print("=" * 60)
    print("HOAN TAT KIEM TRA!")
    print("=" * 60)


if __name__ == "__main__":
    main()
