"""
PyTorch Dataset cho ODIR-5K.

Hỗ trợ:
  - Multi-task learning: phân loại bệnh (8 nhãn) + dự đoán tuổi
  - Augmentation qua Albumentations
  - Chuẩn hóa tuổi theo training set stats
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.transforms import get_transforms
from src.utils import LABELS, compute_age_stats, compute_class_weights

cv2.setNumThreads(0)  # Tránh conflict với DataLoader multiprocessing


class ODIRDataset(Dataset):
    """Dataset cho ODIR-5K fundus images.

    Mỗi sample trả về:
        image:    FloatTensor [3, H, W]   – ảnh đã transform
        labels:   FloatTensor [8]         – multi-hot labels [N,D,G,C,A,H,M,O]
        age:      FloatTensor [1]         – tuổi đã chuẩn hóa
        filename: str                     – tên file ảnh
    """

    def __init__(
        self,
        csv_path: str | Path,
        img_dir: str | Path,
        transforms: Any = None,
        age_mean: float | None = None,
        age_std: float | None = None,
        age_min_filter: int = 5,
    ) -> None:
        """
        Args:
            csv_path: Đường dẫn CSV split (train.csv, val.csv, test.csv)
            img_dir: Thư mục chứa ảnh enhanced
            transforms: Albumentations Compose object
            age_mean: Mean tuổi từ training set (để chuẩn hóa)
            age_std: Std tuổi từ training set (để chuẩn hóa)
            age_min_filter: Lọc bỏ các hồ sơ có tuổi < giá trị này.
                            Mặc định 5 — loại 28 hồ sơ có tuổi=1 trong ODIR-5K.
                            Đặt 0 để tắt lọc (giữ toàn bộ dữ liệu).
        """
        df = pd.read_csv(csv_path)

        # --- Lọc tuổi bất thường ---
        if age_min_filter > 0:
            before = len(df)
            df = df[df["Patient Age"] >= age_min_filter].copy()
            removed = before - len(df)
            if removed > 0:
                print(
                    f"[ODIRDataset] Đã lọc {removed} hồ sơ có tuổi < {age_min_filter} "
                    f"(còn lại {len(df)}/{before} dòng) — {csv_path.name if hasattr(csv_path, 'name') else csv_path}"
                )

        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transforms = transforms
        self.age_mean = age_mean
        self.age_std = age_std
        self.age_min_filter = age_min_filter

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        # --- Đọc ảnh ---
        img_path = self.img_dir / row["filename"]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Khong tim thay anh: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- Augmentation ---
        if self.transforms is not None:
            augmented = self.transforms(image=image)
            image = augmented["image"]  # Already tensor from ToTensorV2
        else:
            # Fallback: chuyển sang tensor thủ công
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # --- Labels bệnh (multi-hot) ---
        labels = torch.FloatTensor([row[label] for label in LABELS])

        # --- Tuổi (chuẩn hóa nếu có stats) ---
        age = float(row["Patient Age"])
        if self.age_mean is not None and self.age_std is not None:
            age = (age - self.age_mean) / self.age_std
        age = torch.FloatTensor([age])

        return {
            "image": image,
            "labels": labels,
            "age": age,
            "filename": row["filename"],
        }


def get_dataloaders(
    splits_dir: str | Path,
    img_dir: str | Path,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    age_min_filter: int = 5,
) -> dict[str, DataLoader]:
    """Tạo DataLoader cho train/val/test.

    Args:
        splits_dir: Thư mục chứa train.csv, val.csv, test.csv
        img_dir: Thư mục chứa ảnh enhanced
        img_size: 224 (CNN) hoặc 384 (Swin-T)
        batch_size: Batch size
        num_workers: Số workers cho DataLoader
        pin_memory: Pin memory cho GPU
        age_min_filter: Lọc bỏ hồ sơ tuổi < giá trị này (mặc định 5).
                        Đặt 0 để tắt lọc và giữ toàn bộ dữ liệu.

    Returns:
        Dict[str, DataLoader] với keys "train", "val", "test"
    """
    splits_dir = Path(splits_dir)
    img_dir = Path(img_dir)

    # Tính age stats từ training set SAU KHI đã lọc tuổi bất thường
    # → đảm bảo Z-score normalization không bị ảnh hưởng bởi tuổi=1
    train_csv = splits_dir / "train.csv"
    train_df_raw = pd.read_csv(train_csv)
    if age_min_filter > 0:
        train_df_raw = train_df_raw[train_df_raw["Patient Age"] >= age_min_filter]
    ages = train_df_raw["Patient Age"].values.astype(float)
    age_mean = float(ages.mean())
    age_std  = float(ages.std())
    print(f"[get_dataloaders] Age stats (sau lọc tuổi >= {age_min_filter}): "
          f"mean={age_mean:.2f}, std={age_std:.2f}, n={len(ages)}")

    # Tạo datasets
    datasets = {}
    for mode in ["train", "val", "test"]:
        csv_path = splits_dir / f"{mode}.csv"
        if not csv_path.exists():
            print(f"[CANH BAO] Khong tim thay {csv_path}, bo qua")
            continue

        transforms = get_transforms(mode=mode, img_size=img_size)

        datasets[mode] = ODIRDataset(
            csv_path=csv_path,
            img_dir=img_dir,
            transforms=transforms,
            age_mean=age_mean,
            age_std=age_std,
            age_min_filter=age_min_filter,
        )

    # Tạo dataloaders
    dataloaders = {}
    for mode, dataset in datasets.items():
        dataloaders[mode] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(mode == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(mode == "train"),
        )

    return dataloaders


# ---------------------------------------------------------------------------
# __init__.py cho src package
# ---------------------------------------------------------------------------
# Tạo file __init__.py nếu chưa có sẽ được xử lý riêng
