"""
Dataset ghép cặp hai mắt (Binocular) cho ODIR-5K Phase 1.

Mỗi mẫu = MỘT bệnh nhân, gồm ảnh mắt trái + mắt phải (nếu có), nhãn nhị phân ở mức bệnh nhân
(0 = Normal nếu cả 2 mắt bình thường, 1 = Pathological nếu ≥1 mắt có bệnh), và tuổi.

Đặc điểm:
- Ghép cặp theo Patient ID (cột 'ID' trong CSV), chống rò rỉ thông tin giữa train/val/test.
- Xử lý bệnh nhân thiếu một mắt: nạp ảnh đen (zero tensor) + cờ boolean missing để model bỏ qua.
- Lọc tuổi bất thường (< age_min_filter, mặc định 5) để tránh nhiễu.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.transforms import get_transforms

# Tắt đa luồng OpenCV để tránh tranh chấp với DataLoader multiprocessing
cv2.setNumThreads(0)


class BinocularDataset(Dataset):
    """Dataset song nhãn: mỗi phần tử là một bệnh nhân (cặp mắt trái + phải)."""

    def __init__(
        self,
        csv_path: str | Path,
        img_dir: str | Path,
        transforms: Any = None,
        img_size: int = 384,
        age_mean: float | None = None,
        age_std: float | None = None,
        age_min_filter: int = 5,
    ) -> None:
        """
        Args:
            csv_path: Đường dẫn train.csv / val.csv / test.csv.
            img_dir: Thư mục chứa ảnh (raw hoặc enhanced) — file dạng <id>_left.jpg, <id>_right.jpg.
            transforms: Pipeline Albumentations (None → chuẩn hóa cơ bản [0,1]).
            img_size: Kích thước ảnh đầu vào (dùng cho zero tensor của mắt thiếu).
            age_mean, age_std: Thống kê tuổi (từ train) để chuẩn hóa Z-score.
            age_min_filter: Lọc bỏ bệnh nhân có tuổi < ngưỡng này.
        """
        df = pd.read_csv(csv_path)

        # Lọc tuổi bất thường
        if age_min_filter > 0:
            before = len(df)
            df = df[df["Patient Age"] >= age_min_filter].copy()
            removed = before - len(df)
            if removed > 0:
                print(f"[BinocularDataset] Loại {removed} dòng tuổi < {age_min_filter} "
                      f"trong {Path(csv_path).name}")

        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transforms = transforms
        self.img_size = img_size
        self.age_mean = age_mean
        self.age_std = age_std

        # --- Ghép cặp theo Patient ID ---
        # ODIR-5K lưu mỗi ảnh là 1 dòng → gộp lại theo 'ID' để có cặp mắt trái/phải.
        self.patients: list[dict] = []
        for patient_id, group in self.df.groupby("ID"):
            first = group.iloc[0]
            age = float(first["Patient Age"])
            # Nhãn nhị phân: N=1 (Normal) → 0 ; N=0 (có bệnh) → 1
            label = 1 - int(first["N"])

            left_fn = right_fn = None
            for _, row in group.iterrows():
                fn = str(row["filename"])
                if "_left" in fn:
                    left_fn = fn
                elif "_right" in fn:
                    right_fn = fn

            self.patients.append({
                "patient_id": int(patient_id),
                "age": age,
                "label": label,
                "left_filename": left_fn,
                "right_filename": right_fn,
            })

        print(f"[BinocularDataset] {len(self.patients)} bệnh nhân từ {Path(csv_path).name}")

    def __len__(self) -> int:
        return len(self.patients)

    def _load_eye(self, filename: str | None) -> tuple[torch.Tensor, bool]:
        """Đọc + transform một ảnh mắt. Trả về (tensor, is_missing).

        Nếu filename None (thiếu mắt) → trả về zero tensor và cờ missing=True.
        """
        if filename is None:
            zero = torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)
            return zero, True

        path = self.img_dir / filename
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            tensor = self.transforms(image=img)["image"]
        else:
            # Dự phòng: resize thủ công + chuẩn hóa [0,1]
            img = cv2.resize(img, (self.img_size, self.img_size))
            tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        return tensor, False

    def __getitem__(self, idx: int) -> dict[str, Any]:
        patient = self.patients[idx]

        left_tensor, left_missing = self._load_eye(patient["left_filename"])
        right_tensor, right_missing = self._load_eye(patient["right_filename"])

        # Chuẩn hóa tuổi Z-score
        age = patient["age"]
        if self.age_mean is not None and self.age_std is not None:
            age = (age - self.age_mean) / self.age_std

        return {
            "left_image": left_tensor,
            "right_image": right_tensor,
            "left_missing": torch.tensor(left_missing, dtype=torch.bool),
            "right_missing": torch.tensor(right_missing, dtype=torch.bool),
            "label": torch.FloatTensor([patient["label"]]),
            "age": torch.FloatTensor([age]),
            "patient_id": patient["patient_id"],
        }


def default_collate(batch: list[dict]) -> dict[str, Any]:
    """Gom lô tiêu chuẩn (không tăng cường) cho DataLoader song nhãn."""
    return {
        "left_image": torch.stack([s["left_image"] for s in batch]),
        "right_image": torch.stack([s["right_image"] for s in batch]),
        "left_missing": torch.stack([s["left_missing"] for s in batch]),
        "right_missing": torch.stack([s["right_missing"] for s in batch]),
        "label": torch.stack([s["label"] for s in batch]),
        "age": torch.stack([s["age"] for s in batch]),
        "patient_id": [s["patient_id"] for s in batch],
    }


def compute_age_stats(train_csv: str | Path, age_min_filter: int = 5) -> tuple[float, float]:
    """Tính mean/std tuổi trên tập Train (sau khi lọc tuổi) để chuẩn hóa nhất quán."""
    df = pd.read_csv(train_csv)
    if age_min_filter > 0:
        df = df[df["Patient Age"] >= age_min_filter]
    ages = df["Patient Age"].values.astype(float)
    return float(ages.mean()), float(ages.std())


def build_dataloaders(
    splits_dir: str | Path,
    img_dir: str | Path,
    img_size: int = 384,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    age_min_filter: int = 5,
    train_collate: Callable | None = None,
) -> tuple[dict[str, DataLoader], float, float]:
    """Tạo DataLoader cho train/val/test ở chế độ song nhãn.

    Args:
        splits_dir: Thư mục chứa train/val/test.csv.
        img_dir: Thư mục ảnh.
        img_size, batch_size, num_workers, pin_memory: Tham số DataLoader.
        age_min_filter: Lọc tuổi bất thường.
        train_collate: Hàm collate cho train (MixUp/CutMix song nhãn). None → default_collate.

    Returns:
        (dataloaders, age_mean, age_std)
    """
    splits_dir = Path(splits_dir)
    img_dir = Path(img_dir)

    age_mean, age_std = compute_age_stats(splits_dir / "train.csv", age_min_filter)
    print(f"[Dataloaders] Tuổi train: mean={age_mean:.2f}, std={age_std:.2f}")

    loaders: dict[str, DataLoader] = {}
    for mode in ("train", "val", "test"):
        csv_path = splits_dir / f"{mode}.csv"
        if not csv_path.exists():
            print(f"[CẢNH BÁO] Thiếu {csv_path}, bỏ qua.")
            continue

        dataset = BinocularDataset(
            csv_path=csv_path,
            img_dir=img_dir,
            transforms=get_transforms(mode=mode, img_size=img_size),
            img_size=img_size,
            age_mean=age_mean,
            age_std=age_std,
            age_min_filter=age_min_filter,
        )

        is_train = (mode == "train")
        collate = (train_collate or default_collate) if is_train else default_collate

        loaders[mode] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=is_train,
            collate_fn=collate,
        )

    return loaders, age_mean, age_std
