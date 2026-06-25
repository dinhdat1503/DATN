"""
Dataset ghép cặp hai mắt (Binocular Dataset) cho ODIR-5K.

File này chứa lớp BinocularDataset phục vụ cho việc huấn luyện mô hình Siamese,
xử lý đồng thời cả hai ảnh mắt (trái và phải) của cùng một bệnh nhân.
Đặc biệt hỗ trợ xử lý các trường hợp bệnh nhân chỉ có 1 mắt trong tập dữ liệu (thiếu mắt).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.transforms import get_transforms

# Vô hiệu hóa phân luồng của OpenCV để tránh tranh chấp tài nguyên với DataLoader multiprocessing
cv2.setNumThreads(0)


class BinocularDataset(Dataset):
    """
    Dataset phục vụ huấn luyện Siamese Network xử lý dữ liệu hai mắt của bệnh nhân.

    Mỗi sample trả về một bộ từ điển chứa ảnh và thông tin của cả mắt trái và mắt phải,
    bao gồm cả nhãn bệnh nhân (0 = Bình thường, 1 = Bệnh lý) và tuổi của bệnh nhân.
    """

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
        Khởi tạo Dataset hai mắt.

        Args:
            csv_path: Đường dẫn tới file CSV chứa thông tin split (train.csv, val.csv, test.csv)
            img_dir: Thư mục chứa các ảnh nhãn khoa đáy mắt đã tiền xử lý
            transforms: Pipeline tăng cường dữ liệu của Albumentations
            img_size: Kích thước ảnh đầu vào (384 cho Swin, 224 cho CNN)
            age_mean: Giá trị trung bình của tuổi từ tập train dùng để chuẩn hóa Z-score
            age_std: Độ lệch chuẩn của tuổi từ tập train dùng để chuẩn hóa Z-score
            age_min_filter: Ngưỡng lọc bỏ tuổi bất thường (mặc định dưới 5 tuổi sẽ bị lọc bỏ)
        """
        # Đọc dữ liệu từ file CSV
        df = pd.read_csv(csv_path)

        # --- Lọc các hồ sơ có tuổi bất thường (ví dụ: tuổi bằng 1) ---
        if age_min_filter > 0:
            before_len = len(df)
            df = df[df["Patient Age"] >= age_min_filter].copy()
            removed = before_len - len(df)
            if removed > 0:
                print(
                    f"[BinocularDataset] Đã loại bỏ {removed} dòng có tuổi < {age_min_filter} "
                    f"trong file {Path(csv_path).name}"
                )

        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transforms = transforms
        self.img_size = img_size
        self.age_mean = age_mean
        self.age_std = age_std

        # --- Ghép cặp dữ liệu theo Patient ID ---
        # Do ODIR-5K lưu mỗi ảnh là một dòng trong CSV (ảnh trái và ảnh phải là các dòng riêng biệt),
        # ta cần gộp các dòng này lại theo mã ID của bệnh nhân để huấn luyện song mắt (binocular).
        self.patients = []
        grouped = self.df.groupby("ID")

        for patient_id, group in grouped:
            # Lấy thông tin chung của bệnh nhân từ dòng đầu tiên trong nhóm
            first_row = group.iloc[0]
            age = float(first_row["Patient Age"])
            
            # Xác định nhãn phân loại nhị phân (Normal vs Pathological):
            # Trong ODIR-5K, nhãn cột 'N' đại diện cho Normal (Bình thường).
            # Do đó: Nếu N = 1 (Bình thường) -> Nhãn nhị phân = 0
            #        Nếu N = 0 (Có bệnh lý)    -> Nhãn nhị phân = 1
            label = 1 - int(first_row["N"])

            left_filename = None
            right_filename = None

            # Quét qua các dòng của bệnh nhân này để tìm tên file ảnh mắt trái và mắt phải
            for _, row in group.iterrows():
                fn = row["filename"]
                if "_left" in fn:
                    left_filename = fn
                elif "_right" in fn:
                    right_filename = fn

            # Lưu lại cấu trúc thông tin bệnh nhân
            self.patients.append({
                "patient_id": int(patient_id),
                "age": age,
                "label": label,
                "left_filename": left_filename,
                "right_filename": right_filename,
            })

        print(f"[BinocularDataset] Đã tải {len(self.patients)} bệnh nhân từ {Path(csv_path).name}")

    def __len__(self) -> int:
        """Trả về tổng số bệnh nhân sau khi đã ghép cặp."""
        return len(self.patients)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Tải và xử lý ảnh hai mắt của bệnh nhân tại vị trí chỉ định.

        Nếu bệnh nhân thiếu một mắt (mắt trái hoặc phải không có trong CSV của split hiện tại),
        ta sẽ tạo một tensor toàn chữ số 0 (zero tensor) đại diện cho mắt thiếu,
        đồng thời thiết lập cờ đánh dấu thiếu mắt tương ứng thành True.
        """
        patient = self.patients[idx]

        # Kiểm tra xem ảnh mắt trái hoặc mắt phải có bị thiếu hay không
        left_missing = patient["left_filename"] is None
        right_missing = patient["right_filename"] is None

        # --- Đọc và chuyển đổi ảnh mắt trái ---
        if not left_missing:
            left_path = self.img_dir / patient["left_filename"]
            left_img = cv2.imread(str(left_path))
            if left_img is None:
                raise FileNotFoundError(f"Không thể đọc ảnh mắt trái: {left_path}")
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        else:
            left_img = None

        # --- Đọc và chuyển đổi ảnh mắt phải ---
        if not right_missing:
            right_path = self.img_dir / patient["right_filename"]
            right_img = cv2.imread(str(right_path))
            if right_img is None:
                raise FileNotFoundError(f"Không thể đọc ảnh mắt phải: {right_path}")
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        else:
            right_img = None

        # --- Áp dụng Transform tăng cường dữ liệu ---
        # Nếu mắt có tồn tại, thực hiện transform và chuyển thành Tensor.
        # Nếu mắt bị khuyết thiếu, khởi tạo một Tensor rỗng (zero tensor) có kích thước tương đương [3, H, W].
        if self.transforms is not None:
            if not left_missing:
                left_tensor = self.transforms(image=left_img)["image"]
            else:
                left_tensor = torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)

            if not right_missing:
                right_tensor = self.transforms(image=right_img)["image"]
            else:
                right_tensor = torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)
        else:
            # Phương án dự phòng nếu không truyền transform: Chuẩn hóa cơ bản [0, 1] và đổi chiều tensor
            if not left_missing:
                left_tensor = torch.from_numpy(left_img.transpose(2, 0, 1)).float() / 255.0
            else:
                left_tensor = torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)

            if not right_missing:
                right_tensor = torch.from_numpy(right_img.transpose(2, 0, 1)).float() / 255.0
            else:
                right_tensor = torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)

        # --- Chuẩn hóa độ tuổi bệnh nhân bằng Z-score ---
        age = patient["age"]
        if self.age_mean is not None and self.age_std is not None:
            age = (age - self.age_mean) / self.age_std
        age_tensor = torch.FloatTensor([age])

        # --- Tạo nhãn phân loại nhị phân ---
        label_tensor = torch.FloatTensor([patient["label"]])

        # Trả về bộ dữ liệu của bệnh nhân
        return {
            "left_image": left_tensor,
            "right_image": right_tensor,
            "left_missing": torch.tensor(left_missing, dtype=torch.bool),
            "right_missing": torch.tensor(right_missing, dtype=torch.bool),
            "label": label_tensor,
            "age": age_tensor,
            "patient_id": patient["patient_id"],
        }


def get_binocular_dataloaders(
    splits_dir: str | Path,
    img_dir: str | Path,
    img_size: int = 384,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    age_min_filter: int = 5,
    collate_fn: Any = None,
) -> dict[str, DataLoader]:
    """
    Hàm tiện ích khởi tạo DataLoader cho cả 3 tập Train, Val, Test ở chế độ song mắt.

    Args:
        splits_dir: Thư mục chứa các tệp train.csv, val.csv, test.csv
        img_dir: Thư mục chứa ảnh đáy mắt nhãn khoa
        img_size: Kích thước ảnh đầu vào
        batch_size: Kích thước lô dữ liệu
        num_workers: Số luồng xử lý song song để nạp dữ liệu
        pin_memory: Tự động copy tensor vào page-locked memory để tăng tốc truyền lên GPU
        age_min_filter: Bộ lọc lọc tuổi bất thường
        collate_fn: Hàm tùy chọn dùng để gom lô đặc biệt (ví dụ: MixUp/CutMix song mắt)

    Returns:
        Một dictionary chứa DataLoader cho các pha: "train", "val", "test"
    """
    splits_dir = Path(splits_dir)
    img_dir = Path(img_dir)

    # Tính toán giá trị trung bình (mean) và độ lệch chuẩn (std) tuổi trên tập Train để chuẩn hóa nhất quán
    train_csv = splits_dir / "train.csv"
    train_df_raw = pd.read_csv(train_csv)
    if age_min_filter > 0:
        train_df_raw = train_df_raw[train_df_raw["Patient Age"] >= age_min_filter]
    
    ages = train_df_raw["Patient Age"].values.astype(float)
    age_mean = float(ages.mean())
    age_std = float(ages.std())
    print(f"[get_binocular_dataloaders] Thống kê tuổi tập Train (sau khi lọc >= {age_min_filter}): "
          f"mean={age_mean:.2f}, std={age_std:.2f}, tổng số mẫu={len(ages)}")

    dataloaders = {}
    for mode in ["train", "val", "test"]:
        csv_path = splits_dir / f"{mode}.csv"
        if not csv_path.exists():
            print(f"[CẢNH BÁO] Không tìm thấy tệp split: {csv_path}, bỏ qua.")
            continue

        # Lấy transform tương ứng với pha hiện tại
        transforms = get_transforms(mode=mode, img_size=img_size)

        # Khởi tạo dataset
        dataset = BinocularDataset(
            csv_path=csv_path,
            img_dir=img_dir,
            transforms=transforms,
            img_size=img_size,
            age_mean=age_mean,
            age_std=age_std,
            age_min_filter=age_min_filter,
        )

        # DataLoader của tập train có xáo trộn dữ liệu (shuffle=True)
        # Chỉ áp dụng collate_fn (MixUp/CutMix) cho tập Train
        current_collate = collate_fn if mode == "train" else None

        dataloaders[mode] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(mode == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(mode == "train"),
            collate_fn=current_collate,
        )

    return dataloaders
