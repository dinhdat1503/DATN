"""
Pipeline tăng cường ảnh (Albumentations) cho ODIR-5K Phase 1.

Thiết kế ưu tiên TÍNH ỔN ĐỊNH giữa các phiên bản Albumentations (1.x ↔ 2.x) vì
môi trường Kaggle có thể cài phiên bản khác local. Vì vậy chỉ dùng các phép biến đổi
lõi luôn tồn tại và ổn định API.

Quy ước y khoa quan trọng:
- KHÓA kênh màu Hue (hue_shift_limit=0): tổn thương võng mạc như xuất huyết có màu đỏ
  đặc trưng; xoay Hue sẽ phá hủy đặc trưng lâm sàng này.
- Chỉ lật ngang (HorizontalFlip) — mắt trái/phải đối xứng nhau. KHÔNG lật dọc / xoay 90°
  vì vi phạm giải phẫu học đáy mắt.
- KHÔNG CLAHE online: ảnh enhanced đã được CLAHE offline, làm lại sẽ gây "double CLAHE".

Train: Resize → HFlip → ShiftScaleRotate nhẹ → Brightness/Contrast hoặc Sat/Val → Normalize → Tensor
Val/Test: Resize → Normalize → Tensor
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Chuẩn hóa theo thống kê ImageNet (khớp với backbone pretrained ImageNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(img_size: int = 384) -> A.Compose:
    """Tăng cường dữ liệu cho tập huấn luyện.

    Args:
        img_size: Kích thước ảnh đầu ra (384 cho cả CNN và Swin trong dự án này).
    """
    return A.Compose([
        A.Resize(height=img_size, width=img_size),

        # --- Hình học (an toàn với giải phẫu võng mạc) ---
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.10,
            rotate_limit=15,      # Xoay nhẹ mô phỏng bệnh nhân nghiêng đầu khi chụp
            border_mode=0,        # Viền đen (BORDER_CONSTANT)
            p=0.5,
        ),

        # --- Màu sắc (bảo toàn sắc đỏ y khoa: KHÓA Hue) ---
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0,
            ),
            A.HueSaturationValue(
                hue_shift_limit=0,    # KHÓA Hue — không đổi màu đỏ của máu võng mạc
                sat_shift_limit=15,
                val_shift_limit=15,
                p=1.0,
            ),
        ], p=0.5),

        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 384) -> A.Compose:
    """Biến đổi cho tập Val/Test: chỉ resize và chuẩn hóa (không tăng cường)."""
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_transforms(mode: str = "train", img_size: int = 384) -> A.Compose:
    """Helper lấy transform theo pha.

    Args:
        mode: "train" → tăng cường mạnh; "val"/"test" → chỉ resize + normalize.
        img_size: Kích thước ảnh đầu vào.
    """
    if mode == "train":
        return get_train_transforms(img_size)
    return get_val_transforms(img_size)
