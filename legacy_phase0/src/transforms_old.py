"""
Augmentation pipeline cho ODIR-5K sử dụng Albumentations.

Hỗ trợ 2 kích thước input:
  - 224×224: cho CNN backbone (ResNet, EfficientNet)
  - 384×384: cho Swin Transformer

Train transforms bao gồm:
  - Geometric: HFlip (p=0.5), ShiftScaleRotate ≤15° (chuẩn giải phẫu y khoa)
  - Color: RandomBrightnessContrast / HueSaturationValue (hue_shift=0 — khóa sắc đỏ võng mạc)
  - Medical: CLAHE (p=0.4, tăng tương phản tổn thương), GaussNoise (p=0.2)
  - Regularization: GaussianBlur (p=0.2), CoarseDropout (p=0.3)
  - Normalize (ImageNet stats) + ToTensorV2

Val/Test transforms:
  - Resize + Normalize + ToTensorV2
"""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalization stats
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(img_size: int = 224) -> A.Compose:
    """Training transforms với augmentation mạnh.

    Args:
        img_size: Kích thước ảnh đầu ra (224 cho CNN, 384 cho Swin-T).
    """
    return A.Compose([
        # Resize
        A.Resize(height=img_size, width=img_size),

        # Geometric augmentations (Chuẩn y khoa võng mạc)
        A.HorizontalFlip(p=0.5),  # Rất an toàn: mắt trái đối xứng mắt phải
        # Đã loại bỏ VerticalFlip và RandomRotate90 vì vi phạm giải phẫu học võng mạc
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,      # Giới hạn xoay nhẹ mô phỏng bệnh nhân nghiêng đầu khi chụp
            border_mode=0,        # BORDER_CONSTANT (viền đen)
            p=0.5,
        ),

        # Color augmentations (Bảo toàn sắc đỏ y khoa)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0,
            ),
            A.HueSaturationValue(
                hue_shift_limit=0,  # Khóa cứng tông màu Hue để không biến đổi màu đỏ của máu võng mạc
                sat_shift_limit=15,
                val_shift_limit=15,
                p=1.0,
            ),
        ], p=0.5),

        # Đã vô hiệu hóa CLAHE online để tránh double CLAHE với ảnh tiền xử lý enhanced_images
        # A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
 
        # Thêm nhiễu cảm biến camera (tương thích Albumentations cả cũ lẫn mới)
        A.GaussNoise(std_range=(0.02, 0.1), p=0.2),
 
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
 
        # Regularization: CoarseDropout thay cho Cutout (tương thích Albumentations 2.x)
        # Giảm kích thước vùng đen để không che khuất hoàng điểm / gai thị
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(img_size // 32, img_size // 16),
            hole_width_range=(img_size // 32, img_size // 16),
            fill=0,
            p=0.3,
        ),

        # Normalize + ToTensor
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 224) -> A.Compose:
    """Validation/Test transforms (chỉ resize và normalize)."""
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_transforms(
    mode: str = "train", img_size: int = 224
) -> A.Compose:
    """Helper để lấy transforms theo mode.

    Args:
        mode: "train", "val", hoặc "test"
        img_size: 224 (CNN) hoặc 384 (Swin-T)
    """
    if mode == "train":
        return get_train_transforms(img_size)
    else:
        return get_val_transforms(img_size)
