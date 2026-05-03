"""
Augmentation pipeline cho ODIR-5K sử dụng Albumentations.

Hỗ trợ 2 kích thước input:
  - 224×224: cho CNN backbone (ResNet, EfficientNet)
  - 384×384: cho Swin Transformer

Train transforms bao gồm:
  - Geometric: HFlip, VFlip, Rotate, ShiftScaleRotate
  - Color: ColorJitter, RandomBrightnessContrast
  - Advanced: CoarseDropout (thay cho Cutout)
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

        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=30,
            border_mode=0,  # BORDER_CONSTANT (đen)
            p=0.5,
        ),

        # Color augmentations
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=1.0,
            ),
        ], p=0.5),

        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        # Regularization: CoarseDropout thay cho Cutout
        A.CoarseDropout(
            max_holes=8,
            max_height=img_size // 8,
            max_width=img_size // 8,
            min_holes=1,
            fill_value=0,
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
