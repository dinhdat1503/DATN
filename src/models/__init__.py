"""
Factory cho các model trong dự án ODIR-5K Multi-task Learning.

Hỗ trợ 2 kiến trúc:
    - 'cnn'  → EfficientNet-B0  (img_size=224, feature_dim=1280)
    - 'swin' → Swin-Tiny        (img_size=224 hoặc 384, feature_dim=768)

Sử dụng:
    from src.models import build_model

    # CNN (EfficientNet-B0)
    model = build_model(model_type='cnn', pretrained=True)

    # Swin Transformer Tiny
    model = build_model(model_type='swin', pretrained=True, img_size=224)

    # Swin Transformer Small với img 384
    model = build_model(model_type='swin', pretrained=True, img_size=384, variant='small')
"""
from __future__ import annotations

from .efficientnet_mtl import EfficientNetMTL
from .swin_mtl import SwinMTL


def build_model(
    model_type: str = "cnn",
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout_cls: float = 0.3,
    dropout_reg: float = 0.2,
    num_labels: int = 8,
    # Swin-specific args
    img_size: int = 224,
    variant: str = "tiny",
) -> EfficientNetMTL | SwinMTL:
    """Tạo model theo kiến trúc được chỉ định.

    Args:
        model_type:      'cnn' (EfficientNet-B0) hoặc 'swin' (Swin Transformer).
        pretrained:      Tải ImageNet weights nếu thư viện hỗ trợ.
        freeze_backbone: Đóng băng backbone.
        dropout_cls:     Dropout tỉ lệ cho classification head.
        dropout_reg:     Dropout tỉ lệ cho regression head.
        num_labels:      Số nhãn bệnh (8 cho ODIR-5K).
        img_size:        Kích thước ảnh — chỉ dùng cho Swin (224 hoặc 384).
        variant:         Biến thể Swin: 'tiny' | 'small' | 'base'.

    Returns:
        Model instance (EfficientNetMTL hoặc SwinMTL).

    Raises:
        ValueError: Nếu model_type không hợp lệ.
    """
    model_type = model_type.lower().strip()

    if model_type in ("cnn", "efficientnet", "efficientnet_b0"):
        return EfficientNetMTL(
            pretrained      = pretrained,
            freeze_backbone = freeze_backbone,
            dropout_cls     = dropout_cls,
            dropout_reg     = dropout_reg,
            num_labels      = num_labels,
        )

    elif model_type in ("swin", "swin_transformer", "swin_tiny"):
        return SwinMTL(
            pretrained      = pretrained,
            img_size        = img_size,
            variant         = variant,
            freeze_backbone = freeze_backbone,
            dropout_cls     = dropout_cls,
            dropout_reg     = dropout_reg,
            num_labels      = num_labels,
        )

    else:
        raise ValueError(
            f"model_type không hợp lệ: '{model_type}'. "
            f"Chọn 'cnn' hoặc 'swin'."
        )


__all__ = ["build_model", "EfficientNetMTL", "SwinMTL"]
