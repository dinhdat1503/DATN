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
from .retfound_mtl import RETFoundMTL
from .binocular_classifier import BinocularClassifier


def build_model(
    model_type: str = "cnn",
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout_cls: float = 0.3,
    dropout_reg: float = 0.2,
    num_labels: int = 8,
    # Swin/ViT-specific args
    img_size: int = 224,
    variant: str = "tiny",
    pretrained_path: str | None = None,
    # Hỗ trợ song mắt
    binocular: bool = False,
) -> nn.Module:
    """Tạo model theo kiến trúc được chỉ định.

    Args:
        model_type:      'cnn' (EfficientNet-B0), 'swin' (Swin Transformer) hoặc 'retfound' (RETFound).
        pretrained:      Tải ImageNet/RETFound weights nếu thư viện hỗ trợ.
        freeze_backbone: Đóng băng backbone.
        dropout_cls:     Dropout tỉ lệ cho classification head.
        dropout_reg:     Dropout tỉ lệ cho regression head.
        num_labels:      Số nhãn bệnh (8 cho ODIR-5K, 1 cho nhị phân).
        img_size:        Kích thước ảnh — dùng cho Swin / RETFound (224 hoặc 384).
        variant:         Biến thể Swin: 'tiny' | 'small' | 'base'.
        pretrained_path: Đường dẫn tới trọng số RETFound pre-trained (.pth).
        binocular:       Bật chế độ Siamese ghép cặp 2 mắt (song nhãn) y sinh.

    Returns:
        Model instance.

    Raises:
        ValueError: Nếu model_type không hợp lệ.
    """
    import torch.nn as nn
    model_type = model_type.lower().strip()

    # --- Trường hợp chế độ song mắt (Binocular Mode) ---
    if binocular:
        print(f"[Model] Khởi tạo mô hình Siamese ghép cặp: backbone={model_type}, img_size={img_size}")
        model = BinocularClassifier(
            backbone_type=model_type,
            pretrained=pretrained,
            img_size=img_size,
            dropout=dropout_cls,
        )
        if freeze_backbone:
            model.freeze_backbone()
        return model

    # --- Trường hợp chế độ đơn mắt gốc (Single Eye Mode) ---
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

    elif model_type in ("retfound", "vit_large", "vit"):
        return RETFoundMTL(
            pretrained      = pretrained,
            pretrained_path = pretrained_path,
            img_size        = img_size,
            freeze_backbone = freeze_backbone,
            dropout_cls     = dropout_cls,
            dropout_reg     = dropout_reg,
            num_labels      = num_labels,
        )

    else:
        raise ValueError(
            f"model_type không hợp lệ: '{model_type}'. "
            f"Chọn 'cnn', 'swin' hoặc 'retfound'."
        )


__all__ = ["build_model", "EfficientNetMTL", "SwinMTL", "RETFoundMTL", "BinocularClassifier"]
