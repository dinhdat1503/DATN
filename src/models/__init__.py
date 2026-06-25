"""
Factory model cho ODIR-5K Phase 1 — chỉ Siamese nhị phân song nhãn.

Sử dụng:
    from src.models import build_model
    model = build_model(model_type="cnn", img_size=384)   # hoặc "swin"
"""

from __future__ import annotations

import torch.nn as nn

from src.models.backbone import build_backbone
from src.models.siamese import BinocularClassifier


def build_model(
    model_type: str = "cnn",
    pretrained: bool = True,
    img_size: int = 384,
    dropout: float = 0.3,
) -> nn.Module:
    """Tạo mô hình Siamese song nhãn.

    Args:
        model_type: 'cnn' (EfficientNet-B0) hoặc 'swin' (Swin-Tiny).
        pretrained: Dùng trọng số pretrained ImageNet.
        img_size: Kích thước ảnh đầu vào.
        dropout: Tỷ lệ dropout của Fusion MLP.

    Returns:
        BinocularClassifier.
    """
    print(f"[Model] BinocularClassifier (Siamese): backbone={model_type}, img_size={img_size}")
    return BinocularClassifier(
        backbone_type=model_type,
        pretrained=pretrained,
        img_size=img_size,
        dropout=dropout,
    )


__all__ = ["build_model", "BinocularClassifier", "build_backbone"]
