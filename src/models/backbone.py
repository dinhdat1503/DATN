"""
Nhà máy (factory) tạo Backbone trích xuất đặc trưng cho ODIR-5K Phase 1.

Dùng trực tiếp thư viện timm — sạch và tiêu chuẩn:
- 'cnn'  → EfficientNet-B0      (feature_dim = 1280)
- 'swin' → Swin Transformer-Tiny (feature_dim = 768)

Backbone trả về vector đặc trưng đã pooling [B, feature_dim] (num_classes=0, global_pool='avg').
Với Swin ở độ phân giải 384, truyền img_size=384 để timm tự nội suy Relative Position Bias.
"""

from __future__ import annotations

import torch.nn as nn


def build_backbone(
    model_type: str = "cnn",
    pretrained: bool = True,
    img_size: int = 384,
) -> tuple[nn.Module, int]:
    """Tạo backbone và trả về (module, feature_dim).

    Args:
        model_type: 'cnn' (EfficientNet-B0) hoặc 'swin' (Swin-Tiny).
        pretrained: Nạp trọng số pretrained ImageNet (cần internet/timm cache).
        img_size: Kích thước ảnh đầu vào (quan trọng với Swin để nội suy vị trí nhúng).

    Returns:
        (backbone, feature_dim)

    Raises:
        ValueError: model_type không hợp lệ.
        ImportError: thiếu thư viện timm.
    """
    model_type = model_type.lower().strip()

    try:
        import timm
    except ImportError as e:
        raise ImportError(
            "Cần cài thư viện timm để tạo backbone: `pip install timm`"
        ) from e

    if model_type in ("cnn", "efficientnet", "efficientnet_b0"):
        backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,       # bỏ tầng phân loại gốc
            global_pool="avg",   # global average pooling → vector đặc trưng
        )
        feature_dim = backbone.num_features  # 1280
        print(f"[Backbone] EfficientNet-B0 (timm)"
              + (" — pretrained ImageNet" if pretrained else " — random init")
              + f", feature_dim={feature_dim}")
        return backbone, feature_dim

    if model_type in ("swin", "swin_tiny", "swin_transformer"):
        # Swin-Tiny: với img_size != 224 cần truyền img_size để nội suy Relative Position Bias
        kwargs = dict(pretrained=pretrained, num_classes=0, global_pool="avg")
        if img_size != 224:
            kwargs["img_size"] = img_size
        backbone = timm.create_model("swin_tiny_patch4_window7_224", **kwargs)
        feature_dim = backbone.num_features  # 768
        print(f"[Backbone] Swin-Tiny (timm, img_size={img_size})"
              + (" — pretrained ImageNet" if pretrained else " — random init")
              + f", feature_dim={feature_dim}")
        return backbone, feature_dim

    raise ValueError(f"model_type không hợp lệ: '{model_type}'. Chọn 'cnn' hoặc 'swin'.")
