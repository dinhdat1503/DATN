"""
Swin Transformer Multi-task Learning cho ODIR-5K.

Kiến trúc: Swin-Tiny (timm) với 2 head đầu ra:
    - Classification Head: Linear(768 → 8)  — multi-label 8 bệnh
    - Regression Head:     Linear(768 → 1)  — dự đoán tuổi

Swin Transformer khác CNN ở chỗ:
    - Dùng Self-Attention thay vì Conv để học đặc trưng
    - Patch-based: chia ảnh thành patches 4×4, embed → tokens
    - Shifted Window Attention: học mối quan hệ TOÀN CỤC giữa các vùng ảnh
    - Input chuẩn: 224×224 (swin_tiny) hoặc 384×384 (swin_small)
    - Feature dim: 768 (Swin-Tiny) / 768 (Swin-Small) / 1024 (Swin-Base)

Fallback (khi không có timm):
    - Dùng lightweight ViT-like model để chạy test local

Sử dụng:
    from src.models.swin_mtl import SwinMTL
    model = SwinMTL(pretrained=True, img_size=224)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class SwinMTL(nn.Module):
    """Swin Transformer Multi-task Learning model.

    Args:
        pretrained:      Tải ImageNet weights nếu timm khả dụng.
        img_size:        Kích thước ảnh đầu vào (224 hoặc 384).
        variant:         'tiny' | 'small' | 'base' — kích thước model.
        freeze_backbone: Đóng băng backbone, chỉ train heads.
        dropout_cls:     Dropout trước classification head.
        dropout_reg:     Dropout trước regression head.
        num_labels:      Số nhãn bệnh (8 cho ODIR-5K).
    """

    # Feature dim theo variant (timm standard)
    FEATURE_DIMS = {
        "tiny":  768,
        "small": 768,
        "base":  1024,
    }

    def __init__(
        self,
        pretrained: bool = True,
        img_size: int = 224,
        variant: str = "tiny",
        freeze_backbone: bool = False,
        dropout_cls: float = 0.3,
        dropout_reg: float = 0.2,
        num_labels: int = 8,
    ) -> None:
        super().__init__()

        self.variant = variant
        self.img_size = img_size
        self.backbone, self.feature_dim = self._build_backbone(
            pretrained, img_size, variant
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.classification_head = nn.Sequential(
            nn.Dropout(p=dropout_cls),
            nn.Linear(self.feature_dim, num_labels),
        )
        self.regression_head = nn.Sequential(
            nn.Dropout(p=dropout_reg),
            nn.Linear(self.feature_dim, 1),
        )
        self._init_heads()

    def _build_backbone(
        self,
        pretrained: bool,
        img_size: int,
        variant: str,
    ) -> tuple[nn.Module, int]:
        """Xây backbone Swin, ưu tiên timm → fallback mini ViT."""

        # Mapping tên model timm theo variant và img_size.
        # Lưu ý: timm chỉ có window12_384 cho base và large.
        # Tiny/Small dùng window7_224 kèm img_size override — timm tự interpolate PE.
        model_names = {
            ("tiny",  224): "swin_tiny_patch4_window7_224",
            ("tiny",  384): "swin_tiny_patch4_window7_224",   # img_size được override bên dưới
            ("small", 224): "swin_small_patch4_window7_224",
            ("small", 384): "swin_small_patch4_window7_224",  # img_size được override bên dưới
            ("base",  224): "swin_base_patch4_window7_224",
            ("base",  384): "swin_base_patch4_window12_384",  # weight chuẩn ở 384
        }
        timm_name = model_names.get((variant, img_size), "swin_tiny_patch4_window7_224")
        feature_dim = self.FEATURE_DIMS.get(variant, 768)

        # Tiny/Small ở 384 cần truyền img_size để timm interpolate positional embeddings
        needs_img_size_override = (variant in ("tiny", "small") and img_size != 224)

        try:
            import timm
            if needs_img_size_override:
                backbone = timm.create_model(
                    timm_name,
                    pretrained=pretrained,
                    num_classes=0,
                    global_pool="avg",
                    img_size=img_size,   # interpolate positional embeddings
                )
            else:
                backbone = timm.create_model(
                    timm_name,
                    pretrained=pretrained,
                    num_classes=0,
                    global_pool="avg",
                )
            print(
                f"[Model] Swin-{variant.capitalize()} ({timm_name}, img_size={img_size})"
                + (" — pretrained ImageNet" if pretrained else " — random init")
            )
            return backbone, backbone.num_features

        except Exception as e:
            print(f"[Model] timm lỗi khi tạo '{timm_name}': {e}")

        # Fallback: mini ViT-like model cho local testing (không pretrained)
        print(
            f"[Model] timm không khả dụng → dùng Mini-ViT fallback (local test only)"
        )
        print(f"[Model] Trên Kaggle/Colab: pip install timm để dùng Swin-{variant.capitalize()}")
        backbone, feature_dim = _build_mini_vit(img_size, feature_dim)
        return backbone, feature_dim

    def _init_heads(self) -> None:
        for m in [self.classification_head, self.regression_head]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                    if layer.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(layer.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        # Đảm bảo features là 2D [B, feature_dim]
        if features.ndim > 2:
            features = features.mean(dim=list(range(2, features.ndim)))
        return {
            "logits":   self.classification_head(features),
            "age_pred": self.regression_head(features),
        }

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def __repr__(self) -> str:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"SwinMTL(variant={self.variant}, img_size={self.img_size}, "
            f"feature_dim={self.feature_dim}, "
            f"total={total:,}, trainable={trainable:,})"
        )


# ---------------------------------------------------------------------------
# Mini ViT fallback (chỉ dùng khi không có timm — local testing)
# ---------------------------------------------------------------------------

def _build_mini_vit(img_size: int, out_dim: int) -> tuple[nn.Module, int]:
    """Lightweight ViT-like model để test pipeline local không cần timm.

    Không dùng để training thực — chỉ để kiểm tra pipeline chạy được.
    Cấu trúc đơn giản: Conv stem → Transformer layers → AvgPool.
    """
    patch_size = 16
    embed_dim  = 256
    num_heads  = 4
    num_layers = 4

    class _PatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.norm = nn.LayerNorm(embed_dim)

        def forward(self, x):
            x = self.proj(x)           # [B, embed_dim, H/p, W/p]
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
            return self.norm(x)

    class _TransformerLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.ff   = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim),
            )
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)

        def forward(self, x):
            attn_out, _ = self.attn(x, x, x)
            x = self.norm1(x + attn_out)
            x = self.norm2(x + self.ff(x))
            return x

    class _MiniViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = _PatchEmbed()
            self.layers      = nn.Sequential(*[_TransformerLayer() for _ in range(num_layers)])
            self.head_proj   = nn.Linear(embed_dim, out_dim)

        def forward(self, x):
            x = self.patch_embed(x)    # [B, N, embed_dim]
            x = self.layers(x)
            x = x.mean(dim=1)          # Global avg over tokens → [B, embed_dim]
            return self.head_proj(x)   # [B, out_dim]

    return _MiniViT(), out_dim
