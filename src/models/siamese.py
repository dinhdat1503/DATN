"""
Mạng Siamese (Mạng Xiêm) nhị phân song nhãn cho ODIR-5K Phase 1.

Kiến trúc trích xuất đặc trưng độc lập từ hai mắt qua MỘT backbone chia sẻ trọng số
(weight sharing), xử lý mắt khuyết thiếu, ghép nối đặc trưng, rồi đưa qua Fusion MLP và
hai nhánh đầu ra: phân loại nhị phân (Normal vs Pathological) và hồi quy tuổi (phụ trợ).

Output (luôn là dict):
    {"logits": Tensor[B, 1], "age_pred": Tensor[B, 1]}
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.backbone import build_backbone


class BinocularClassifier(nn.Module):
    """Mạng Siamese xử lý cặp mắt trái/phải, phân loại nhị phân + hồi quy tuổi."""

    def __init__(
        self,
        backbone_type: str = "cnn",
        pretrained: bool = True,
        img_size: int = 384,
        dropout: float = 0.3,
    ) -> None:
        """
        Args:
            backbone_type: 'cnn' (EfficientNet-B0) hoặc 'swin' (Swin-Tiny).
            pretrained: Dùng trọng số pretrained ImageNet.
            img_size: Kích thước ảnh đầu vào.
            dropout: Tỷ lệ dropout trong Fusion MLP.
        """
        super().__init__()
        self.backbone_type = backbone_type.lower()
        self.img_size = img_size

        # Backbone chia sẻ trọng số cho cả hai mắt
        self.backbone, self.feature_dim = build_backbone(
            model_type=backbone_type, pretrained=pretrained, img_size=img_size
        )

        # Fusion MLP: ghép nối đặc trưng 2 mắt [2*D] → 512
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * self.feature_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(p=dropout),
        )

        # Nhánh phân loại nhị phân (1 neuron) và hồi quy tuổi (1 neuron)
        self.classification_head = nn.Linear(512, 1)
        self.regression_head = nn.Linear(512, 1)

        self._init_new_layers()

    def _init_new_layers(self) -> None:
        """Khởi tạo He (Kaiming) cho các tầng tuyến tính mới (fusion + 2 heads)."""
        for module in [self.fusion_mlp, self.classification_head, self.regression_head]:
            layers = module if isinstance(module, nn.Sequential) else [module]
            for layer in layers:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

    def forward(
        self,
        left_image: torch.Tensor,
        right_image: torch.Tensor,
        left_missing: torch.Tensor,
        right_missing: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Lan truyền tiến.

        Args:
            left_image, right_image: [B, 3, H, W].
            left_missing, right_missing: [B] boolean — True nếu mắt đó bị thiếu.

        Returns:
            {"logits": [B, 1], "age_pred": [B, 1]}
        """
        # 1. Trích xuất đặc trưng 2 mắt qua backbone chia sẻ
        left_feat = self.backbone(left_image)    # [B, D]
        right_feat = self.backbone(right_image)  # [B, D]

        # 2. Ép đặc trưng của mắt thiếu về 0 (loại nhiễu của ảnh đen)
        left_mask = (~left_missing).float().unsqueeze(1)    # [B, 1]
        right_mask = (~right_missing).float().unsqueeze(1)  # [B, 1]
        left_feat = left_feat * left_mask
        right_feat = right_feat * right_mask

        # 3. Ghép nối + Fusion MLP
        fused = torch.cat([left_feat, right_feat], dim=-1)  # [B, 2D]
        fused = self.fusion_mlp(fused)                       # [B, 512]

        # 4. Hai nhánh đầu ra
        return {
            "logits": self.classification_head(fused),   # [B, 1]
            "age_pred": self.regression_head(fused),     # [B, 1]
        }

    def freeze_backbone(self) -> None:
        """Đóng băng backbone (Stage 1 của two-stage training)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Mở khóa backbone để tinh chỉnh toàn mạng (Stage 2)."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def __repr__(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (f"BinocularClassifier(backbone={self.backbone_type}, "
                f"feature_dim={self.feature_dim}, total={total:,}, trainable={trainable:,})")
