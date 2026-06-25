"""
Mô hình Siamese (Mạng Xiêm) nhị phân hai mắt y sinh cho ODIR-5K.

Mô hình này trích xuất đặc trưng từ hai mắt (mắt trái và mắt phải) bằng cách sử dụng
chung một mạng Backbone (chia sẻ trọng số - weight sharing). Đặc trưng của hai mắt sau đó
được xử lý khuyết thiếu (nếu có), ghép nối (concatenate) và đưa qua một nhánh Fusion MLP
trước khi chuyển vào các nhánh đầu ra dự đoán phân loại bệnh lý và độ tuổi của bệnh nhân.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from src.models.efficientnet_mtl import EfficientNetMTL
from src.models.swin_mtl import SwinMTL


class BinocularClassifier(nn.Module):
    """
    Kiến trúc Siamese Network xử lý dữ liệu song mắt y sinh (left + right).

    Đầu vào:
        left_image: Tensor [B, 3, H, W]
        right_image: Tensor [B, 3, H, W]
        left_missing: Tensor Boolean [B]
        right_missing: Tensor Boolean [B]

    Đầu ra:
        Một dict chứa:
            "logits": Tensor [B, 1] — Dự đoán nhị phân (Bình thường vs Bệnh lý)
            "age_pred": Tensor [B, 1] — Dự đoán tuổi võng mạc (hỗ trợ điều hòa)
    """

    def __init__(
        self,
        backbone_type: str = "cnn",
        pretrained: bool = True,
        img_size: int = 384,
        dropout: float = 0.3,
    ) -> None:
        """
        Khởi tạo mô hình Siamese.

        Args:
            backbone_type: Loại backbone trích xuất đặc trưng ('cnn' hoặc 'swin')
            pretrained: Sử dụng trọng số pre-trained ImageNet hay không
            img_size: Kích thước ảnh đầu vào (384 cho Swin, 224 cho CNN)
            dropout: Tỷ lệ ngắt kết nối neuron ngẫu nhiên để điều hòa mô hình
        """
        super().__init__()
        self.backbone_type = backbone_type.lower()
        self.img_size = img_size

        # --- BƯỚC 1: Trích xuất và cấu hình Backbone chia sẻ trọng số ---
        # Ta tái sử dụng bộ khung trích xuất đặc trưng cực kỳ ổn định từ các file MTL có sẵn
        if self.backbone_type == "cnn":
            # EfficientNet-B0: FEATURE_DIM = 1280
            base_model = EfficientNetMTL(pretrained=pretrained, num_labels=1)
            self.backbone = base_model.backbone
            self.feature_dim = base_model.FEATURE_DIM
        elif self.backbone_type == "swin":
            # Swin Transformer Tiny: feature_dim = 768
            base_model = SwinMTL(
                pretrained=pretrained,
                img_size=img_size,
                variant="tiny",
                num_labels=1
            )
            self.backbone = base_model.backbone
            self.feature_dim = base_model.feature_dim
        else:
            raise ValueError(f"Không hỗ trợ backbone_type: {backbone_type}. Chọn 'cnn' hoặc 'swin'.")

        # --- BƯỚC 2: Định nghĩa tầng Fusion MLP (Kết hợp hai mắt) ---
        # Do ta ghép nối đặc trưng của mắt trái và mắt phải, số chiều đầu vào sẽ tăng gấp đôi (2 * feature_dim)
        # Tầng này chiếu đặc trưng ghép nối xuống không gian 512 chiều để phân loại.
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * self.feature_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(p=dropout),
        )

        # --- BƯỚC 3: Nhánh phân loại nhị phân (Binary Classification Head) ---
        # Đầu ra là 1 neuron duy nhất (logits) phục vụ cho dự đoán nhị phân (Normal vs Pathological)
        self.classification_head = nn.Linear(512, 1)

        # --- BƯỚC 4: Nhánh hồi quy tuổi (Auxiliary Age Regression Head) ---
        # Nhiệm vụ phụ trợ hỗ trợ mô hình học thêm thông tin sinh học của cấu trúc võng mạc
        self.regression_head = nn.Linear(512, 1)

        # Khởi tạo trọng số cho các tầng tuyến tính mới theo phương pháp He
        self._init_weights()

    def _init_weights(self) -> None:
        """Khởi tạo trọng số He (Kaiming) cho các tầng fully connected mới."""
        for m in [self.fusion_mlp, self.classification_head, self.regression_head]:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Sequential):
                for layer in m:
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
        """
        Thực hiện lan truyền tiến (Forward Pass).

        Args:
            left_image: Tensor mắt trái [B, 3, H, W]
            right_image: Tensor mắt phải [B, 3, H, W]
            left_missing: Cờ boolean đánh dấu thiếu mắt trái [B]
            right_missing: Cờ boolean đánh dấu thiếu mắt phải [B]

        Returns:
            Dict chứa logits phân loại bệnh ("logits") và dự đoán tuổi ("age_pred")
        """
        # 1. Trích xuất đặc trưng từ hai mắt qua cùng một mạng Backbone chia sẻ trọng số
        left_feat = self.backbone(left_image)    # [B, feature_dim]
        right_feat = self.backbone(right_image)  # [B, feature_dim]

        # 2. Xử lý các trường hợp thiếu mắt (Missing Eyes):
        # Nếu một mắt bị thiếu, ta dùng cờ thiếu mắt để ép đặc trưng tương ứng về 0.
        # Điều này giúp loại bỏ thông tin nhiễu của ảnh giả (zero tensor) đưa vào backbone.
        left_mask = (~left_missing).float().unsqueeze(1)    # [B, 1]
        right_mask = (~right_missing).float().unsqueeze(1)  # [B, 1]

        left_feat = left_feat * left_mask
        right_feat = right_feat * right_mask

        # 3. Ghép nối đặc trưng từ hai mắt
        fused_feat = torch.cat([left_feat, right_feat], dim=-1)  # [B, 2 * feature_dim]

        # 4. Đưa qua tầng Fusion MLP giảm số chiều và tăng cường khả năng học
        fused_feat = self.fusion_mlp(fused_feat)  # [B, 512]

        # 5. Phân nhánh dự đoán đa nhiệm nhị phân và hồi quy tuổi
        logits = self.classification_head(fused_feat)   # [B, 1]
        age_pred = self.regression_head(fused_feat)     # [B, 1]

        return {
            "logits": logits,
            "age_pred": age_pred,
        }

    def unfreeze_backbone(self) -> None:
        """Mở khóa toàn bộ tham số của backbone để tinh chỉnh tinh (Fine-Tuning)."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def freeze_backbone(self) -> None:
        """Đóng băng toàn bộ tham số của backbone để huấn luyện nhanh các heads mới."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def __repr__(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"BinocularClassifier(backbone={self.backbone_type}, "
            f"feature_dim={self.feature_dim}, total={total:,}, trainable={trainable:,})"
        )
