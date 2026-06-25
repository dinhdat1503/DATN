"""
EfficientNet-B0 Multi-task Learning — Pure PyTorch implementation.

Tương thích với mọi phiên bản PyTorch (không cần torchvision/timm).
Kiến trúc EfficientNet-B0 chuẩn: MBConv blocks + SE attention.

Khi chạy trên Kaggle/Colab (có torchvision/timm), dùng pretrained=True
để tải ImageNet weights → hiệu quả tốt hơn đáng kể.

Cấu trúc:
    Backbone: EfficientNet-B0 (MBConv × 7 stages)
    feature_dim: 1280
    ClassificationHead: Dropout → Linear(1280 → 8)  — multi-label
    RegressionHead:     Dropout → Linear(1280 → 1)  — age prediction

Output format:
    {"logits": Tensor[B,8], "age_pred": Tensor[B,1]}
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks của EfficientNet
# ---------------------------------------------------------------------------

class ConvBnAct(nn.Module):
    """Conv + BatchNorm + SiLU activation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride, padding,
            groups=groups, bias=False
        )
        self.bn  = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block (SE attention)."""

    def __init__(self, in_ch: int, reduced_ch: int) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, reduced_ch, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(reduced_ch, in_ch, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck với SE và Stochastic Depth."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expand_ratio: int,
        kernel_size: int,
        stride: int,
        se_ratio: float = 0.25,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        mid_ch = in_ch * expand_ratio
        se_ch  = max(1, int(in_ch * se_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBnAct(in_ch, mid_ch, 1, 1, 0))

        pad = (kernel_size - 1) // 2
        layers += [
            ConvBnAct(mid_ch, mid_ch, kernel_size, stride, pad, groups=mid_ch),
            SqueezeExcite(mid_ch, se_ch),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
        ]
        self.block = nn.Sequential(*layers)
        self.drop_path_rate = drop_path_rate

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_path_rate == 0.0:
            return x
        keep = 1.0 - self.drop_path_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device) < keep
        return x * mask / keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = self._drop_path(out) + x
        return out


# ---------------------------------------------------------------------------
# EfficientNet-B0 backbone
# ---------------------------------------------------------------------------

# Cấu hình các stages của EfficientNet-B0
# (expand_ratio, out_ch, num_layers, kernel_size, stride)
_B0_CONFIG = [
    (1, 16,  1, 3, 1),   # Stage 1: MBConv1
    (6, 24,  2, 3, 2),   # Stage 2: MBConv6
    (6, 40,  2, 5, 2),   # Stage 3: MBConv6
    (6, 80,  3, 3, 2),   # Stage 4: MBConv6
    (6, 112, 3, 5, 1),   # Stage 5: MBConv6
    (6, 192, 4, 5, 2),   # Stage 6: MBConv6
    (6, 320, 1, 3, 1),   # Stage 7: MBConv6
]


class EfficientNetB0Backbone(nn.Module):
    """EfficientNet-B0 feature extractor (output: 1280-dim vector)."""

    def __init__(self, drop_path_rate: float = 0.2) -> None:
        super().__init__()

        # Stem
        self.stem = ConvBnAct(      3, 32, 3, 2, 1)

        # Tính tổng số blocks để stochastic depth
        total_blocks = sum(cfg[2] for cfg in _B0_CONFIG)
        block_idx = 0

        # MBConv stages
        stages = []
        in_ch = 32
        for expand_ratio, out_ch, num_layers, ks, stride in _B0_CONFIG:
            stage_blocks = []
            for i in range(num_layers):
                dp = drop_path_rate * block_idx / total_blocks
                stage_blocks.append(MBConv(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    expand_ratio=expand_ratio,
                    kernel_size=ks,
                    stride=stride if i == 0 else 1,
                    drop_path_rate=dp,
                ))
                block_idx += 1
            stages.append(nn.Sequential(*stage_blocks))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        # Head conv: 320 → 1280
        self.head_conv = ConvBnAct(320, 1280, 1, 1, 0)
        self.avgpool   = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.head_conv(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)  # [B, 1280]


# ---------------------------------------------------------------------------
# Multi-task model
# ---------------------------------------------------------------------------

class EfficientNetMTL(nn.Module):
    """Mô hình Học sâu Đa nhiệm (Multi-task Learning) dựa trên backbone EfficientNet-B0.
    
    Kiến trúc này giải quyết đồng thời 2 bài toán (Đa nhiệm):
    1. Phân loại bệnh lý võng mạc (Classification Head): Đầu ra là vector 8 chiều (nếu chạy multi-label) 
       hoặc 1 chiều (nếu chạy binary nhị phân sàng lọc).
    2. Dự đoán tuổi võng mạc (Regression Head): Đầu ra là 1 giá trị liên tục biểu thị tuổi đã chuẩn hóa.
    
    Các tham số khởi tạo:
        pretrained: Nếu True, tự động tải và sử dụng trọng số đã được tiền huấn luyện trên tập ImageNet-1K.
        freeze_backbone: Đóng băng toàn bộ trọng số của backbone trích xuất đặc trưng, chỉ huấn luyện các lớp phân loại/hồi quy.
        dropout_cls: Tỷ lệ ngắt kết nối ngẫu nhiên (Dropout) cho nhánh phân loại bệnh.
        dropout_reg: Tỷ lệ ngắt kết nối ngẫu nhiên (Dropout) cho nhánh dự đoán tuổi.
        num_labels: Số lượng lớp đầu ra cho nhánh phân loại (8 cho đa nhãn ODIR-5K, 1 cho phân loại nhị phân).
    """

    FEATURE_DIM = 1280  # Kích thước vector đặc trưng đầu ra của EfficientNet-B0 sau lớp Pooling

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_cls: float = 0.3,
        dropout_reg: float = 0.2,
        num_labels: int = 8,
    ) -> None:
        super().__init__()

        # --- BƯỚC 1: Khởi tạo mạng Backbone trích xuất đặc trưng ---
        self.backbone = self._build_backbone(pretrained)

        # Nếu cấu hình yêu cầu đóng băng backbone (ví dụ trong Stage 1 của Two-Stage training)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False  # Tắt tính toán gradient để cố định trọng số backbone

        # --- BƯỚC 2: Định nghĩa Nhánh Phân loại Bệnh lý (Classification Head) ---
        # Nhánh này nhận đầu vào là vector 1280 chiều, đi qua các lớp Fully Connected và LayerNorm
        # để đưa ra dự đoán xác suất mắc bệnh.
        self.classification_head = nn.Sequential(
            nn.Dropout(p=dropout_cls),                    # Tránh overfitting
            nn.Linear(self.FEATURE_DIM, 512),             # Tầng trung gian chuyển đổi từ 1280 về 512 chiều
            nn.LayerNorm(512),                            # Chuẩn hóa Layer giúp ổn định phân phối đặc trưng
            nn.SiLU(),                                    # Hàm kích hoạt phi tuyến SiLU
            nn.Dropout(p=dropout_cls),                    # Lớp dropout thứ hai tăng tính điều hòa
            nn.Linear(512, num_labels),                   # Tầng đầu ra cuối cùng (logits) cho các lớp bệnh lý
        )

        # --- BƯỚC 3: Định nghĩa Nhánh Dự đoán Tuổi (Regression Head) ---
        # Nhánh này nhận cùng đầu vào là đặc trưng 1280 chiều từ backbone để hồi quy độ tuổi võng mạc.
        self.regression_head = nn.Sequential(
            nn.Dropout(p=dropout_reg),                    # Dropout riêng cho nhánh hồi quy tuổi
            nn.Linear(self.FEATURE_DIM, 256),             # Tầng trung gian đưa từ 1280 về 256 chiều
            nn.LayerNorm(256),                            # Chuẩn hóa Layer giúp hội tụ nhanh
            nn.SiLU(),                                    # Hàm kích hoạt SiLU
            nn.Dropout(p=dropout_reg),                    # Dropout thứ hai
            nn.Linear(256, 1),                            # Đầu ra 1 chiều (tuổi võng mạc dạng chuẩn hóa Z-score)
        )

        # --- BƯỚC 4: Khởi tạo tham số (Trọng số) cho các đầu Heads ---
        self._init_heads()

    def _build_backbone(self, pretrained: bool) -> nn.Module:
        """Xây dựng mạng backbone trích xuất đặc trưng, tự động tương thích môi trường.
        
        Thứ tự ưu tiên nạp mô hình:
        1. Thư viện torchvision: Thư viện chuẩn đi kèm PyTorch, nạp nhanh và tương thích tốt.
        2. Thư viện timm (PyTorch Image Models): Thư viện SOTA cho Computer Vision.
        3. Fallback: Tự xây dựng thủ công EfficientNet-B0 (không có pretrained) nếu thiếu thư viện trên local.
        """
        # Thử nạp bằng torchvision trước
        try:
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            tv_model = efficientnet_b0(weights=weights)

            # Đóng gói lại chỉ lấy các lớp trích xuất đặc trưng và lớp pooling trung bình
            class _TVBackbone(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.features = m.features
                    self.pool = m.avgpool

                def forward(self, x):
                    # Đầu ra sau tích chập là [B, 1280, H_feat, W_feat], đi qua pooling thành [B, 1280, 1, 1]
                    # và được trải phẳng thành vector 2D [B, 1280]
                    return torch.flatten(self.pool(self.features(x)), 1)

            print("[Model] Dùng torchvision EfficientNet-B0" +
                  (" (pretrained ImageNet)" if pretrained else " (random init)"))
            return _TVBackbone(tv_model)
        except Exception:
            pass

        # Thử nạp bằng timm nếu torchvision lỗi
        try:
            import timm
            m = timm.create_model(
                'efficientnet_b0',
                pretrained=pretrained,
                num_classes=0,            # Thiết lập num_classes=0 để timm tự động lược bỏ tầng phân loại gốc
                global_pool='avg',        # Tự động thực hiện Global Average Pooling đầu ra
            )
            print("[Model] Dùng timm EfficientNet-B0" +
                  (" (pretrained ImageNet)" if pretrained else " (random init)"))
            return m
        except Exception:
            pass

        # Phương án dự phòng cuối cùng: Tự khởi tạo backbone bằng mã nguồn PyTorch thủ công trong file
        print("[Model] Dùng pure-PyTorch EfficientNet-B0 (random init — không pretrained)")
        print("[Model] Để dùng pretrained trên Kaggle/Colab: pip install timm")
        return EfficientNetB0Backbone()

    def _init_heads(self) -> None:
        """Khởi tạo trọng số cho các tầng tuyến tính mới theo phương pháp He (Kaiming).
        
        Phương pháp He giúp ổn định phương sai của các luồng kích hoạt ở đầu tiến trình huấn luyện,
        ngăn ngừa hiện tượng tiêu biến hoặc bùng nổ gradient.
        """
        for m in [self.classification_head, self.regression_head]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                    if layer.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(layer.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Luồng truyền xuôi (Forward Pass) của mạng đa nhiệm.
        
        Đầu vào:
            x: Tensor ảnh đáy mắt võng mạc, kích thước [Batch_size, 3, Chiều_cao, Chiều_rộng]
        Đầu ra:
            Dict chứa kết quả dự đoán của 2 nhiệm vụ:
            - "logits": Tensor [B, num_labels] (Chưa qua sigmoid)
            - "age_pred": Tensor [B, 1] (Tuổi chuẩn hóa Z-score)
        """
        # 1. Trích xuất đặc trưng thông qua mạng Backbone
        features = self.backbone(x)  # Kích thước đầu ra: [B, 1280]
        
        # 2. Truyền đặc trưng qua 2 nhánh song song và trả về kết quả dưới dạng Dict
        return {
            "logits":   self.classification_head(features),  # Phân loại bệnh lý võng mạc
            "age_pred": self.regression_head(features),      # Dự đoán tuổi võng mạc
        }

    def unfreeze_backbone(self) -> None:
        """Mở khóa (Unfreeze) toàn bộ tham số của backbone để huấn luyện / tinh chỉnh."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def freeze_backbone(self) -> None:
        """Đóng băng (Freeze) toàn bộ tham số của backbone để bảo toàn trọng số pretrained."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def __repr__(self) -> str:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"EfficientNetMTL("
            f"total={total:,}, trainable={trainable:,})"
        )
