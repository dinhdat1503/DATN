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
    """Mô hình Học sâu Đa nhiệm (Multi-task Learning) sử dụng backbone Swin Transformer.
    
    Khác biệt cốt lõi của Swin Transformer so với CNN truyền thống:
    - Thay vì dùng tích chập cục bộ (convolution), Swin sử dụng cơ chế Tự chú ý (Self-Attention)
      trên các cửa sổ dịch chuyển (Shifted Windows). Điều này giúp mô hình nhận diện và kết nối các tổn thương
      võng mạc đáy mắt nằm phân rải rác trên toàn bộ bức ảnh võng mạc một cách hiệu quả hơn.
    - Cấu trúc đa nhiệm chia sẻ biểu diễn đặc trưng (Shared Representation) được nạp từ trọng số pre-trained ImageNet,
      sau đó chuyển tiếp qua 2 nhánh Linear Heads tương tự EfficientNet.

    Các tham số khởi tạo:
        pretrained:      Nạp trọng số ImageNet-1K pre-trained nếu thư viện timm sẵn có.
        img_size:        Kích thước ảnh đầu vào (thường là 224 hoặc 384).
        variant:         Biến thể kích thước của Swin ('tiny' | 'small' | 'base'). Tiny phù hợp cho VRAM 16GB.
        freeze_backbone: Đóng băng backbone, chỉ huấn luyện các lớp tuyến tính mới.
        dropout_cls:     Tỷ lệ ngắt neuron ngẫu nhiên cho nhánh phân loại bệnh.
        dropout_reg:     Tỷ lệ ngắt neuron ngẫu nhiên cho nhánh dự đoán tuổi võng mạc.
        num_labels:      Số lượng nhãn phân loại đầu ra (8 cho ODIR-5K, 1 cho nhị phân Normal/Pathological).
    """

    # Kích thước đặc trưng đầu ra ứng với mỗi phiên bản của timm Swin Transformer
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
        
        # --- BƯỚC 1: Xây dựng mạng Backbone Swin Transformer ---
        self.backbone, self.feature_dim = self._build_backbone(
            pretrained, img_size, variant
        )

        # Đóng băng trọng số của Swin Transformer nếu cấu hình yêu cầu (Stage 1)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # --- BƯỚC 2: Định nghĩa Nhánh Phân loại Bệnh lý (Classification Head) ---
        # Nhánh này nhận vector đặc trưng (768 chiều cho Swin-Tiny) và chiếu về không gian 8 nhãn bệnh lý
        # hoặc 1 nhãn nhị phân.
        self.classification_head = nn.Sequential(
            nn.Dropout(p=dropout_cls),                    # Ngăn chặn overfitting
            nn.Linear(self.feature_dim, 512),             # Tầng FC 1
            nn.LayerNorm(512),                            # Chuẩn hóa phân phối kích hoạt
            nn.SiLU(),                                    # Kích hoạt phi tuyến SiLU
            nn.Dropout(p=dropout_cls),                    # Dropout 2
            nn.Linear(512, num_labels),                   # Logits đầu ra phân loại
        )
        
        # --- BƯỚC 3: Định nghĩa Nhánh Dự đoán Tuổi (Regression Head) ---
        # Nhánh hồi quy tuổi đáy mắt võng mạc, sử dụng chung đặc trưng trích xuất từ Swin backbone.
        self.regression_head = nn.Sequential(
            nn.Dropout(p=dropout_reg),                    # Dropout riêng cho hồi quy
            nn.Linear(self.feature_dim, 256),             # Tầng FC 1
            nn.LayerNorm(256),                            # Chuẩn hóa Layer
            nn.SiLU(),                                    # Kích hoạt SiLU
            nn.Dropout(p=dropout_reg),                    # Dropout 2
            nn.Linear(256, 1),                            # Đầu ra 1 chiều (tuổi chuẩn hóa Z-score)
        )
        
        # Khởi tạo ngẫu nhiên có kiểm soát cho các MLP heads
        self._init_heads()

    def _build_backbone(
        self,
        pretrained: bool,
        img_size: int,
        variant: str,
    ) -> tuple[nn.Module, int]:
        """Xây dựng mạng backbone Swin Transformer, hỗ trợ nội suy nhúng vị trí y sinh.
        
        Trong timm, khi thay đổi kích thước ảnh từ 224 lên 384, vị trí nhúng tương đối (Relative Position Embeddings)
        bị lệch. Chúng ta truyền tham số `img_size=img_size` trực tiếp vào hàm `timm.create_model`
        để kích hoạt cơ chế nội suy song tuyến tính giúp nạp thành công mô hình ở độ phân giải 384x384.
        """

        # Bản đồ ánh xạ tên mô hình trong thư viện timm
        model_names = {
            ("tiny",  224): "swin_tiny_patch4_window7_224",
            ("tiny",  384): "swin_tiny_patch4_window7_224",   # Sẽ kích hoạt nội suy kích thước ảnh ở dưới
            ("small", 224): "swin_small_patch4_window7_224",
            ("small", 384): "swin_small_patch4_window7_224",  # Sẽ kích hoạt nội suy kích thước ảnh ở dưới
            ("base",  224): "swin_base_patch4_window7_224",
            ("base",  384): "swin_base_patch4_window12_384",  # Bản mẫu chuẩn ở độ phân giải 384
        }
        timm_name = model_names.get((variant, img_size), "swin_tiny_patch4_window7_224")
        feature_dim = self.FEATURE_DIMS.get(variant, 768)

        # Tiny và Small ở 384x384 cần ghi đè img_size để timm nội suy ma trận Relative Position Bias
        needs_img_size_override = (variant in ("tiny", "small") and img_size != 224)

        try:
            import timm
            if needs_img_size_override:
                backbone = timm.create_model(
                    timm_name,
                    pretrained=pretrained,
                    num_classes=0,            # Bỏ lớp phân loại gốc 1000 lớp của ImageNet
                    global_pool="avg",        # Pooling trung bình đầu ra
                    img_size=img_size,        # Ép timm tự động thực hiện nội suy nhúng vị trí tương đối
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

        # Trường hợp dự phòng (Fallback): Sử dụng mô hình ViT mini tự tạo để chạy test local không bị crash
        print(
            f"[Model] timm không khả dụng → dùng Mini-ViT fallback (local test only)"
        )
        print(f"[Model] Trên Kaggle/Colab: pip install timm để dùng Swin-{variant.capitalize()}")
        backbone, feature_dim = _build_mini_vit(img_size, feature_dim)
        return backbone, feature_dim

    def _init_heads(self) -> None:
        """He initialization cho các tầng tuyến tính mới."""
        for m in [self.classification_head, self.regression_head]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                    if layer.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(layer.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass của mô hình SwinMTL.
        
        Đầu vào:
            x: Tensor ảnh đáy mắt kích thước [B, 3, H, W]
        Đầu ra:
            Dict chứa logits phân loại bệnh lý ("logits") và dự đoán tuổi ("age_pred")
        """
        # 1. Trích xuất đặc trưng từ Swin Transformer backbone
        features = self.backbone(x)
        
        # 2. Đảm bảo đặc trưng có dạng 2D [Batch_size, Feature_dim]
        # (Nếu đầu ra chưa được trải phẳng hoặc pool)
        if features.ndim > 2:
            features = features.mean(dim=list(range(2, features.ndim)))
            
        # 3. Đưa qua 2 nhánh đầu ra đa nhiệm song song
        return {
            "logits":   self.classification_head(features),  # Phân loại bệnh lý
            "age_pred": self.regression_head(features),      # Dự đoán tuổi
        }

    def unfreeze_backbone(self) -> None:
        """Mở khóa toàn bộ tham số của Swin Transformer backbone để tinh chỉnh."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def freeze_backbone(self) -> None:
        """Đóng băng toàn bộ tham số của Swin Transformer backbone."""
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
