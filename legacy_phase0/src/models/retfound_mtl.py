"""
RETFound (Medical Foundation Model for Ophthalmology) Multi-task Learning cho ODIR-5K.

Kiến trúc: ViT-Large (patch_size=16, embed_dim=1024, depth=24) với 2 head đầu ra:
    - Classification Head: Linear(1024 → 8)  — multi-label 8 bệnh
    - Regression Head:     Linear(1024 → 1)  — dự đoán tuổi

Sử dụng trọng số pre-trained y sinh học chuyên biệt cho võng mạc mắt (RETFound).
"""
from __future__ import annotations

import math
from pathlib import Path
from functools import partial
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────
# Hàm nội suy positional embedding khi kích thước ảnh khác 224
# ──────────────────────────────────────────────────────────────
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        
        if orig_size != new_size:
            print(f"[Model] Nội suy pos_embed từ {orig_size}x{orig_size} thành {new_size}x{new_size}")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


# ──────────────────────────────────────────────────────────────
# Khai báo mô hình ViT của MAE / RETFound kế thừa từ timm
# ──────────────────────────────────────────────────────────────
try:
    import timm
    from timm.models.vision_transformer import VisionTransformer as TimmVisionTransformer

    class VisionTransformer(TimmVisionTransformer):
        """ Vision Transformer hỗ trợ Global Average Pooling (dùng cho RETFound) """
        def __init__(self, custom_global_pool=False, **kwargs):
            super().__init__(**kwargs)
            self.custom_global_pool = custom_global_pool
            if self.custom_global_pool:
                norm_layer = kwargs.get('norm_layer', partial(nn.LayerNorm, eps=1e-6))
                embed_dim = kwargs['embed_dim']
                self.fc_norm = norm_layer(embed_dim)
                # Xóa lớp norm cũ của timm để tránh nhầm lẫn
                if hasattr(self, 'norm'):
                    del self.norm

        def forward_features(self, x, *args, **kwargs):
            B = x.shape[0]
            x = self.patch_embed(x)

            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for blk in self.blocks:
                x = blk(x)

            if self.custom_global_pool:
                x = x[:, 1:, :].mean(dim=1)  # global pool không lấy cls token
                outcome = self.fc_norm(x)
            else:
                x = self.norm(x)
                outcome = x[:, 0]

            return outcome

    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    # Định nghĩa class rỗng làm placeholder khi không có timm (local testing)
    class VisionTransformer(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.num_features = 1024
            self.patch_embed = nn.Identity()
        def forward(self, x):
            return torch.zeros(x.shape[0], 1024, device=x.device)


class RETFoundMTL(nn.Module):
    """Mô hình RETFound (ViT-Large) Multi-task Learning.

    Args:
        pretrained:      Có tải trọng số pre-trained hay không.
        pretrained_path: Đường dẫn tới file trọng số RETFound (ví dụ: 'pretrained_weights/RETFound_cfp_weights.pth').
        img_size:        Kích thước ảnh đầu vào (mặc định 224).
        freeze_backbone: Đóng băng backbone, chỉ train heads.
        dropout_cls:     Dropout tỉ lệ cho classification head.
        dropout_reg:     Dropout tỉ lệ cho regression head.
        num_labels:      Số nhãn bệnh (8 cho ODIR-5K).
    """
    def __init__(
        self,
        pretrained: bool = True,
        pretrained_path: str | None = None,
        img_size: int = 224,
        freeze_backbone: bool = False,
        dropout_cls: float = 0.3,
        dropout_reg: float = 0.2,
        num_labels: int = 8,
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.feature_dim = 1024  # ViT-Large có embedding dimension là 1024

        if HAS_TIMM:
            self.backbone = VisionTransformer(
                img_size=img_size,
                patch_size=16,
                embed_dim=1024,
                depth=24,
                num_heads=16,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                custom_global_pool=True,
                num_classes=0,  # trả về đặc trưng trực tiếp
            )
            print(f"[Model] Đã khởi tạo cấu trúc ViT-Large (RETFound) với img_size={img_size}")
            
            # Load trọng số pre-trained nếu có yêu cầu
            if pretrained and pretrained_path:
                self.load_pretrained_weights(pretrained_path)
            elif pretrained:
                # Tìm mặc định
                default_path = "pretrained_weights/RETFound_cfp_weights.pth"
                self.load_pretrained_weights(default_path)
        else:
            print("[Model] ⚠️ timm không khả dụng! Tạo Dummy Backbone cho local test.")
            self.backbone = VisionTransformer()

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

    def load_pretrained_weights(self, pretrained_path: str):
        path = Path(pretrained_path)
        if not path.exists():
            print(f"[Model] ⚠️ Không tìm thấy tệp trọng số RETFound tại: {path}")
            print(f"        Hãy tải về và đặt tệp tin tại đó, hoặc truyền đúng đường dẫn qua cấu hình.")
            return False
            
        print(f"[Model] Đang tải trọng số RETFound từ {path}...")
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                checkpoint_model = checkpoint["model"]
            else:
                checkpoint_model = checkpoint

            # Loại bỏ heads không khớp
            state_dict = self.backbone.state_dict()
            for k in ["head.weight", "head.bias", "fc.weight", "fc.bias"]:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"  [Model] Loại bỏ key {k} khỏi checkpoint")
                    del checkpoint_model[k]

            # Nội suy positional embedding
            interpolate_pos_embed(self.backbone, checkpoint_model)

            # Load vào model
            msg = self.backbone.load_state_dict(checkpoint_model, strict=False)
            print(f"  [Model] Nạp trọng số RETFound hoàn tất: {msg}")
            return True
        except Exception as e:
            print(f"  [Model] ❌ Không thể nạp trọng số RETFound: {e}")
            return False

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
        features = self.backbone.forward_features(x)
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
            f"RETFoundMTL(img_size={self.img_size}, feature_dim={self.feature_dim}, "
            f"total={total:,}, trainable={trainable:,})"
        )
