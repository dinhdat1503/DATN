"""
Hàm mất mát (Loss) cho ODIR-5K Phase 1 — Phân loại nhị phân đa nhiệm.

Gồm:
1. BinaryFocalLoss: Focal Loss nhị phân, xử lý mất cân bằng lớp (32% Normal / 68% Pathological),
   hỗ trợ nhãn mềm (soft label) sinh ra do MixUp/CutMix.
2. MultiTaskLoss: Tổng hợp Loss phân loại bệnh (Focal) + λ_age × Loss hồi quy tuổi (SmoothL1).

Công thức tổng: Total = FocalLoss(logits, label) + λ_age × SmoothL1(age_pred, age_true)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BinaryFocalLoss(nn.Module):
    """Focal Loss cho phân loại nhị phân.

    Giảm trọng số của các mẫu dễ học (gamma) và cân bằng hai lớp (alpha), giúp mô hình
    tập trung vào lớp khó / hiếm. Tương thích nhãn mềm nhờ tổ hợp tuyến tính theo targets.

    Tham khảo: Lin et al. (2017), "Focal Loss for Dense Object Detection".
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-8) -> None:
        """
        Args:
            alpha: Trọng số cho lớp dương (Pathological, y=1). 1-alpha cho lớp âm (Normal, y=0).
                   Trong dự án alpha được tính tự động = N_normal / N_total (≈ 0.323).
            gamma: Hệ số focus — càng lớn càng giảm đóng góp của mẫu dễ.
            eps: Hằng số nhỏ chống log(0).
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Logit thô chưa qua sigmoid, shape [B, 1].
            targets: Nhãn thực tế (cứng 0/1 hoặc mềm trong [0,1]), shape [B, 1].

        Returns:
            Giá trị loss trung bình (scalar tensor).
        """
        probs = torch.sigmoid(logits).clamp(min=self.eps, max=1.0 - self.eps)

        # Loss cho thành phần nhãn dương (y=1) và âm (y=0)
        loss_pos = -self.alpha * ((1.0 - probs) ** self.gamma) * torch.log(probs)
        loss_neg = -(1.0 - self.alpha) * (probs ** self.gamma) * torch.log(1.0 - probs)

        # Tổ hợp tuyến tính theo nhãn — đúng cho cả nhãn mềm MixUp/CutMix
        loss = targets * loss_pos + (1.0 - targets) * loss_neg
        return loss.mean()


class MultiTaskLoss(nn.Module):
    """Loss đa nhiệm = Phân loại bệnh (Focal) + λ_age × Hồi quy tuổi (SmoothL1).

    Nhánh hồi quy tuổi đóng vai trò nhiệm vụ phụ trợ (auxiliary) giúp backbone học thêm
    biểu diễn sinh học của cấu trúc võng mạc, gián tiếp hỗ trợ phân loại bệnh.
    """

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        lam_age: float = 0.05,
    ) -> None:
        """
        Args:
            focal_alpha: Alpha của Binary Focal Loss.
            focal_gamma: Gamma của Binary Focal Loss.
            lam_age: Hệ số trọng số cho loss tuổi (mặc định 0.05 — hằng số khóa cứng của dự án).
        """
        super().__init__()
        self.lam_age = lam_age
        self.cls_loss_fn = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        # SmoothL1 robust với outlier hơn MSE — phù hợp hồi quy tuổi
        self.reg_loss_fn = nn.SmoothL1Loss(reduction="mean", beta=1.0)
        print(f"[Loss] MultiTaskLoss = Focal(alpha={focal_alpha:.4f}, gamma={focal_gamma}) "
              f"+ {lam_age} × SmoothL1(age)")

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        age_pred: torch.Tensor,
        age_true: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Tính tổng loss đa nhiệm.

        Args:
            logits: [B, 1] logit phân loại.
            labels: [B, 1] nhãn nhị phân (0/1, có thể mềm).
            age_pred: [B, 1] tuổi dự đoán (chuẩn hóa Z-score).
            age_true: [B, 1] tuổi thực (chuẩn hóa Z-score).

        Returns:
            (total_loss, detail) với detail là dict chứa giá trị từng thành phần (float).
        """
        cls_loss = self.cls_loss_fn(logits, labels)
        reg_loss = self.reg_loss_fn(age_pred, age_true)
        total = cls_loss + self.lam_age * reg_loss

        detail = {
            "loss_total": float(total.item()),
            "loss_cls": float(cls_loss.item()),
            "loss_reg": float(reg_loss.item()),
        }
        return total, detail

    def __repr__(self) -> str:
        return f"MultiTaskLoss(focal + {self.lam_age}×SmoothL1)"
