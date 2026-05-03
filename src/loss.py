"""
Multi-task Loss cho ODIR-5K.

Joint Loss = BCE(pos_weight) + λ × SmoothL1(age)

Trong đó:
    BCE:       BCEWithLogitsLoss có pos_weight để xử lý class imbalance
    SmoothL1:  Huber loss cho tuổi (robust hơn MAE với outliers)
    λ (lambda): Hệ số cân bằng 2 task (mặc định 0.1)

Lý do dùng SmoothL1 thay MAE:
    - MAE gradient = ±1 (không liên tục tại 0) → không ổn định khi training
    - SmoothL1 = MSE khi |x|<1, MAE khi |x|≥1 → ổn định + robust

Vì tuổi đã được chuẩn hóa Z-score (mean~0, std~1), SmoothL1 loss thường
nằm trong khoảng 0.2–1.0, trong khi BCE nằm trong 0.3–1.5.
Với λ=0.1, đóng góp của regression task ~10% tổng loss.

Usage:
    from src.loss import MultiTaskLoss
    criterion = MultiTaskLoss(pos_weight=pos_weight_tensor, lam=0.1)
    loss, detail = criterion(logits, labels, age_pred, age_true)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """Joint loss cho multi-task learning: Classification + Regression.

    Args:
        pos_weight: FloatTensor [8] — trọng số cho BCEWithLogitsLoss.
                    pos_weight[i] = neg_count[i] / pos_count[i]
        lam: Hệ số cân bằng cho regression loss (mặc định 0.1).
        device: Device để chuyển pos_weight lên.
    """

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        lam: float = 0.1,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.lam = lam

        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.to(device))
        else:
            self.pos_weight = None

        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight,
            reduction="mean",
        )
        self.smooth_l1 = nn.SmoothL1Loss(reduction="mean", beta=1.0)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        age_pred: torch.Tensor,
        age_true: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Tính joint loss.

        Args:
            logits:   FloatTensor [B, 8] — raw logits từ model
            labels:   FloatTensor [B, 8] — ground truth labels (0/1 hoặc soft)
            age_pred: FloatTensor [B, 1] — predicted normalized age
            age_true: FloatTensor [B, 1] — ground truth normalized age

        Returns:
            total_loss: Scalar tensor (differentiable)
            detail: Dict chứa breakdown từng loss component
        """
        # Cập nhật pos_weight nếu cần (khi device thay đổi)
        if self.pos_weight is not None and self.pos_weight.device != logits.device:
            self.bce = nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(logits.device),
                reduction="mean",
            )

        cls_loss = self.bce(logits, labels)
        reg_loss = self.smooth_l1(age_pred, age_true)

        total_loss = cls_loss + self.lam * reg_loss

        detail = {
            "loss_total":  total_loss.item(),
            "loss_cls":    cls_loss.item(),
            "loss_reg":    reg_loss.item(),
            "lam":         self.lam,
        }

        return total_loss, detail

    def to(self, device):
        """Override to() để cập nhật BCELoss khi chuyển device."""
        super().to(device)
        if self.pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(device),
                reduction="mean",
            )
        return self

    def __repr__(self) -> str:
        has_pw = self.pos_weight is not None
        return f"MultiTaskLoss(lam={self.lam}, pos_weight={'yes' if has_pw else 'no'})"
