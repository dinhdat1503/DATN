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


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (ASL) cho multi-label classification.
    Tham khảo paper: "Asymmetric Loss for Multi-Label Classification" (ICCV 2021)
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = True,
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: [B, C] - logits, y: [B, C] - targets (0.0 hoặc 1.0)
        xs_pos = torch.sigmoid(x)
        xs_neg = 1.0 - xs_pos

        # Asymmetric Clipping cho lớp âm tính
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Tính toán loss nhị phân cơ bản
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Áp dụng Asymmetric Focusing (Trọng số Focal động)
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            loss_pos_weights = (1.0 - xs_pos) ** self.gamma_pos
            loss_neg_weights = (1.0 - xs_neg) ** self.gamma_neg
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss = los_pos * loss_pos_weights + los_neg * loss_neg_weights

        return -loss.mean()


class MultiTaskLoss(nn.Module):
    """Joint loss cho multi-task learning: Classification + Regression.

    Args:
        pos_weight: FloatTensor [8] — trọng số cho BCEWithLogitsLoss (chỉ dùng khi loss_type='bce').
        lam: Hệ số cân bằng cho regression loss (mặc định 0.1).
        loss_type: 'bce' hoặc 'asl'.
        gamma_neg, gamma_pos, clip: Tham số cho Asymmetric Loss.
        device: Device để chuyển pos_weight lên.
    """

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        lam: float = 0.1,
        loss_type: str = "bce",
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.lam = lam
        self.loss_type = loss_type.lower().strip()

        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.to(device))
        else:
            self.pos_weight = None

        if self.loss_type == "asl":
            self.cls_loss_fn = AsymmetricLoss(
                gamma_neg=gamma_neg,
                gamma_pos=gamma_pos,
                clip=clip,
            )
            print(f"[Loss] Sử dụng Asymmetric Loss (ASL): gamma_neg={gamma_neg}, gamma_pos={gamma_pos}, clip={clip}")
        else:
            self.cls_loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight,
                reduction="mean",
            )
            print("[Loss] Sử dụng Binary Cross Entropy (BCE)")

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
        # Cập nhật pos_weight nếu cần (khi device thay đổi) và loss_type là bce
        if self.loss_type == "bce" and self.pos_weight is not None and self.pos_weight.device != logits.device:
            self.cls_loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(logits.device),
                reduction="mean",
            )

        cls_loss = self.cls_loss_fn(logits, labels)
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
        if self.loss_type == "bce" and self.pos_weight is not None:
            self.cls_loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(device),
                reduction="mean",
            )
        return self

    def __repr__(self) -> str:
        if self.loss_type == "asl":
            return f"MultiTaskLoss(loss_type=asl, lam={self.lam})"
        has_pw = self.pos_weight is not None
        return f"MultiTaskLoss(loss_type=bce, lam={self.lam}, pos_weight={'yes' if has_pw else 'no'})"
