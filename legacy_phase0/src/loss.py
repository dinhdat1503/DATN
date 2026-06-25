"""
Multi-task Loss và các hàm Loss nâng cao cho ODIR-5K.

File này định nghĩa các hàm loss dùng cho việc huấn luyện đa nhiệm (phân loại + tuổi):
1. BinaryFocalLoss: Xử lý mất cân bằng lớp ở mức độ nhị phân (Normal vs Pathological)
2. AsymmetricLoss (ASL): Xử lý nhãn đa nhiệm bất đối xứng
3. MultiTaskLoss: Lớp tổng hợp tính joint loss kết hợp phân loại và hồi quy tuổi y sinh.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss cho phân loại nhị phân (Binary Focal Loss).
    Giảm trọng số của các mẫu dễ học, tập trung học các mẫu khó (mẫu bệnh lý hiếm gặp).
    
    Hỗ trợ nhãn mềm (soft labels) phát sinh do quá trình MixUp hoặc CutMix.
    Tham khảo: Lin, T. Y. et al. (2017). Focal Loss for Dense Object Detection.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-8) -> None:
        """
        Khởi tạo Focal Loss.

        Args:
            alpha: Trọng số cân bằng hai lớp (mặc định 0.25 cho lớp thiểu số bệnh lý)
            gamma: Tham số Focus điều khiển mức độ phạt mẫu dễ học (mặc định 2.0)
            eps: Giá trị nhỏ để chống tràn số / log(0)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Tính toán loss.

        Args:
            logits: Đầu ra dự đoán chưa kích hoạt của mô hình [B, 1]
            targets: Nhãn thực tế (độ chính xác đơn hoặc mềm) [B, 1]
        """
        probs = torch.sigmoid(logits)
        
        # Tránh lỗi tràn số hoặc log(0) bằng cách kẹp giá trị
        probs = probs.clamp(min=self.eps, max=1.0 - self.eps)

        # Tính toán loss cho hai trường hợp nhãn dương (1) và âm (0)
        loss_pos = -self.alpha * ((1.0 - probs) ** self.gamma) * torch.log(probs)
        loss_neg = -(1.0 - self.alpha) * (probs ** self.gamma) * torch.log(1.0 - probs)

        # Kết hợp tuyến tính (tương thích cả nhãn mềm MixUp/CutMix)
        loss = targets * loss_pos + (1.0 - targets) * loss_neg

        return loss.mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) cho multi-label classification.
    Phạt các mẫu âm tính sai sót một cách bất đối xứng, giúp giải quyết bài toán class imbalance.
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
    """
    Joint Loss liên kết nhiệm vụ Phân loại bệnh lý võng mạc và Hồi quy độ tuổi.

    Công thức: Total_Loss = Loss_phân_loại + lam_age × Loss_tuổi (SmoothL1)
    """

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        lam: float = 0.1,
        loss_type: str = "bce",
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        device: str | torch.device = "cpu",
    ) -> None:
        """
        Khởi tạo lớp Loss đa nhiệm.

        Args:
            pos_weight: Trọng số dương tính cho BCEWithLogitsLoss
            lam: Hệ số phạt của nhánh tuổi (mặc định 0.05 - 0.1)
            loss_type: Loại hàm loss ('bce', 'asl' hoặc 'focal')
            focal_alpha: Trọng số alpha trong Focal Loss
            focal_gamma: Tham số gamma trong Focal Loss
        """
        super().__init__()
        self.lam = lam
        self.loss_type = loss_type.lower().strip()

        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.to(device))
        else:
            self.pos_weight = None

        # Khởi tạo hàm loss phân loại tương ứng cấu hình
        if self.loss_type == "focal":
            self.cls_loss_fn = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            print(f"[Loss] Sử dụng Binary Focal Loss: alpha={focal_alpha}, gamma={focal_gamma}")
        elif self.loss_type == "asl":
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

        # Smooth L1 được sử dụng cho tuổi vì nó robust hơn MSE và ít nhạy cảm với các outliers
        self.smooth_l1 = nn.SmoothL1Loss(reduction="mean", beta=1.0)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        age_pred: torch.Tensor,
        age_true: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Tính toán joint loss đa nhiệm.
        """
        # Cập nhật pos_weight nếu có sự dịch chuyển device (với BCE)
        if self.loss_type == "bce" and self.pos_weight is not None and self.pos_weight.device != logits.device:
            self.cls_loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(logits.device),
                reduction="mean",
            )

        # Tính toán chi tiết các loss thành phần
        cls_loss = self.cls_loss_fn(logits, labels)
        reg_loss = self.smooth_l1(age_pred, age_true)

        # Tổng hợp loss có trọng số
        total_loss = cls_loss + self.lam * reg_loss

        detail = {
            "loss_total": total_loss.item(),
            "loss_cls": cls_loss.item(),
            "loss_reg": reg_loss.item(),
            "lam": self.lam,
        }

        return total_loss, detail

    def to(self, device):
        """Đảm bảo các buffer con được chuyển device chính xác."""
        super().to(device)
        if self.loss_type == "bce" and self.pos_weight is not None:
            self.cls_loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(device),
                reduction="mean",
            )
        return self

    def __repr__(self) -> str:
        if self.loss_type == "focal":
            return f"MultiTaskLoss(loss_type=focal, lam={self.lam})"
        elif self.loss_type == "asl":
            return f"MultiTaskLoss(loss_type=asl, lam={self.lam})"
        has_pw = self.pos_weight is not None
        return f"MultiTaskLoss(loss_type=bce, lam={self.lam}, pos_weight={'yes' if has_pw else 'no'})"
