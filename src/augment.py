"""
Tăng cường dữ liệu ĐỒNG BỘ song nhãn (Binocular MixUp & CutMix) cho ODIR-5K Phase 1.

Đây là các lớp Collator dùng làm `collate_fn` của DataLoader train. Điểm mấu chốt:
phép trộn (MixUp/CutMix) phải áp dụng **giống hệt nhau** lên cả mắt trái và mắt phải của
cùng một bệnh nhân — cùng tỷ lệ lambda, cùng hoán vị, cùng vùng cắt — để bảo toàn tính
nhất quán sinh học giữa hai mắt.

Nhãn (label) và tuổi (age) được trộn mềm theo lambda → sinh nhãn mềm (soft label) mà
BinaryFocalLoss hỗ trợ. Cờ thiếu mắt được kết hợp bằng phép AND (chỉ thực sự thiếu nếu
cả hai mẫu đem trộn đều thiếu mắt đó).
"""

from __future__ import annotations

import numpy as np
import torch


def _stack_batch(batch: list[dict]) -> dict:
    """Gom các trường của một batch thành tensor lô."""
    return {
        "left_image": torch.stack([s["left_image"] for s in batch]),
        "right_image": torch.stack([s["right_image"] for s in batch]),
        "left_missing": torch.stack([s["left_missing"] for s in batch]),
        "right_missing": torch.stack([s["right_missing"] for s in batch]),
        "label": torch.stack([s["label"] for s in batch]),
        "age": torch.stack([s["age"] for s in batch]),
        "patient_id": [s["patient_id"] for s in batch],
    }


class BinocularMixUp:
    """MixUp đồng bộ: trộn tuyến tính cặp mắt của bệnh nhân A với bệnh nhân B theo cùng lambda."""

    def __init__(self, alpha: float = 0.4, prob: float = 0.5, seed: int | None = None) -> None:
        self.alpha = alpha
        self.prob = prob
        self.rng = np.random.default_rng(seed)

    def __call__(self, data: dict) -> dict:
        if self.rng.random() > self.prob:
            return data

        lam = float(self.rng.beta(self.alpha, self.alpha))
        lam = max(lam, 1.0 - lam)  # giữ ảnh gốc chiếm đa số → ổn định

        n = data["label"].size(0)
        perm = torch.from_numpy(self.rng.permutation(n))

        data["left_image"] = lam * data["left_image"] + (1.0 - lam) * data["left_image"][perm]
        data["right_image"] = lam * data["right_image"] + (1.0 - lam) * data["right_image"][perm]
        data["label"] = lam * data["label"] + (1.0 - lam) * data["label"][perm]
        data["age"] = lam * data["age"] + (1.0 - lam) * data["age"][perm]
        data["left_missing"] = data["left_missing"] & data["left_missing"][perm]
        data["right_missing"] = data["right_missing"] & data["right_missing"][perm]
        return data


class BinocularCutMix:
    """CutMix đồng bộ: dán cùng một vùng chữ nhật từ bệnh nhân B sang A trên cả hai mắt."""

    def __init__(self, alpha: float = 1.0, prob: float = 0.5, seed: int | None = None) -> None:
        self.alpha = alpha
        self.prob = prob
        self.rng = np.random.default_rng(seed)

    def _rand_bbox(self, w: int, h: int, lam: float) -> tuple[int, int, int, int]:
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w, cut_h = int(w * cut_ratio), int(h * cut_ratio)
        cx, cy = int(self.rng.integers(w)), int(self.rng.integers(h))
        x1, y1 = max(0, cx - cut_w // 2), max(0, cy - cut_h // 2)
        x2, y2 = min(w, cx + cut_w // 2), min(h, cy + cut_h // 2)
        return x1, y1, x2, y2

    def __call__(self, data: dict) -> dict:
        if self.rng.random() > self.prob:
            return data

        lam_init = float(self.rng.beta(self.alpha, self.alpha))
        _, _, h, w = data["left_image"].shape
        x1, y1, x2, y2 = self._rand_bbox(w, h, lam_init)
        lam = 1.0 - (x2 - x1) * (y2 - y1) / (w * h)  # lambda thực theo diện tích cắt

        n = data["label"].size(0)
        perm = torch.from_numpy(self.rng.permutation(n))

        data["left_image"][:, :, y1:y2, x1:x2] = data["left_image"][perm, :, y1:y2, x1:x2]
        data["right_image"][:, :, y1:y2, x1:x2] = data["right_image"][perm, :, y1:y2, x1:x2]
        data["label"] = lam * data["label"] + (1.0 - lam) * data["label"][perm]
        data["age"] = lam * data["age"] + (1.0 - lam) * data["age"][perm]
        data["left_missing"] = data["left_missing"] & data["left_missing"][perm]
        data["right_missing"] = data["right_missing"] & data["right_missing"][perm]
        return data


class BinocularAugmentCollator:
    """Collator gom lô + áp dụng MixUp/CutMix đồng bộ theo cấu hình.

    Nếu bật cả hai → mỗi batch tung đồng xu 50/50 chọn MixUp hoặc CutMix.
    """

    def __init__(
        self,
        use_mixup: bool = False,
        use_cutmix: bool = False,
        mixup_alpha: float = 0.4,
        mixup_prob: float = 0.5,
        cutmix_alpha: float = 1.0,
        cutmix_prob: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.mixup = BinocularMixUp(mixup_alpha, mixup_prob, seed) if use_mixup else None
        self.cutmix = BinocularCutMix(cutmix_alpha, cutmix_prob, seed) if use_cutmix else None
        self.rng = np.random.default_rng(seed)

    def __call__(self, batch: list[dict]) -> dict:
        data = _stack_batch(batch)

        if self.use_mixup and self.use_cutmix:
            return self.mixup(data) if self.rng.random() < 0.5 else self.cutmix(data)
        if self.use_mixup:
            return self.mixup(data)
        if self.use_cutmix:
            return self.cutmix(data)
        return data

    def __repr__(self) -> str:
        return f"BinocularAugmentCollator(mixup={self.use_mixup}, cutmix={self.use_cutmix})"
