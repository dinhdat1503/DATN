"""
Tăng cường dữ liệu cặp mắt đồng bộ (Binocular MixUp & CutMix Collators) cho ODIR-5K.

File này định nghĩa các lớp Collator dùng làm `collate_fn` trong PyTorch DataLoader.
Các kỹ thuật MixUp và CutMix được thực hiện đồng bộ trên cả hai mắt (mắt trái và mắt phải)
của cùng một bệnh nhân, sử dụng cùng tỷ lệ trộn (lambda) và cùng hoán vị (permutation).
Đồng thời, cờ thiếu mắt cũng được kết hợp đồng bộ bằng phép toán logic AND.
"""

from __future__ import annotations

import numpy as np
import torch


class BinocularMixUpCollator:
    """
    Collator áp dụng MixUp đồng bộ lên cặp mắt của bệnh nhân.
    Trộn ảnh mắt trái của A với mắt trái của B, và mắt phải của A với mắt phải của B
    sử dụng cùng tỷ lệ lambda được lấy mẫu từ phân phối Beta.
    """

    def __init__(
        self,
        alpha: float = 0.4,
        prob: float = 0.5,
        seed: int | None = None,
    ) -> None:
        """
        Khởi tạo MixUp Collator.

        Args:
            alpha: Tham số của phân phối Beta (Beta(alpha, alpha)).
            prob: Xác suất áp dụng MixUp trên mỗi lô dữ liệu (batch).
            seed: Seed ngẫu nhiên để tái tạo kết quả.
        """
        if alpha <= 0:
            raise ValueError(f"alpha phải lớn hơn 0, nhận được: {alpha}")
        if not (0.0 < prob <= 1.0):
            raise ValueError(f"prob phải nằm trong khoảng (0, 1], nhận được: {prob}")

        self.alpha = alpha
        self.prob = prob
        self.rng = np.random.default_rng(seed)

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor | list]:
        """
        Gom lô và trộn MixUp đồng bộ.
        """
        # --- BƯỚC 1: Stack các trường dữ liệu thành tensor lô ---
        left_images = torch.stack([s["left_image"] for s in batch])
        right_images = torch.stack([s["right_image"] for s in batch])
        left_missing = torch.stack([s["left_missing"] for s in batch])
        right_missing = torch.stack([s["right_missing"] for s in batch])
        labels = torch.stack([s["label"] for s in batch])
        ages = torch.stack([s["age"] for s in batch])
        patient_ids = [s["patient_id"] for s in batch]

        # --- BƯỚC 2: Quyết định có áp dụng MixUp không ---
        if self.rng.random() > self.prob:
            return {
                "left_image": left_images,
                "right_image": right_images,
                "left_missing": left_missing,
                "right_missing": right_missing,
                "label": labels,
                "age": ages,
                "patient_id": patient_ids,
                "mixed": False,
                "mixup_lambda": torch.ones(1),
            }

        # --- BƯỚC 3: Lấy mẫu tỷ lệ trộn lambda từ Beta(alpha, alpha) ---
        lam = float(self.rng.beta(self.alpha, self.alpha))
        # Đảm bảo lambda >= 0.5 để ảnh gốc chiếm đa số (tăng tính ổn định)
        lam = max(lam, 1.0 - lam)

        # --- BƯỚC 4: Tạo hoán vị ngẫu nhiên cho lô ---
        n = len(batch)
        perm = torch.from_numpy(self.rng.permutation(n))

        # --- BƯỚC 5: Thực hiện trộn ảnh và thông tin đồng bộ ---
        left_images_mix = lam * left_images + (1.0 - lam) * left_images[perm]
        right_images_mix = lam * right_images + (1.0 - lam) * right_images[perm]

        # Trộn nhãn mềm (soft labels)
        labels_mix = lam * labels + (1.0 - lam) * labels[perm]

        # Trộn tuổi võng mạc
        ages_mix = lam * ages + (1.0 - lam) * ages[perm]

        # Kết hợp cờ thiếu mắt: Chỉ thực sự khuyết thiếu nếu CẢ HAI mẫu đem trộn đều thiếu mắt đó
        left_missing_mix = left_missing & left_missing[perm]
        right_missing_mix = right_missing & right_missing[perm]

        return {
            "left_image": left_images_mix,
            "right_image": right_images_mix,
            "left_missing": left_missing_mix,
            "right_missing": right_missing_mix,
            "label": labels_mix,
            "age": ages_mix,
            "patient_id": patient_ids,
            "mixed": True,
            "mixup_lambda": torch.tensor(lam),
        }


class BinocularCutMixCollator:
    """
    Collator áp dụng CutMix đồng bộ lên cặp mắt của bệnh nhân.
    Cắt một vùng chữ nhật ngẫu nhiên từ cặp mắt của bệnh nhân B và dán đè lên cặp mắt của bệnh nhân A
    sử dụng cùng một tọa độ vùng cắt và tỷ lệ lambda tương ứng.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        prob: float = 0.5,
        seed: int | None = None,
    ) -> None:
        """
        Khởi tạo CutMix Collator.

        Args:
            alpha: Tham số Beta để lấy mẫu tỷ lệ diện tích ban đầu.
            prob: Xác suất áp dụng CutMix trên mỗi lô dữ liệu.
            seed: Seed ngẫu nhiên.
        """
        if alpha <= 0:
            raise ValueError(f"alpha phải lớn hơn 0, nhận được: {alpha}")
        if not (0.0 < prob <= 1.0):
            raise ValueError(f"prob phải nằm trong khoảng (0, 1], nhận được: {prob}")

        self.alpha = alpha
        self.prob = prob
        self.rng = np.random.default_rng(seed)

    def _rand_bbox(self, W: int, H: int, lam: float) -> tuple[int, int, int, int]:
        """Tạo tọa độ hình chữ nhật ngẫu nhiên dựa trên tỷ lệ lambda."""
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)

        # Chọn tọa độ tâm ngẫu nhiên
        cx = int(self.rng.integers(W))
        cy = int(self.rng.integers(H))

        # Cắt và giới hạn trong phạm vi ảnh
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(W, cx + cut_w // 2)
        y2 = min(H, cy + cut_h // 2)

        return x1, y1, x2, y2

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor | list]:
        """
        Gom lô và dán đè CutMix đồng bộ.
        """
        # --- BƯỚC 1: Stack các trường dữ liệu ---
        left_images = torch.stack([s["left_image"] for s in batch])
        right_images = torch.stack([s["right_image"] for s in batch])
        left_missing = torch.stack([s["left_missing"] for s in batch])
        right_missing = torch.stack([s["right_missing"] for s in batch])
        labels = torch.stack([s["label"] for s in batch])
        ages = torch.stack([s["age"] for s in batch])
        patient_ids = [s["patient_id"] for s in batch]

        # --- BƯỚC 2: Quyết định có áp dụng CutMix không ---
        if self.rng.random() > self.prob:
            return {
                "left_image": left_images,
                "right_image": right_images,
                "left_missing": left_missing,
                "right_missing": right_missing,
                "label": labels,
                "age": ages,
                "patient_id": patient_ids,
                "mixed": False,
                "cutmix_lambda": torch.ones(1),
            }

        # --- BƯỚC 3: Lấy mẫu lambda ban đầu từ Beta(alpha, alpha) ---
        lam_init = float(self.rng.beta(self.alpha, self.alpha))

        # --- BƯỚC 4: Xác định tọa độ vùng cắt (Bounding Box) ---
        # Do hai ảnh mắt trái/phải cùng kích thước [B, 3, H, W], ta lấy kích thước từ left_images
        _, _, H, W = left_images.shape
        x1, y1, x2, y2 = self._rand_bbox(W, H, lam_init)

        # --- BƯỚC 5: Tính toán lambda thực tế dựa trên diện tích cắt ---
        cut_area = (x2 - x1) * (y2 - y1)
        total_area = W * H
        lam = 1.0 - cut_area / total_area

        # --- BƯỚC 6: Tạo hoán vị ngẫu nhiên ---
        n = len(batch)
        perm = torch.from_numpy(self.rng.permutation(n))

        # --- BƯỚC 7: Thực hiện dán đè đồng bộ cùng tọa độ cho cả hai mắt ---
        left_images_mix = left_images.clone()
        left_images_mix[:, :, y1:y2, x1:x2] = left_images[perm, :, y1:y2, x1:x2]

        right_images_mix = right_images.clone()
        right_images_mix[:, :, y1:y2, x1:x2] = right_images[perm, :, y1:y2, x1:x2]

        # Trộn nhãn và tuổi theo lambda thực tế
        labels_mix = lam * labels + (1.0 - lam) * labels[perm]
        ages_mix = lam * ages + (1.0 - lam) * ages[perm]

        # Kết hợp cờ khuyết thiếu mắt bằng logic AND
        left_missing_mix = left_missing & left_missing[perm]
        right_missing_mix = right_missing & right_missing[perm]

        return {
            "left_image": left_images_mix,
            "right_image": right_images_mix,
            "left_missing": left_missing_mix,
            "right_missing": right_missing_mix,
            "label": labels_mix,
            "age": ages_mix,
            "patient_id": patient_ids,
            "mixed": True,
            "cutmix_lambda": torch.tensor(lam),
        }


class BinocularAugmentCollator:
    """
    Collator kết hợp thông minh: Tự động chọn ngẫu nhiên giữa MixUp và CutMix
    nếu cả hai được bật, hoặc chỉ chạy một trong hai tuỳ thuộc cấu hình.
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
        """
        Khởi tạo bộ Gom và Trộn kết hợp.
        """
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix

        self.mixup_collator = (
            BinocularMixUpCollator(alpha=mixup_alpha, prob=mixup_prob, seed=seed)
            if use_mixup
            else None
        )
        self.cutmix_collator = (
            BinocularCutMixCollator(alpha=cutmix_alpha, prob=cutmix_prob, seed=seed)
            if use_cutmix
            else None
        )

        self.rng = np.random.default_rng(seed)

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor | list]:
        """
        Gom lô và áp dụng phép tăng cường ngẫu nhiên.
        """
        # Nếu cả hai phương pháp đều bật, tung đồng xu 50/50 để quyết định
        if self.use_mixup and self.use_cutmix:
            if self.rng.random() < 0.5:
                return self.mixup_collator(batch)
            else:
                return self.cutmix_collator(batch)

        # Nếu chỉ bật MixUp
        if self.use_mixup:
            return self.mixup_collator(batch)

        # Nếu chỉ bật CutMix
        if self.use_cutmix:
            return self.cutmix_collator(batch)

        # Phương án dự phòng: gom lô tiêu chuẩn không tăng cường (không trộn nhãn)
        left_images = torch.stack([s["left_image"] for s in batch])
        right_images = torch.stack([s["right_image"] for s in batch])
        left_missing = torch.stack([s["left_missing"] for s in batch])
        right_missing = torch.stack([s["right_missing"] for s in batch])
        labels = torch.stack([s["label"] for s in batch])
        ages = torch.stack([s["age"] for s in batch])
        patient_ids = [s["patient_id"] for s in batch]

        return {
            "left_image": left_images,
            "right_image": right_images,
            "left_missing": left_missing,
            "right_missing": right_missing,
            "label": labels,
            "age": ages,
            "patient_id": patient_ids,
            "mixed": False,
        }

    def __repr__(self) -> str:
        return (
            f"BinocularAugmentCollator(use_mixup={self.use_mixup}, use_cutmix={self.use_cutmix})"
        )
