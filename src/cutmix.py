"""
CutMix Collator cho ODIR-5K Multi-task Learning.

CutMix cắt một vùng hình chữ nhật ngẫu nhiên từ ảnh B và dán đè lên ảnh A.
Lambda (λ) được tính lại theo diện tích vùng cắt thực tế:

    image_mix[y1:y2, x1:x2] ← image_B[y1:y2, x1:x2]
    λ = 1 - (x2-x1)(y2-y1) / (W × H)
    labels_mix = λ × labels_A + (1-λ) × labels_B
    age_mix    = λ × age_A    + (1-λ) × age_B

Khác với MixUp (trộn pixel toàn ảnh), CutMix giữ nguyên cấu trúc
từng vùng → phù hợp hơn với ảnh y tế (mạch máu, tổn thương cục bộ).

Tài liệu gốc:
    Yun, S. et al. (2019). CutMix: Regularization Strategy to Train
    Strong Classifiers with Localizable Features.
    ICCV 2019. https://arxiv.org/abs/1905.04899

Cách dùng:
    from src.cutmix import CutMixCollator
    collator = CutMixCollator(alpha=1.0, prob=0.5)
    loader = DataLoader(dataset, batch_size=32, collate_fn=collator)
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


class CutMixCollator:
    """Collate function áp dụng CutMix lên từng batch.

    CutMix hoạt động ở cấp độ batch:
    - Nhận batch gồm N samples
    - Tạo permutation (hoán vị) ngẫu nhiên của batch
    - Với mỗi cặp (A, B): cắt vùng chữ nhật từ B, dán đè lên A
    - λ được tính lại theo diện tích vùng cắt thực tế

    Ưu điểm so với MixUp trong bài toán ảnh y tế:
    - Giữ nguyên cấu trúc cục bộ (mạch máu, tổn thương không bị mờ)
    - Mô hình học được định vị đặc trưng (localization)
    - Tăng đa dạng vùng không gian mà không làm méo pixel

    Args:
        alpha: Tham số Beta distribution để lấy mẫu λ ban đầu.
               α = 1.0 → λ đều trong [0,1] (chuẩn theo paper gốc).
               α nhỏ hơn → vùng cắt thường nhỏ hơn.
        prob:  Xác suất áp dụng CutMix lên mỗi batch. Mặc định 0.5.
        seed:  Seed ngẫu nhiên. None = không cố định.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        prob: float = 0.5,
        seed: int | None = None,
    ) -> None:
        if alpha <= 0:
            raise ValueError(f"alpha phải > 0, nhận được: {alpha}")
        if not (0.0 < prob <= 1.0):
            raise ValueError(f"prob phải trong (0, 1], nhận được: {prob}")

        self.alpha = alpha
        self.prob  = prob
        self.rng   = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Hàm tính toán vùng cắt (bounding box)
    # ------------------------------------------------------------------

    def _rand_bbox(
        self,
        W: int,
        H: int,
        lam: float,
    ) -> tuple[int, int, int, int]:
        """Tính toán bounding box ngẫu nhiên cho vùng cắt.

        Kích thước vùng cắt tỉ lệ với sqrt(1 - λ):
            W_cut = W × sqrt(1 - λ)
            H_cut = H × sqrt(1 - λ)

        Trung tâm (cx, cy) được chọn ngẫu nhiên trong ảnh.
        Vùng được clip vào [0, W] × [0, H].

        Args:
            W: Chiều rộng ảnh (số cột)
            H: Chiều cao ảnh (số hàng)
            lam: Lambda từ Beta(α,α) — dùng để tính kích thước cắt

        Returns:
            (x1, y1, x2, y2) — tọa độ góc trên-trái và dưới-phải
        """
        cut_ratio = np.sqrt(1.0 - lam)          # tỉ lệ cạnh vùng cắt
        cut_w = int(W * cut_ratio)               # chiều rộng vùng cắt
        cut_h = int(H * cut_ratio)               # chiều cao vùng cắt

        # Tâm ngẫu nhiên
        cx = int(self.rng.integers(W))
        cy = int(self.rng.integers(H))

        # Tính tọa độ và clip vào biên ảnh
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(W, cx + cut_w // 2)
        y2 = min(H, cy + cut_h // 2)

        return x1, y1, x2, y2

    # ------------------------------------------------------------------
    # Hàm chính được DataLoader gọi cho mỗi batch
    # ------------------------------------------------------------------

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor | list]:
        """Nhận list samples từ Dataset, trả về batch đã CutMix.

        Args:
            batch: List của N dict, mỗi dict gồm:
                   {"image": Tensor[3,H,W], "labels": Tensor[8],
                    "age": Tensor[1], "filename": str}

        Returns:
            Dict với các Tensor đã stack và (tuỳ xác suất) đã CutMix.
        """
        # ── Bước 1: Stack từng trường thành Tensor batch ──────────────
        images    = torch.stack([s["image"]  for s in batch])   # [N,3,H,W]
        labels    = torch.stack([s["labels"] for s in batch])   # [N,8]
        ages      = torch.stack([s["age"]    for s in batch])   # [N,1]
        filenames = [s["filename"] for s in batch]              # list[str]

        # ── Bước 2: Quyết định có CutMix batch này không ─────────────
        if self.rng.random() > self.prob:
            return {
                "image":         images,
                "labels":        labels,
                "age":           ages,
                "filename":      filenames,
                "cutmix_lambda": torch.ones(1),   # λ=1 → không cắt gì
                "mixed":         False,
            }

        # ── Bước 3: Lấy λ ban đầu từ Beta(α, α) ──────────────────────
        lam_init = float(self.rng.beta(self.alpha, self.alpha))

        # ── Bước 4: Tính bounding box dựa trên λ ban đầu ─────────────
        _, _, H, W = images.shape
        x1, y1, x2, y2 = self._rand_bbox(W, H, lam_init)

        # ── Bước 5: Tính λ thực tế từ diện tích vùng cắt ─────────────
        # λ_actual = 1 - (diện tích vùng B) / (diện tích toàn ảnh)
        # Khi vùng cắt lớn: λ nhỏ → nhãn B chiếm tỉ lệ lớn
        # Khi vùng cắt nhỏ: λ lớn → nhãn A vẫn chiếm chủ đạo
        cut_area = (x2 - x1) * (y2 - y1)
        total_area = W * H
        lam = 1.0 - cut_area / total_area         # λ thực tế

        # ── Bước 6: Tạo hoán vị ngẫu nhiên ───────────────────────────
        n = len(batch)
        perm = torch.randperm(n)

        # ── Bước 7: Dán vùng B vào ảnh A (in-place trên bản copy) ─────
        # images[perm] là batch B (đã xáo thứ tự)
        images_mix = images.clone()
        images_mix[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]

        # ── Bước 8: Trộn nhãn theo λ thực tế ─────────────────────────
        # Giống MixUp: nhãn soft, tỉ lệ theo diện tích vùng của mỗi ảnh
        labels_mix = lam * labels + (1.0 - lam) * labels[perm]

        # ── Bước 9: Trộn tuổi theo λ thực tế ─────────────────────────
        ages_mix = lam * ages + (1.0 - lam) * ages[perm]

        return {
            "image":         images_mix,
            "labels":        labels_mix,
            "age":           ages_mix,
            "filename":      filenames,          # giữ filename gốc (debug)
            "cutmix_lambda": torch.tensor(lam),
            "mixed":         True,
        }

    def __repr__(self) -> str:
        return (
            f"CutMixCollator(alpha={self.alpha}, prob={self.prob})"
        )


# ---------------------------------------------------------------------------
# Hàm tiện ích: tạo DataLoader đã tích hợp CutMix
# ---------------------------------------------------------------------------

def get_cutmix_dataloader(
    dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    alpha: float = 1.0,
    prob: float = 0.5,
    seed: int | None = None,
) -> DataLoader:
    """Tạo DataLoader cho training với CutMix collator.

    Chỉ dùng cho tập TRAIN. Val/Test KHÔNG dùng CutMix
    vì cần đánh giá trên nhãn thật (0/1), không phải soft label.

    Args:
        dataset:     ODIRDataset instance
        batch_size:  Kích thước batch
        num_workers: Số worker đọc dữ liệu song song
        pin_memory:  True nếu dùng GPU
        alpha:       Tham số Beta distribution (mặc định 1.0 theo paper)
        prob:        Xác suất CutMix mỗi batch (0.0–1.0)
        seed:        Seed ngẫu nhiên

    Returns:
        DataLoader với CutMix collate_fn
    """
    collator = CutMixCollator(alpha=alpha, prob=prob, seed=seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,           # BẮT BUỘC shuffle=True khi dùng CutMix
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,         # Bỏ batch cuối nếu nhỏ hơn batch_size
        collate_fn=collator,    # ← CutMix được áp dụng ở đây
    )
