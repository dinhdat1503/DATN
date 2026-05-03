"""
MixUp Collator cho ODIR-5K Multi-task Learning.

MixUp trộn 2 ảnh và nhãn theo tỉ lệ λ ~ Beta(alpha, alpha):
    image_mix  = λ × img_A + (1-λ) × img_B
    labels_mix = λ × lbl_A + (1-λ) × lbl_B
    age_mix    = λ × age_A + (1-λ) × age_B

Tài liệu gốc:
    Zhang, H. et al. (2018). mixup: Beyond Empirical Risk Minimization.
    ICLR 2018. https://arxiv.org/abs/1710.09412

Cách dùng:
    from src.mixup import MixUpCollator
    collator = MixUpCollator(alpha=0.4, prob=0.5)
    loader = DataLoader(dataset, batch_size=32, collate_fn=collator)
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


class MixUpCollator:
    """Collate function áp dụng MixUp lên từng batch.

    Thay vì augment từng ảnh riêng lẻ, MixUp hoạt động ở cấp độ batch:
    - Nhận batch gồm N samples
    - Tạo ra permutation (hoán vị) ngẫu nhiên của batch
    - Trộn batch gốc với batch đã hoán vị theo tỉ lệ λ

    Args:
        alpha: Tham số của phân phối Beta. Giá trị càng nhỏ → λ càng
               gần 0 hoặc 1 (ít trộn). Giá trị càng lớn → λ gần 0.5
               (trộn mạnh). Khuyến nghị: 0.2–0.4 cho ảnh y tế.
        prob:  Xác suất áp dụng MixUp lên mỗi batch. Mặc định 0.5
               (50% batch được MixUp, 50% giữ nguyên).
        seed:  Seed ngẫu nhiên. None = không cố định.
    """

    def __init__(
        self,
        alpha: float = 0.4,
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
    # Hàm chính được DataLoader gọi cho mỗi batch
    # ------------------------------------------------------------------

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor | list]:
        """Nhận list samples từ Dataset, trả về batch đã MixUp.

        Args:
            batch: List của N dict, mỗi dict gồm:
                   {"image": Tensor[3,H,W], "labels": Tensor[8],
                    "age": Tensor[1], "filename": str}

        Returns:
            Dict với các Tensor đã stack và (tuỳ xác suất) đã MixUp.
        """
        # ── Bước 1: Stack từng trường thành Tensor batch ──────────────
        images    = torch.stack([s["image"]  for s in batch])   # [N,3,H,W]
        labels    = torch.stack([s["labels"] for s in batch])   # [N,8]
        ages      = torch.stack([s["age"]    for s in batch])   # [N,1]
        filenames = [s["filename"] for s in batch]              # list[str]

        # ── Bước 2: Quyết định có MixUp batch này không ───────────────
        if self.rng.random() > self.prob:
            # Không MixUp — trả về batch gốc kèm lambda=1 (không trộn)
            return {
                "image":    images,
                "labels":   labels,
                "age":      ages,
                "filename": filenames,
                "mixup_lambda": torch.ones(1),   # dùng để debug/log
                "mixed": False,
            }

        # ── Bước 3: Lấy λ từ phân phối Beta(alpha, alpha) ─────────────
        # Beta(α,α) đối xứng quanh 0.5:
        #   α=0.2 → λ thường rất gần 0 hoặc 1 (trộn nhẹ)
        #   α=0.4 → λ phân tán hơn về 0.5 (trộn vừa)
        #   α=1.0 → λ đồng đều [0,1] (trộn mạnh)
        lam = float(self.rng.beta(self.alpha, self.alpha))

        # Đảm bảo ảnh A luôn chiếm tỉ lệ lớn hơn (lam >= 0.5)
        # → giúp nhãn mới gần với nhãn gốc hơn, ổn định hơn khi train
        lam = max(lam, 1.0 - lam)

        # ── Bước 4: Tạo hoán vị ngẫu nhiên của batch ─────────────────
        n = len(batch)
        perm = torch.randperm(n)   # [N] — chỉ số ngẫu nhiên

        # ── Bước 5: Trộn ảnh ──────────────────────────────────────────
        # images[perm] là batch B (đã xáo thứ tự)
        # Kết quả: mỗi ảnh là hỗn hợp của 2 ảnh khác nhau trong batch
        images_mix = lam * images + (1.0 - lam) * images[perm]

        # ── Bước 6: Trộn nhãn (soft labels) ──────────────────────────
        # Nhãn mới KHÔNG còn là 0/1 cứng mà là giá trị thực [0,1]
        # Ví dụ: [G=0.7, N=0.3] thay vì [G=1, N=0]
        labels_mix = lam * labels + (1.0 - lam) * labels[perm]

        # ── Bước 7: Trộn tuổi ─────────────────────────────────────────
        # Tuổi cũng được nội suy tuyến tính
        # Ví dụ: 0.7×55 + 0.3×70 = 59.5 tuổi
        ages_mix = lam * ages + (1.0 - lam) * ages[perm]

        return {
            "image":    images_mix,
            "labels":   labels_mix,
            "age":      ages_mix,
            "filename": filenames,          # giữ filename gốc (chỉ để debug)
            "mixup_lambda": torch.tensor(lam),
            "mixed": True,
        }

    def __repr__(self) -> str:
        return (
            f"MixUpCollator(alpha={self.alpha}, prob={self.prob})"
        )


# ---------------------------------------------------------------------------
# Hàm tiện ích: tạo DataLoader đã tích hợp MixUp
# ---------------------------------------------------------------------------

def get_mixup_dataloader(
    dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    alpha: float = 0.4,
    prob: float = 0.5,
    seed: int | None = None,
) -> DataLoader:
    """Tạo DataLoader cho training với MixUp collator.

    Chỉ dùng cho tập TRAIN. Val/Test KHÔNG dùng MixUp
    vì cần đánh giá trên nhãn thật (0/1), không phải soft label.

    Args:
        dataset:     ODIRDataset instance
        batch_size:  Kích thước batch
        num_workers: Số worker đọc dữ liệu song song
        pin_memory:  True nếu dùng GPU
        alpha:       Tham số Beta distribution (0.2–0.4)
        prob:        Xác suất MixUp mỗi batch (0.0–1.0)
        seed:        Seed ngẫu nhiên

    Returns:
        DataLoader với MixUp collate_fn
    """
    collator = MixUpCollator(alpha=alpha, prob=prob, seed=seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,           # BẮT BUỘC shuffle=True khi dùng MixUp
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,         # Bỏ batch cuối nếu nhỏ hơn batch_size
        collate_fn=collator,    # ← MixUp được áp dụng ở đây
    )
