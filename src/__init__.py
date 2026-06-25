"""
Package src cho ODIR-5K Phase 1 — Phân loại nhị phân Siamese song nhãn (Normal vs Pathological)
kèm hồi quy tuổi võng mạc phụ trợ.

Export công khai các thành phần chính để tiện import.
"""

from __future__ import annotations

from src.augment import BinocularAugmentCollator
from src.config import load_config, set_seed
from src.dataset import BinocularDataset, build_dataloaders
from src.engine import evaluate_test, fit, run_epoch
from src.losses import BinaryFocalLoss, MultiTaskLoss
from src.metrics import compute_binary_metrics, find_best_threshold
from src.models import build_model

__all__ = [
    "BinocularAugmentCollator",
    "load_config",
    "set_seed",
    "BinocularDataset",
    "build_dataloaders",
    "evaluate_test",
    "fit",
    "run_epoch",
    "BinaryFocalLoss",
    "MultiTaskLoss",
    "compute_binary_metrics",
    "find_best_threshold",
    "build_model",
]
