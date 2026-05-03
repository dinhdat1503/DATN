"""
src — ODIR-5K training package.

Exports công khai:
    Dataset & DataLoader
        ODIRDataset, get_dataloaders

    Augmentation Collators
        MixUpCollator, get_mixup_dataloader
        CutMixCollator, get_cutmix_dataloader

    Transforms
        get_transforms, get_train_transforms, get_val_transforms

    Utilities
        LABELS, LABEL_NAMES
        compute_class_weights, compute_age_stats
        normalize_age, denormalize_age
        compute_multilabel_metrics
        load_metadata, get_pos_weight_from_metadata
"""

from src.dataset import ODIRDataset, get_dataloaders
from src.mixup import MixUpCollator, get_mixup_dataloader
from src.cutmix import CutMixCollator, get_cutmix_dataloader
from src.transforms import get_transforms, get_train_transforms, get_val_transforms
from src.utils import (
    LABELS,
    LABEL_NAMES,
    compute_class_weights,
    compute_age_stats,
    normalize_age,
    denormalize_age,
    compute_multilabel_metrics,
    load_metadata,
    get_pos_weight_from_metadata,
)

__all__ = [
    # Dataset
    "ODIRDataset",
    "get_dataloaders",
    # Augmentation collators
    "MixUpCollator",
    "get_mixup_dataloader",
    "CutMixCollator",
    "get_cutmix_dataloader",
    # Transforms
    "get_transforms",
    "get_train_transforms",
    "get_val_transforms",
    # Utilities
    "LABELS",
    "LABEL_NAMES",
    "compute_class_weights",
    "compute_age_stats",
    "normalize_age",
    "denormalize_age",
    "compute_multilabel_metrics",
    "load_metadata",
    "get_pos_weight_from_metadata",
]
