"""
Utility functions cho ODIR-5K training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

LABELS = ["N", "D", "G", "C", "A", "H", "M", "O"]
LABEL_NAMES = {
    "N": "Normal",
    "D": "Diabetes Retinopathy",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "Age-related Macular Degeneration",
    "H": "Hypertension Retinopathy",
    "M": "Pathological Myopia",
    "O": "Other Diseases",
}


def get_label_names() -> dict[str, str]:
    """Trả về mapping từ ký hiệu nhãn sang tên đầy đủ."""
    return LABEL_NAMES.copy()


def compute_class_weights(
    train_csv_path: str | Path,
) -> torch.Tensor:
    """Tính pos_weight tensor cho BCEWithLogitsLoss.

    pos_weight[i] = num_negative[i] / num_positive[i]

    Returns:
        torch.FloatTensor shape [8]
    """
    df = pd.read_csv(train_csv_path)
    n_total = len(df)
    weights = []
    for label in LABELS:
        n_pos = int(df[label].sum())
        n_neg = n_total - n_pos
        w = n_neg / max(n_pos, 1)
        weights.append(w)
    return torch.FloatTensor(weights)


def compute_age_stats(
    train_csv_path: str | Path,
) -> dict[str, float]:
    """Tính mean và std tuổi từ training set.

    Returns:
        {"mean": float, "std": float}
    """
    df = pd.read_csv(train_csv_path)
    ages = df["Patient Age"].values.astype(float)
    return {
        "mean": float(np.mean(ages)),
        "std": float(np.std(ages)),
    }


def load_metadata(
    metadata_path: str | Path,
) -> dict:
    """Đọc metadata.json chứa class weights, age stats, split info."""
    with open(metadata_path) as f:
        return json.load(f)


def get_pos_weight_from_metadata(
    metadata: dict,
) -> torch.Tensor:
    """Lấy pos_weight tensor từ metadata dict.

    Returns:
        torch.FloatTensor shape [8]
    """
    weights = metadata["class_weights"]
    return torch.FloatTensor([weights[label] for label in LABELS])


def normalize_age(
    age: float | np.ndarray,
    mean: float,
    std: float,
) -> float | np.ndarray:
    """Chuẩn hóa tuổi: (age - mean) / std."""
    return (age - mean) / std


def denormalize_age(
    age_norm: float | np.ndarray,
    mean: float,
    std: float,
) -> float | np.ndarray:
    """Giải chuẩn hóa tuổi: age_norm * std + mean."""
    return age_norm * std + mean


def compute_multilabel_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float | list[float] | torch.Tensor = 0.5,
) -> dict[str, float]:
    """Tính các metrics cho multi-label classification.

    Args:
        preds: predicted probabilities [batch, 8]
        targets: ground truth labels [batch, 8]
        threshold: ngưỡng (hoặc danh sách 8 ngưỡng) để chuyển prob -> binary

    Returns:
        Dict chứa accuracy, precision, recall, f1 (macro)
    """
    # Chuyển threshold sang Tensor có kích thước phù hợp để so sánh broadcast
    if isinstance(threshold, (list, np.ndarray)):
        thresh_tensor = torch.FloatTensor(threshold).to(preds.device)
    elif isinstance(threshold, torch.Tensor):
        thresh_tensor = threshold.to(preds.device)
    else:
        thresh_tensor = torch.FloatTensor([threshold] * preds.shape[1]).to(preds.device)

    pred_binary = (preds >= thresh_tensor).float()

    # Per-label metrics
    tp = (pred_binary * targets).sum(dim=0)
    fp = (pred_binary * (1 - targets)).sum(dim=0)
    fn = ((1 - pred_binary) * targets).sum(dim=0)
    tn = ((1 - pred_binary) * (1 - targets)).sum(dim=0)

    # Precision, Recall per label
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Macro average
    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1 = f1.mean().item()

    # Overall accuracy
    correct = (pred_binary == targets).float().mean().item()

    # AUC (nếu có sklearn)
    result = {
        "accuracy": correct,
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
        "f1_macro": macro_f1,
    }

    # Per-label F1
    for i, label in enumerate(LABELS):
        result[f"f1_{label}"] = f1[i].item()

    return result


def find_best_thresholds(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> list[float]:
    """Tìm ngưỡng (threshold) tối ưu động cho từng class để tối đa hóa F1-score.

    Quét qua các ngưỡng từ 0.05 đến 0.95 với bước nhảy 0.01 độc lập trên tập Validation.

    Args:
        preds: predicted probabilities [N, 8] từ validation set
        targets: ground truth nhãn thật [N, 8]

    Returns:
        Danh sách 8 số thực đại diện cho ngưỡng tối ưu của 8 bệnh.
    """
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    n_classes = preds_np.shape[1]
    best_thresholds = []

    for c in range(n_classes):
        best_f1 = -1.0
        best_thresh = 0.5
        # Quét dải ngưỡng từ 0.05 đến 0.95
        for thresh in np.arange(0.05, 0.95, 0.01):
            pred_binary = (preds_np[:, c] >= thresh).astype(float)
            tp = np.sum(pred_binary * targets_np[:, c])
            fp = np.sum(pred_binary * (1 - targets_np[:, c]))
            fn = np.sum((1 - pred_binary) * targets_np[:, c])

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)
        best_thresholds.append(best_thresh)

    return best_thresholds
