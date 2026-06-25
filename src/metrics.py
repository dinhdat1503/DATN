"""
Chỉ số đánh giá (Metrics) cho phân loại nhị phân y sinh — ODIR-5K Phase 1.

Gồm:
1. compute_binary_metrics: Accuracy, Precision, Sensitivity (Recall), Specificity, F1, AUC-ROC.
2. find_best_threshold: Tìm ngưỡng tối ưu theo Chỉ số Youden (J = Sensitivity + Specificity − 1).
3. compute_age_metrics: MAE và hệ số tương quan Pearson cho tuổi (đã giải chuẩn hóa).

Các hàm chấp nhận đầu vào là torch.Tensor / numpy.ndarray / list.
Nhãn mềm (do MixUp/CutMix) được tự động nhị phân hóa (>= 0.5) khi tính confusion matrix / AUC.
"""

from __future__ import annotations

import numpy as np


def _to_numpy(x) -> np.ndarray:
    """Chuẩn hóa đầu vào (tensor / list / ndarray) về numpy float 1 chiều."""
    if hasattr(x, "detach"):  # torch.Tensor (có thể trên CUDA)
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).flatten()


def compute_binary_metrics(preds, targets, threshold: float = 0.5) -> dict[str, float]:
    """Tính bộ chỉ số y sinh cho phân loại nhị phân.

    Args:
        preds: Xác suất dự đoán (sau sigmoid) [N] hoặc [N, 1].
        targets: Nhãn thực [N] hoặc [N, 1]. 0 = Normal, 1 = Pathological. Nhãn mềm được làm tròn.
        threshold: Ngưỡng chuyển xác suất → nhãn nhị phân.

    Returns:
        Dict: accuracy, precision, sensitivity, specificity, f1, auc.
    """
    p = _to_numpy(preds)
    y = (_to_numpy(targets) >= 0.5).astype(np.float64)  # nhị phân hóa nhãn (an toàn với soft label)

    pred_bin = (p >= threshold).astype(np.float64)

    tp = float(np.sum(pred_bin * y))
    fp = float(np.sum(pred_bin * (1.0 - y)))
    fn = float(np.sum((1.0 - pred_bin) * y))
    tn = float(np.sum((1.0 - pred_bin) * (1.0 - y)))

    accuracy = float((pred_bin == y).mean())
    precision = tp / (tp + fp + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)   # Recall / TPR / Độ nhạy
    specificity = tn / (tn + fp + 1e-8)   # TNR / Độ đặc hiệu
    f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-8)

    # AUC-ROC độc lập ngưỡng
    auc = 0.5
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y)) > 1:
            auc = float(roc_auc_score(y, p))
    except Exception:
        auc = 0.5

    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "auc": float(auc),
    }


def find_best_threshold(preds, targets) -> float:
    """Tìm ngưỡng phân loại tối ưu theo Chỉ số Youden J = Sensitivity + Specificity − 1.

    Quét ngưỡng từ 0.05 đến 0.95 (bước 0.01) trên tập Validation.

    Args:
        preds: Xác suất dự đoán [N].
        targets: Nhãn thực [N].

    Returns:
        Ngưỡng tối ưu (float).
    """
    p = _to_numpy(preds)
    y = (_to_numpy(targets) >= 0.5).astype(np.float64)

    best_j = -1.0
    best_thresh = 0.5
    for thresh in np.arange(0.05, 0.95, 0.01):
        pred_bin = (p >= thresh).astype(np.float64)
        tp = np.sum(pred_bin * y)
        fp = np.sum(pred_bin * (1.0 - y))
        fn = np.sum((1.0 - pred_bin) * y)
        tn = np.sum((1.0 - pred_bin) * (1.0 - y))

        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_thresh = float(thresh)
    return best_thresh


def compute_age_metrics(age_pred, age_true, age_mean: float, age_std: float) -> dict[str, float]:
    """Tính MAE (năm) và Pearson cho tuổi sau khi giải chuẩn hóa Z-score.

    Args:
        age_pred: Tuổi dự đoán đã chuẩn hóa Z-score.
        age_true: Tuổi thực đã chuẩn hóa Z-score.
        age_mean, age_std: Thống kê dùng để giải chuẩn hóa về đơn vị năm.

    Returns:
        Dict: mae (năm), pearson.
    """
    pred = _to_numpy(age_pred) * age_std + age_mean
    true = _to_numpy(age_true) * age_std + age_mean

    mae = float(np.mean(np.abs(pred - true)))
    if len(pred) > 1 and pred.std() > 1e-8 and true.std() > 1e-8:
        pearson = float(np.corrcoef(pred, true)[0, 1])
    else:
        pearson = 0.0
    return {"mae": mae, "pearson": pearson}
