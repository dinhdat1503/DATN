"""
Confusion Analysis cho ODIR-5K Multi-label Classification.

Phân tích:
  1. F1-score theo từng bệnh (per-label) → tìm bệnh yếu nhất
  2. Ma trận nhầm lẫn đa nhãn (Co-occurrence Confusion Matrix)
     → Khi model dự đoán sai bệnh X, nó thực tế hay nhầm với bệnh nào?
  3. False Positive Analysis: Model đoán CÓ bệnh X nhưng sai → bệnh nhân thực tế mắc bệnh gì?
  4. False Negative Analysis: Model BỎ SÓT bệnh X → model đoán bệnh nhân mắc bệnh gì thay thế?

Usage:
    PYTHONPATH=.venv/lib/python3.12/site-packages python3 confusion_analysis.py --model cnn
    PYTHONPATH=.venv/lib/python3.12/site-packages python3 confusion_analysis.py --model swin
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import ODIRDataset
from src.models import build_model
from src.transforms import get_transforms
from src.utils import LABELS, LABEL_NAMES

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

CONFIGS = {
    "cnn": {
        "checkpoint": "results/exp_3_cnn_preprocess_with_aug/best.pth",
        "model_type": "cnn",
        "img_size": 384,
        "variant": "tiny",
    },
    "swin": {
        "checkpoint": "results/exp_6_swin_no_preprocess/best.pth",
        "model_type": "swin",
        "img_size": 384,
        "variant": "tiny",
    },
}


# ──────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────

def load_model_and_predict(
    config: dict,
    device: torch.device,
    checkpoint: str | None = None,
    img_dir: str | None = None,
    splits_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load model, chạy inference trên test set, trả về probs, preds, targets."""

    project_root = Path(__file__).parent
    ckpt_path = Path(checkpoint) if checkpoint else (project_root / config["checkpoint"])

    if not ckpt_path.exists():
        print(f"[LỖI] Không tìm thấy checkpoint: {ckpt_path}")
        sys.exit(1)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Detect checkpoint format: wrapped dict vs raw state_dict
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        # Wrapped format: {"model_state": ..., "config": ..., ...}
        state_dict = ckpt["model_state"]
        cfg = ckpt.get("config", {})
    elif isinstance(ckpt, dict) and any(k.startswith("backbone.") or k.startswith("classification_head.") for k in ckpt.keys()):
        # Raw state_dict format
        state_dict = ckpt
        cfg = {}
    else:
        print(f"[LỖI] Không nhận dạng được format checkpoint. Keys: {list(ckpt.keys())[:5]}")
        sys.exit(1)

    # Build model
    m_cfg = cfg.get("model", {})
    model = build_model(
        model_type=config["model_type"],
        pretrained=False,
        freeze_backbone=False,
        dropout_cls=m_cfg.get("dropout_cls", 0.3),
        dropout_reg=m_cfg.get("dropout_reg", 0.2),
        img_size=config["img_size"],
        variant=config.get("variant", "tiny"),
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    print(f"[Model] Loaded checkpoint: {ckpt_path}")

    # Load test dataset
    s_dir = Path(splits_dir) if splits_dir else (project_root / "archive" / "splits_clean")
    i_dir = Path(img_dir) if img_dir else (project_root / "archive" / "enhanced_images")

    # Tính age stats từ train
    train_df = pd.read_csv(s_dir / "train.csv")
    train_df = train_df[train_df["Patient Age"] >= 5]
    age_mean = float(train_df["Patient Age"].mean())
    age_std = float(train_df["Patient Age"].std())

    test_dataset = ODIRDataset(
        csv_path=s_dir / "test.csv",
        img_dir=i_dir,
        transforms=get_transforms(mode="val", img_size=config["img_size"]),
        age_mean=age_mean,
        age_std=age_std,
    )

    loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=0
    )

    # Inference
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["labels"]
            output = model(images)
            probs = torch.sigmoid(output["logits"]).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.numpy())

    probs = np.concatenate(all_probs, axis=0)        # [N, 8]
    targets = np.concatenate(all_targets, axis=0)     # [N, 8]
    preds = (probs >= 0.5).astype(float)              # [N, 8]

    print(f"[Data] Test set: {len(probs)} samples\n")
    return probs, preds, targets


# ──────────────────────────────────────────────────────────────
# Analysis Functions
# ──────────────────────────────────────────────────────────────

def per_label_metrics(preds: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
    """Tính TP, FP, FN, TN, Precision, Recall, F1 cho từng bệnh."""
    rows = []
    for i, label in enumerate(LABELS):
        tp = int(np.sum((preds[:, i] == 1) & (targets[:, i] == 1)))
        fp = int(np.sum((preds[:, i] == 1) & (targets[:, i] == 0)))
        fn = int(np.sum((preds[:, i] == 0) & (targets[:, i] == 1)))
        tn = int(np.sum((preds[:, i] == 0) & (targets[:, i] == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = tp + fn  # số ca thực tế mắc bệnh

        rows.append({
            "Bệnh": f"{label} ({LABEL_NAMES[label]})",
            "Support": support,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
        })

    return pd.DataFrame(rows)


def false_positive_analysis(preds: np.ndarray, targets: np.ndarray) -> dict[str, pd.Series]:
    """Khi model đoán CÓ bệnh X nhưng SAI (FP), bệnh nhân thực tế mắc bệnh gì?"""
    result = {}
    for i, label in enumerate(LABELS):
        fp_mask = (preds[:, i] == 1) & (targets[:, i] == 0)
        if fp_mask.sum() == 0:
            result[label] = pd.Series(dtype=float)
            continue
        # Các bệnh thực tế của những ca FP này
        actual_labels = targets[fp_mask]
        counts = {}
        for j, other_label in enumerate(LABELS):
            if j == i:
                continue
            counts[other_label] = int(actual_labels[:, j].sum())
        result[label] = pd.Series(counts).sort_values(ascending=False)
    return result


def false_negative_analysis(preds: np.ndarray, targets: np.ndarray) -> dict[str, pd.Series]:
    """Khi model BỎ SÓT bệnh X (FN), model nghĩ bệnh nhân mắc bệnh gì thay thế?"""
    result = {}
    for i, label in enumerate(LABELS):
        fn_mask = (preds[:, i] == 0) & (targets[:, i] == 1)
        if fn_mask.sum() == 0:
            result[label] = pd.Series(dtype=float)
            continue
        # Các bệnh model dự đoán cho những ca FN này
        predicted_labels = preds[fn_mask]
        counts = {}
        for j, other_label in enumerate(LABELS):
            if j == i:
                continue
            counts[other_label] = int(predicted_labels[:, j].sum())
        result[label] = pd.Series(counts).sort_values(ascending=False)
    return result


def confusion_matrix_multilabel(preds: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
    """Ma trận nhầm lẫn đa nhãn: Confusion[i][j] = số lần label i bị nhầm thành label j.

    Cụ thể: Khi model sai trên label i (FP hoặc FN), label j có mặt bao nhiêu lần?
    """
    n_labels = len(LABELS)
    confusion = np.zeros((n_labels, n_labels), dtype=int)

    for i in range(n_labels):
        # Các mẫu mà model sai trên label i
        wrong_mask = preds[:, i] != targets[:, i]
        if wrong_mask.sum() == 0:
            continue
        for j in range(n_labels):
            if i == j:
                # Đường chéo: số mẫu bị sai trên chính label đó
                confusion[i][j] = int(wrong_mask.sum())
            else:
                # Ngoài đường chéo: trong những mẫu model sai label i,
                # bao nhiêu mẫu thực tế CÓ label j?
                confusion[i][j] = int(targets[wrong_mask, j].sum())

    df = pd.DataFrame(
        confusion,
        index=[f"{l} (thực tế)" for l in LABELS],
        columns=[f"{l}" for l in LABELS],
    )
    return df


def probability_confusion(probs: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
    """Phân tích xác suất trung bình: khi mắc bệnh X, model cho xác suất bao nhiêu cho từng bệnh?

    Giúp phát hiện bệnh nào có xác suất 'rò rỉ' cao sang bệnh khác.
    """
    n_labels = len(LABELS)
    avg_probs = np.zeros((n_labels, n_labels))

    for i in range(n_labels):
        mask = targets[:, i] == 1  # Các mẫu thực tế mắc bệnh i
        if mask.sum() == 0:
            continue
        for j in range(n_labels):
            avg_probs[i][j] = float(probs[mask, j].mean())

    df = pd.DataFrame(
        np.round(avg_probs, 4),
        index=[f"{l} (thực mắc)" for l in LABELS],
        columns=[f"P({l})" for l in LABELS],
    )
    return df


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def df_to_md(df: pd.DataFrame, show_index: bool = False) -> str:
    """Chuyển DataFrame sang markdown table thủ công (không cần tabulate)."""
    cols = list(df.columns)
    if show_index:
        cols = [df.index.name or ""] + cols

    rows_data = []
    for idx, row in df.iterrows():
        vals = [str(row[c]) for c in df.columns]
        if show_index:
            vals = [str(idx)] + vals
        rows_data.append(vals)

    # Tính độ rộng tối thiểu cho mỗi cột
    col_widths = [len(c) for c in cols]
    for row_vals in rows_data:
        for i, v in enumerate(row_vals):
            col_widths[i] = max(col_widths[i], len(v))

    # Build header
    header = "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cols)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"

    # Build rows
    body_lines = []
    for row_vals in rows_data:
        line = "| " + " | ".join(v.ljust(col_widths[i]) for i, v in enumerate(row_vals)) + " |"
        body_lines.append(line)

    return "\n".join([header, sep] + body_lines)


# ──────────────────────────────────────────────────────────────
# Report Generator
# ──────────────────────────────────────────────────────────────

def generate_report(
    model_name: str,
    probs: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
) -> str:
    """Tạo báo cáo Markdown chi tiết."""

    lines = []
    lines.append(f"# 🔍 Báo Cáo Confusion Analysis — {model_name.upper()}")
    lines.append("")
    lines.append(f"> Phân tích trên **{len(preds)} mẫu** tập Test với ngưỡng θ=0.5")
    lines.append("")

    # ── 1. Per-label metrics ──
    lines.append("---")
    lines.append("## 1. F1-Score Theo Từng Bệnh (Per-Label Metrics)")
    lines.append("")
    df_metrics = per_label_metrics(preds, targets)
    lines.append(df_to_md(df_metrics, show_index=False))
    lines.append("")

    # Tìm bệnh yếu nhất
    worst = df_metrics.loc[df_metrics["F1"].idxmin()]
    best = df_metrics.loc[df_metrics["F1"].idxmax()]
    lines.append(f"- **🔴 Bệnh yếu nhất:** {worst['Bệnh']} — F1 = **{worst['F1']}** (FN={worst['FN']}, FP={worst['FP']})")
    lines.append(f"- **🟢 Bệnh mạnh nhất:** {best['Bệnh']} — F1 = **{best['F1']}**")
    lines.append("")

    # ── 2. Confusion Matrix ──
    lines.append("---")
    lines.append("## 2. Ma Trận Nhầm Lẫn Đa Nhãn (Multi-label Confusion Matrix)")
    lines.append("")
    lines.append("> Đọc theo **hàng**: Khi model dự đoán **sai** label ở hàng → bao nhiêu mẫu trong đó thực tế **có** label ở cột?")
    lines.append("> Đường chéo = tổng số mẫu bị sai trên chính label đó.")
    lines.append("")
    df_confusion = confusion_matrix_multilabel(preds, targets)
    lines.append(df_to_md(df_confusion, show_index=True))
    lines.append("")

    # ── 3. Probability Confusion ──
    lines.append("---")
    lines.append("## 3. Ma Trận Xác Suất Trung Bình (Probability Confusion)")
    lines.append("")
    lines.append("> Đọc: Khi bệnh nhân **thực tế mắc** bệnh ở hàng → model cho xác suất trung bình cho từng bệnh ở cột là bao nhiêu?")
    lines.append("> **Xác suất cao ngoài đường chéo** = model hay nhầm 2 bệnh này với nhau.")
    lines.append("")
    df_prob = probability_confusion(probs, targets)
    lines.append(df_to_md(df_prob, show_index=True))
    lines.append("")

    # ── 4. False Positive Analysis ──
    lines.append("---")
    lines.append("## 4. False Positive Analysis (Model Đoán CÓ Nhưng SAI)")
    lines.append("")
    lines.append("> Khi model **đoán bệnh nhân CÓ bệnh X nhưng SAI**, bệnh nhân đó thực tế mắc bệnh gì?")
    lines.append("")
    fp_analysis = false_positive_analysis(preds, targets)
    for label, series in fp_analysis.items():
        fp_count = int(((preds[:, LABELS.index(label)] == 1) & (targets[:, LABELS.index(label)] == 0)).sum())
        if fp_count == 0:
            lines.append(f"### {label} ({LABEL_NAMES[label]}) — 0 FP ✅")
            lines.append("")
            continue
        lines.append(f"### {label} ({LABEL_NAMES[label]}) — {fp_count} FP")
        lines.append(f"Bệnh thực tế của {fp_count} ca bị đoán nhầm CÓ {label}:")
        lines.append("")
        for other_label, count in series.items():
            if count > 0:
                pct = count / fp_count * 100
                lines.append(f"- **{other_label}** ({LABEL_NAMES[other_label]}): {count} ca ({pct:.1f}%)")
        lines.append("")

    # ── 5. False Negative Analysis ──
    lines.append("---")
    lines.append("## 5. False Negative Analysis (Model BỎ SÓT Bệnh)")
    lines.append("")
    lines.append("> Khi model **bỏ sót bệnh X** (có bệnh nhưng model đoán không), model nghĩ bệnh nhân mắc bệnh gì?")
    lines.append("")
    fn_analysis = false_negative_analysis(preds, targets)
    for label, series in fn_analysis.items():
        fn_count = int(((preds[:, LABELS.index(label)] == 0) & (targets[:, LABELS.index(label)] == 1)).sum())
        if fn_count == 0:
            lines.append(f"### {label} ({LABEL_NAMES[label]}) — 0 FN ✅")
            lines.append("")
            continue
        lines.append(f"### {label} ({LABEL_NAMES[label]}) — {fn_count} FN")
        lines.append(f"Trong {fn_count} ca bị bỏ sót, model đoán bệnh nhân mắc:")
        lines.append("")
        for other_label, count in series.items():
            if count > 0:
                pct = count / fn_count * 100
                lines.append(f"- **{other_label}** ({LABEL_NAMES[other_label]}): {count} ca ({pct:.1f}%)")
        lines.append("")

    # ── 6. Tóm tắt & Gợi ý ──
    lines.append("---")
    lines.append("## 6. Tóm Tắt & Gợi Ý Cải Thiện F1-Score")
    lines.append("")

    # Tự động phát hiện top confusion pairs từ probability matrix
    df_prob_raw = probability_confusion(probs, targets)
    prob_values = df_prob_raw.values.copy()
    np.fill_diagonal(prob_values, 0)  # Bỏ đường chéo

    # Top 5 cặp nhầm lẫn
    lines.append("### Top cặp bệnh dễ nhầm lẫn nhất (theo xác suất rò rỉ):")
    lines.append("")
    flat_indices = np.argsort(prob_values.ravel())[::-1]
    seen = set()
    count = 0
    for idx in flat_indices:
        if count >= 5:
            break
        i, j = divmod(idx, len(LABELS))
        pair = tuple(sorted([LABELS[i], LABELS[j]]))
        if pair in seen:
            continue
        seen.add(pair)
        lines.append(
            f"{count+1}. **{LABELS[i]}** ↔ **{LABELS[j]}** "
            f"({LABEL_NAMES[LABELS[i]]} ↔ {LABEL_NAMES[LABELS[j]]}): "
            f"P({LABELS[j]}|mắc {LABELS[i]}) = **{prob_values[i][j]:.4f}**, "
            f"P({LABELS[i]}|mắc {LABELS[j]}) = **{prob_values[j][i]:.4f}**"
        )
        count += 1

    lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Confusion Analysis cho ODIR-5K")
    parser.add_argument(
        "--model", choices=["cnn", "swin"], default="cnn",
        help="Chọn model để phân tích (cnn hoặc swin)",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Đường dẫn custom checkpoint file",
    )
    parser.add_argument(
        "--img_dir", default=None,
        help="Đường dẫn custom thư mục ảnh",
    )
    parser.add_argument(
        "--splits_dir", default=None,
        help="Đường dẫn custom thư mục splits",
    )
    parser.add_argument(
        "--output", default=None,
        help="Đường dẫn file output markdown (mặc định: docs/confusion_analysis_{model}.md)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    config = CONFIGS[args.model]
    probs, preds, targets = load_model_and_predict(
        config=config,
        device=device,
        checkpoint=args.checkpoint,
        img_dir=args.img_dir,
        splits_dir=args.splits_dir,
    )

    # Generate report
    report = generate_report(args.model, probs, preds, targets)

    # Save
    output_path = args.output or f"docs/confusion_analysis_{args.model}.md"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  ✅ Báo cáo đã lưu: {output_path}")
    print(f"{'='*60}")

    # Print summary to console
    print("\n" + "─" * 60)
    print("  TÓM TẮT NHANH")
    print("─" * 60)

    df_metrics = per_label_metrics(preds, targets)
    print("\n📊 F1-Score theo bệnh:")
    for _, row in df_metrics.iterrows():
        bar = "█" * int(row["F1"] * 30)
        print(f"  {row['Bệnh']:45s} F1={row['F1']:.4f} {bar}")

    worst = df_metrics.loc[df_metrics["F1"].idxmin()]
    print(f"\n🔴 Bệnh cần cải thiện nhất: {worst['Bệnh']} (F1={worst['F1']})")


if __name__ == "__main__":
    main()
