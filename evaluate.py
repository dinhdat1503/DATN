"""
Evaluation script — So sánh 2 thực nghiệm preprocessing.

Đọc test_results.json từ mỗi thực nghiệm và tạo bảng so sánh.

Usage:
    # So sánh 2 thực nghiệm đã chạy xong
    python evaluate.py

    # Chỉ đánh giá 1 thực nghiệm cụ thể
    python evaluate.py --exp results/exp_raw

    # Chạy lại inference trên test set (cần checkpoint)
    python evaluate.py --rerun --config configs/exp_raw.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.utils import LABELS, LABEL_NAMES

try:
    from sklearn.metrics import (
        roc_auc_score,
        f1_score,
        classification_report,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(exp_dir: Path) -> dict | None:
    """
    Đọc file kết quả JSON từ thư mục thực nghiệm.
    Hỗ trợ nhiều tên file: test_results.json, results.json, r*_result.json
    """
    # Ưu tiên theo thứ tự
    candidates = (
        list(exp_dir.glob("test_results.json"))
        + list(exp_dir.glob("results.json"))
        + sorted(exp_dir.glob("r*_result.json"))
        + sorted(exp_dir.glob("*.json"))
    )
    # Lọc bỏ file config.yaml bị lẫn (chỉ lấy .json)
    candidates = [p for p in candidates if p.suffix == ".json"]

    if not candidates:
        print(f"  [WARN] Không tìm thấy file JSON nào trong: {exp_dir}")
        return None

    results_path = candidates[0]
    print(f"  [INFO] Đọc kết quả: {results_path.name}")
    with open(results_path) as f:
        data = json.load(f)

    # Chuẩn hóa key: một số file lưu dưới dạng nested {"test": {...}}
    if "test" in data and isinstance(data["test"], dict):
        flat = {"experiment": data.get("exp", exp_dir.name)}
        flat.update({f"test_{k}": v for k, v in data["test"].items()})
        if "best_val_f1" in data:
            flat["best_val_f1"] = data["best_val_f1"]
        return flat

    # Đảm bảo có trường experiment
    if "experiment" not in data:
        data["experiment"] = data.get("exp", exp_dir.name)
    return data


def format_value(v, fmt=".4f") -> str:
    if isinstance(v, float):
        return f"{v:{fmt}}"
    return str(v)


def print_comparison_table(results_list: list[dict]) -> str:
    """In bảng so sánh và trả về string markdown."""
    if not results_list:
        print("[ERROR] Không có kết quả để so sánh!")
        return ""

    exp_names = [r["experiment"] for r in results_list]

    # --- Metrics chính ---
    main_metrics = [
        ("test_f1_macro",       "F1-macro",         ".4f"),
        ("test_auc_roc",        "AUC-ROC (macro)",  ".4f"),
        ("test_accuracy",       "Accuracy",         ".4f"),
        ("test_precision_macro","Precision (macro)", ".4f"),
        ("test_recall_macro",   "Recall (macro)",   ".4f"),
        ("test_age_mae",        "Age MAE (years)",  ".2f"),
        ("test_age_pearson",    "Age Pearson r",    ".4f"),
    ]

    # --- Per-label F1 ---
    per_label_metrics = [
        (f"test_f1_{label}", f"F1 - {label} ({LABEL_NAMES.get(label,'?')})", ".4f")
        for label in LABELS
    ]

    lines = []
    lines.append("\n" + "="*72)
    lines.append("  KẾT QUẢ SO SÁNH: TIỀN XỬ LÝ ẢNH — ABLATION STUDY")
    lines.append("="*72)

    # Header
    col_width = max(len(n) for n in exp_names) + 2
    header = f"  {'Metric':<35}" + "".join(f"{n:>{col_width}}" for n in exp_names)
    lines.append(header)
    lines.append("  " + "─"*70)

    # Main metrics
    lines.append("\n  [A] CLASSIFICATION METRICS")
    for key, label, fmt in main_metrics[:5]:
        row = f"  {label:<35}"
        for r in results_list:
            val = r.get(key, -1)
            row += f"{format_value(val, fmt):>{col_width}}"
        lines.append(row)

    lines.append("\n  [B] REGRESSION METRICS (Tuổi sinh học)")
    for key, label, fmt in main_metrics[5:]:
        row = f"  {label:<35}"
        for r in results_list:
            val = r.get(key, -1)
            row += f"{format_value(val, fmt):>{col_width}}"
        lines.append(row)

    lines.append("\n  [C] PER-LABEL F1 SCORE")
    for key, label, fmt in per_label_metrics:
        row = f"  {label:<35}"
        for r in results_list:
            val = r.get(key, -1)
            row += f"{format_value(val, fmt):>{col_width}}"
        lines.append(row)

    lines.append("\n" + "="*72)

    # --- Delta Analysis (nếu có đúng 2 thực nghiệm) ---
    if len(results_list) == 2:
        r_raw, r_enh = results_list[0], results_list[1]
        lines.append("\n  [D] PHÂN TÍCH DELTA (Enhanced - Raw)")
        lines.append("  " + "─"*70)

        all_metrics = main_metrics + per_label_metrics
        improvements = []
        regressions  = []

        for key, label, fmt in all_metrics:
            v_raw = r_raw.get(key, None)
            v_enh = r_enh.get(key, None)
            if v_raw is None or v_enh is None or v_raw < 0 or v_enh < 0:
                continue

            delta = v_enh - v_raw
            # MAE: delta âm = cải thiện (nhỏ hơn = tốt hơn)
            if "mae" in key:
                sign = "↑" if delta < 0 else ("↓" if delta > 0 else "=")
                is_better = delta < 0
            else:
                sign = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
                is_better = delta > 0

            tag = "✅" if is_better else ("⚠️" if delta != 0 else "═")
            line = f"  {tag} {label:<35}  Δ = {delta:+.4f}  {sign}"
            lines.append(line)

            if is_better:
                improvements.append(label)
            elif delta != 0:
                regressions.append(label)

        lines.append("\n  " + "─"*70)
        lines.append(f"  Cải thiện ({len(improvements)}): {', '.join(improvements) if improvements else 'Không có'}")
        lines.append(f"  Suy giảm  ({len(regressions)}):  {', '.join(regressions) if regressions else 'Không có'}")

        # Kết luận
        f1_delta = r_enh.get("test_f1_macro", 0) - r_raw.get("test_f1_macro", 0)
        mae_delta = r_enh.get("test_age_mae", 0) - r_raw.get("test_age_mae", 0)
        lines.append("\n  [KẾT LUẬN]")
        if f1_delta > 0.01:
            lines.append("  ✅ Tiền xử lý nâng cao (Ben Graham + CLAHE) CẢI THIỆN đáng kể")
            lines.append(f"     F1-macro tăng {f1_delta:+.4f} → NÊN sử dụng trong pipeline cuối.")
        elif f1_delta > 0:
            lines.append("  ✅ Tiền xử lý nâng cao cải thiện nhẹ — cân nhắc dùng.")
        elif f1_delta > -0.01:
            lines.append("  ≈  Hiệu quả tương đương — tiền xử lý không gây hại.")
        else:
            lines.append("  ⚠️  Tiền xử lý nâng cao KHÔNG cải thiện — cần xem xét lại.")

    lines.append("="*72 + "\n")
    output = "\n".join(lines)
    print(output)
    return output


def save_markdown_report(output: str, results_dir: Path) -> None:
    """Lưu bảng so sánh dưới dạng markdown."""
    md_path = results_dir / "comparison_table.md"

    md = "# Ablation Study — Kết Quả So Sánh Tiền Xử Lý\n\n"
    md += "**Câu hỏi nghiên cứu:** Tiền xử lý ảnh nâng cao (Ben Graham + CLAHE) có thực sự cải thiện hiệu suất mô hình không?\n\n"
    md += "```\n" + output + "\n```\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n→ Báo cáo so sánh lưu tại: {md_path}")


def rerun_inference(config_path: str) -> None:
    """Chạy lại inference trên test set với best model checkpoint."""
    import yaml
    from src.dataset import get_dataloaders
    from src.loss import MultiTaskLoss
    from src.models import build_model
    from src.utils import get_pos_weight_from_metadata, load_metadata

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    project_root = Path(config_path).parent.parent
    splits_dir   = project_root / cfg["splits_dir"]
    img_dir      = project_root / cfg["img_dir"]
    results_dir  = project_root / cfg["output"]["results_dir"]
    best_path    = results_dir / "best_model.pth"

    if not best_path.exists():
        print(f"[ERROR] Không tìm thấy checkpoint: {best_path}")
        return

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata   = load_metadata(splits_dir / "metadata.json")
    pos_weight = get_pos_weight_from_metadata(metadata)
    age_mean   = metadata["age_stats"]["mean"]
    age_std    = metadata["age_stats"]["std"]

    tr_cfg = cfg["training"]
    dataloaders = get_dataloaders(
        splits_dir=splits_dir,
        img_dir=img_dir,
        img_size=tr_cfg["img_size"],
        batch_size=tr_cfg["batch_size"],
        num_workers=0,
    )

    m_cfg = cfg["model"]
    model = build_model(
        model_type      = cfg.get("model_type", "cnn"),          # cần thiết để phân biệt CNN vs Swin
        pretrained      = False,       # Load từ checkpoint — không cần pretrained
        freeze_backbone = m_cfg.get("freeze_backbone", False),
        dropout_cls     = m_cfg.get("dropout_cls", 0.3),
        dropout_reg     = m_cfg.get("dropout_reg", 0.2),
        img_size        = tr_cfg.get("img_size", 224),            # Swin/ViT cần biết img_size
        variant         = m_cfg.get("variant", "tiny"),           # tiny|small|base cho Swin
        pretrained_path = m_cfg.get("pretrained_path", None),
    ).to(device)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    criterion = MultiTaskLoss(pos_weight=pos_weight, lam=cfg["loss"]["lam"], device=device)

    # Import run_epoch từ train.py
    sys.path.insert(0, str(project_root))
    from train import run_epoch

    print(f"\nChạy lại inference: {cfg['experiment_name']} — test set")
    test_m = run_epoch(
        model, dataloaders["test"], criterion, None,
        device, "test", age_mean, age_std,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "experiment": cfg["experiment_name"],
            **test_m,
        }, f, indent=2)
    print(f"→ Kết quả lưu tại: {results_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="So sánh kết quả 2 thực nghiệm preprocessing"
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Thư mục gốc chứa kết quả (mặc định: results/)"
    )
    parser.add_argument(
        "--exps", nargs="+",
        default=[
            "results/exp_1_cnn_no_preprocess",
            "results/exp_2_cnn_preprocess_no_aug",
            "results/exp_3_cnn_preprocess_with_aug",
            "results/exp_4_retfound_no_preprocess",
            "results/exp_5_retfound_preprocess_no_aug",
            "results/exp_6_retfound_preprocess_with_aug",
        ],
        help="Danh sách thư mục thực nghiệm cần so sánh"
    )
    parser.add_argument(
        "--rerun", action="store_true",
        help="Chạy lại inference trước khi so sánh"
    )
    parser.add_argument(
        "--config", default=None,
        help="Config file (dùng với --rerun)"
    )
    args = parser.parse_args()

    # Rerun inference nếu cần
    if args.rerun and args.config:
        rerun_inference(args.config)

    # Collect results
    project_root = Path(__file__).parent
    results_list = []
    for exp_path in args.exps:
        exp_dir = project_root / exp_path if not Path(exp_path).is_absolute() else Path(exp_path)
        r = load_results(exp_dir)
        if r:
            results_list.append(r)

    if not results_list:
        print("[INFO] Chưa có kết quả. Hãy chạy training trước:")
        print("  python train.py --config configs/exp_raw.yaml")
        print("  python train.py --config configs/exp_enhanced.yaml")
        print("  python evaluate.py")
        sys.exit(0)

    # Print & save comparison
    output = print_comparison_table(results_list)
    results_root = project_root / args.results_dir
    results_root.mkdir(parents=True, exist_ok=True)
    if output:
        save_markdown_report(output, results_root)
