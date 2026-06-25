"""
So sánh kết quả các thực nghiệm ODIR-5K Phase 1 → sinh bảng Ablation Study.

Đọc file results/<exp>/test_results.json của các thực nghiệm và lập bảng markdown
so sánh (Accuracy, AUC, F1, Sensitivity, Specificity, Age MAE) ở ngưỡng tối ưu Youden.

Cách dùng:
    python evaluate.py
    python evaluate.py --results-dir results
    python evaluate.py --exps results/exp_3_cnn_binary_enhanced_aug results/exp_6_swin_binary_enhanced_aug
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Thứ tự hiển thị mặc định 6 thực nghiệm
DEFAULT_ORDER = [
    "exp_1_cnn_binary_raw",
    "exp_2_cnn_binary_enhanced",
    "exp_3_cnn_binary_enhanced_aug",
    "exp_4_swin_binary_raw",
    "exp_5_swin_binary_enhanced",
    "exp_6_swin_binary_enhanced_aug",
]

LABELS = {
    "exp_1_cnn_binary_raw": ("EfficientNet-B0", "raw (ảnh gốc)", "❌"),
    "exp_2_cnn_binary_enhanced": ("EfficientNet-B0", "enhanced", "❌"),
    "exp_3_cnn_binary_enhanced_aug": ("EfficientNet-B0", "enhanced", "✅"),
    "exp_4_swin_binary_raw": ("Swin-Tiny", "raw (ảnh gốc)", "❌"),
    "exp_5_swin_binary_enhanced": ("Swin-Tiny", "enhanced", "❌"),
    "exp_6_swin_binary_enhanced_aug": ("Swin-Tiny", "enhanced", "✅"),
}


def load_result(exp_dir: Path) -> dict | None:
    """Đọc test_results.json của một thực nghiệm (None nếu chưa có)."""
    path = exp_dir / "test_results.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_table(exp_dirs: list[Path]) -> str:
    """Sinh bảng markdown so sánh từ danh sách thư mục thực nghiệm."""
    header = (
        "# BẢNG SO SÁNH ABLATION — ODIR-5K Phase 1 (Nhị phân Siamese song nhãn)\n\n"
        "Chỉ số trên **tập Test** ở **ngưỡng tối ưu Youden** (tìm trên Validation). "
        "Nhãn: 0 = Normal, 1 = Pathological.\n\n"
        "| EXP | Kiến trúc | Ảnh | Aug | Accuracy | AUC-ROC | F1 | Sensitivity | Specificity | Age MAE |\n"
        "| :--: | :-- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |\n"
    )
    rows = []
    for d in exp_dirs:
        name = d.name
        res = load_result(d)
        arch, img, aug = LABELS.get(name, (name, "?", "?"))
        if res is None:
            rows.append(f"| {name} | {arch} | {img} | {aug} | *chưa chạy* | — | — | — | — | — |")
            continue
        m = res["metrics_threshold_optimal"]
        rows.append(
            f"| {name.replace('_binary', '').replace('exp_', 'EXP ')} | {arch} | {img} | {aug} "
            f"| {m['test_accuracy']:.4f} | {m['test_auc']:.4f} | {m['test_f1']:.4f} "
            f"| {m['test_sensitivity']:.4f} | {m['test_specificity']:.4f} | {m['test_age_mae']:.2f}y |"
        )
    return header + "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="So sánh kết quả Ablation Study Phase 1")
    parser.add_argument("--results-dir", default="results", help="Thư mục chứa kết quả các EXP")
    parser.add_argument("--exps", nargs="*", default=None, help="Danh sách thư mục EXP cụ thể")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if args.exps:
        exp_dirs = [Path(e) for e in args.exps]
    else:
        exp_dirs = []
        for name in DEFAULT_ORDER:
            d = results_dir / name
            if d.exists():
                exp_dirs.append(d)
        if not exp_dirs:  # fallback: mọi thư mục con có test_results.json
            exp_dirs = sorted(p.parent for p in results_dir.glob("*/test_results.json"))

    table = build_table(exp_dirs)
    out_path = results_dir / "comparison_table.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table)

    print(table)
    print(f"\n→ Đã lưu bảng so sánh: {out_path}")


if __name__ == "__main__":
    main()
