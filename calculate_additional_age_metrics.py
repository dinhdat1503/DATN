"""
Script tính toán thêm các chỉ số đánh giá tuổi sinh học (RMSE, R², Correlation, Mean Bias, v.v.)
cho tất cả 6 thực nghiệm trên tập Test.

Cách chạy:
    d:\\DOANTOTNGHIEP\\DOANTOTNGHIEP\\.venv_win\\Scripts\\python.exe calculate_additional_age_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Thêm thư mục hiện tại vào PATH để import được src
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import load_config, resolve_splits_dir, resolve_img_dir
from src.dataset import BinocularDataset
from src.models import build_model
from src.transforms import get_transforms

def main():
    print("================================================================")
    print(" BẮT ĐẦU TÍNH TOÁN CÁC CHỈ SỐ TUỔI SINH HỌC BỔ SUNG")
    print("================================================================")

    # 1. Đọc metadata để lấy mean/std của tuổi dùng cho giải chuẩn hóa
    meta_path = project_root / "archive" / "splits_clean" / "metadata.json"
    if not meta_path.exists():
        print(f"❌ Không tìm thấy file metadata tại {meta_path}")
        return
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    age_mean = meta["age_stats"]["mean"]
    age_std = meta["age_stats"]["std"]
    print(f"📊 Thông kê tuổi chuẩn hóa: Mean = {age_mean:.4f}, Std = {age_std:.4f}")

    # Danh sách 6 thực nghiệm cần tính toán
    exps = [
        "exp_1_cnn_binary_raw",
        "exp_2_cnn_binary_enhanced",
        "exp_3_cnn_binary_enhanced_aug",
        "exp_4_swin_binary_raw",
        "exp_5_swin_binary_enhanced",
        "exp_6_swin_binary_enhanced_aug",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Chạy trên thiết bị: {device}\n")

    summary_rows = []

    for exp in exps:
        exp_dir = project_root / "results" / exp
        config_path = exp_dir / "config.yaml"
        model_path = exp_dir / "best_model.pth"

        if not config_path.exists() or not model_path.exists():
            print(f"⚠️ Thực nghiệm {exp} không đầy đủ (thiếu config hoặc model) -> Bỏ qua.")
            continue

        print(f"🔍 Đang đánh giá {exp}...")
        cfg = load_config(config_path)

        # Cấu hình dữ liệu
        splits_dir = resolve_splits_dir(cfg, project_root)
        img_dir = resolve_img_dir(cfg, project_root)
        img_size = cfg["training"]["img_size"]

        # Dataset & Dataloader trên tập Test
        test_transforms = get_transforms("val", img_size=img_size)
        test_dataset = BinocularDataset(
            csv_path=splits_dir / "test.csv",
            img_dir=img_dir,
            transforms=test_transforms,
            img_size=img_size,
            age_mean=age_mean,
            age_std=age_std,
            age_min_filter=cfg["training"].get("age_min_filter", 5),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg["training"].get("batch_size", 8),
            shuffle=False,
            num_workers=0, # Chạy an toàn trên Windows
        )

        # Tạo model và load trọng số
        model_type = cfg.get("model_type", "cnn")
        model = build_model(
            model_type=model_type,
            pretrained=False,
            img_size=img_size,
            dropout=cfg["model"].get("dropout", 0.3),
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model = model.to(device)
        model.eval()

        all_preds = []
        all_trues = []

        # Inference loop
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Predicting {exp}"):
                left_img = batch["left_image"].to(device)
                right_img = batch["right_image"].to(device)
                left_missing = batch["left_missing"].to(device)
                right_missing = batch["right_missing"].to(device)
                
                # Forward pass
                outputs = model(left_img, right_img, left_missing, right_missing)
                age_pred = outputs["age_pred"].cpu().numpy().flatten()
                age_true = batch["age"].numpy().flatten()

                # Giải chuẩn hóa về năm ngay lập tức
                pred_years = age_pred * age_std + age_mean
                true_years = age_true * age_std + age_mean

                all_preds.extend(pred_years)
                all_trues.extend(true_years)

        # Tính toán các chỉ số hồi quy tuổi
        y_pred = np.array(all_preds)
        y_true = np.array(all_trues)
        errors = y_pred - y_true
        abs_errors = np.abs(errors)

        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        pearson = float(np.corrcoef(y_pred, y_true)[0, 1])
        
        # Hệ số R²
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - (ss_res / ss_tot))

        # Phân tích sai số chi tiết
        mean_bias = float(np.mean(errors))
        std_error = float(np.std(errors))
        min_error = float(np.min(errors))
        max_error = float(np.max(errors))
        p25 = float(np.percentile(errors, 25))
        p50 = float(np.percentile(errors, 50))
        p75 = float(np.percentile(errors, 75))

        metrics = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "pearson": pearson,
            "mean_bias": mean_bias,
            "std_error": std_error,
            "min_error": min_error,
            "max_error": max_error,
            "percentiles": {
                "p25": p25,
                "p50": p50,
                "p75": p75
            }
        }

        # Lưu kết quả
        out_json_path = exp_dir / "test_age_additional_metrics.json"
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"✅ Đã lưu kết quả bổ sung tại: {out_json_path}")
        print(f"   MAE: {mae:.2f} y | RMSE: {rmse:.2f} y | R²: {r2:.4f} | Pearson r: {pearson:.4f}\n")

        summary_rows.append({
            "exp": exp.replace("_binary", "").replace("exp_", "EXP ").upper(),
            "backbone": model_type.upper(),
            "preprocess": cfg["img_dir"].split("/")[-1].replace("_images", ""),
            "aug": "Có" if cfg["augmentation"].get("use_mixup", False) or cfg["augmentation"].get("use_cutmix", False) else "Không",
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "pearson": pearson,
            "bias": mean_bias,
            "std": std_error
        })

    # In bảng tổng hợp dạng Markdown
    print("\n" + "="*80)
    print(" BẢNG TỔNG HỢP CHỈ SỐ DỰ ĐOÁN TUỔI SINH HỌC TRÊN TẬP TEST")
    print("="*80)
    print("| Thực nghiệm | Backbone | Tiền xử lý | Augment | MAE (năm) | RMSE (năm) | Hệ số R² | Pearson r | Mean Bias | Std Error |")
    print("|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in summary_rows:
        print(f"| {r['exp']} | {r['backbone']} | {r['preprocess']} | {r['aug']} | {r['mae']:.2f} | {r['rmse']:.2f} | {r['r2']:.4f} | {r['pearson']:.4f} | {r['bias']:.2f} | {r['std']:.2f} |")
    print("="*80)

if __name__ == "__main__":
    main()
