#!/usr/bin/env python3
"""
Unified script to calculate Subset Accuracy, Hamming Accuracy, and Per-class Accuracy
for both CNN (EXP 3) and Swin Transformer (EXP 6) on the Test Set.

Usage:
    # To run for CNN (EXP 3)
    PYTHONPATH=.venv/lib/python3.12/site-packages python3 calculate_accuracy.py --model cnn
    
    # To run for Swin (EXP 6)
    PYTHONPATH=.venv/lib/python3.12/site-packages python3 calculate_accuracy.py --model swin
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Add root directory to python path
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import ODIRDataset
from src.models import build_model
from src.transforms import get_transforms
from src.utils import LABELS, LABEL_NAMES


def get_args():
    parser = argparse.ArgumentParser(
        description="Tính toán độ chính xác Accuracy đa nhãn ODIR-5K"
    )
    parser.add_argument(
        "--model", type=str, default="cnn", choices=["cnn", "swin"],
        help="Kiến trúc mạng cần đánh giá: 'cnn' hoặc 'swin'"
    )
    parser.add_argument(
        "--threshold-mode", type=str, default="default", choices=["default", "optimal"],
        help="Chế độ ngưỡng chẩn đoán: 'default' (0.5 cho tất cả) hoặc 'optimal' (quét động tối ưu)"
    )
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Sử dụng thiết bị: {device}")

    # 1. Setup paths based on model type
    if args.model == "cnn":
        checkpoint_path = Path("results/exp_3_cnn_preprocess_with_aug/best.pth")
        config_path = Path("results/exp_3_cnn_preprocess_with_aug/config.yaml")
        results_json_path = Path("results/exp_3_cnn_preprocess_with_aug/test_results.json")
    else:
        checkpoint_path = Path("results/exp_6_swin_preprocess_with_aug/best.pth")
        config_path = Path("results/exp_6_swin_preprocess_with_aug/config.yaml")
        results_json_path = Path("results/exp_6_swin_preprocess_with_aug/test_results.json")

    if not checkpoint_path.exists():
        print(f"[ERROR] Không tìm thấy tệp trọng số (checkpoint) tại: {checkpoint_path}")
        if args.model == "swin":
            print("  → Bạn hãy tải file 'best.pth' từ thư mục output của Kaggle Swin (EXP 6) về đặt vào thư mục:")
            print(f"    {checkpoint_path.parent}/")
        return
        
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        
    model_type = cfg.get("model_type", "cnn") if args.model == "cnn" else "swin"
    img_size = cfg["training"]["img_size"]
    batch_size = cfg["training"]["batch_size"]
    
    splits_dir = Path(cfg.get("splits_dir", "archive/splits_clean"))
    img_dir = Path("archive/enhanced_images")
    metadata_path = splits_dir / "metadata.json"
    
    with open(metadata_path) as f:
        meta = json.load(f)
    age_mean = meta["age_stats"]["mean"]
    age_std = meta["age_stats"]["std"]
    
    # 2. Get Test DataLoader
    print(f"[Data] Đang chuẩn bị Test DataLoader (ảnh enhanced)...")
    tf_val = get_transforms("val", img_size)
    test_dataset = ODIRDataset(
        csv_path=str(splits_dir / "test.csv"),
        img_dir=str(img_dir),
        transforms=tf_val,
        age_mean=age_mean,
        age_std=age_std
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # 3. Load Model
    print(f"[Model] Đang nạp mô hình {model_type.upper()}...")
    model = build_model(
        model_type=model_type,
        pretrained=False,
        img_size=img_size,
        variant=cfg.get("model", {}).get("variant", "tiny")
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    
    # 4. Handle thresholds
    thresholds = [0.5] * 8
    if args.threshold_mode == "optimal":
        if results_json_path.exists():
            try:
                with open(results_json_path) as f:
                    tr_data = json.load(f)
                if "optimal_thresholds" in tr_data:
                    thresholds = tr_data["optimal_thresholds"]
                    print(f"[Thresholds] Nạp ngưỡng tối ưu động: {[round(t, 2) for t in thresholds]}")
            except Exception:
                print("[WARN] Lỗi khi đọc test_results.json, chuyển về dùng ngưỡng mặc định [0.5].")
        else:
            print("[INFO] Không tìm thấy test_results.json, sử dụng ngưỡng mặc định [0.5].")
    else:
        print("[Thresholds] Sử dụng ngưỡng mặc định [0.5] cho tất cả các lớp.")
    
    # 5. Inference
    print(f"[Inference] Đang chạy dự đoán trên tập kiểm thử độc lập (Test Set gồm {len(test_dataset)} ảnh)...")
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            lbls = batch["labels"].numpy()
            
            outputs = model(imgs)
            probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
            
            all_probs.extend(probs)
            all_targets.extend(lbls)
            
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Apply thresholds
    thresh_np = np.array(thresholds)
    all_preds = (all_probs >= thresh_np).astype(int)
    
    # A. Subset Accuracy (Exact match across all 8 classes)
    subset_acc = np.all(all_preds == all_targets, axis=1).mean()
    
    # B. Hamming Accuracy (Average accuracy per binary prediction)
    hamming_acc = (all_preds == all_targets).mean()
    
    # C. Per-class Accuracy
    per_class_acc = {}
    for idx, label in enumerate(LABELS):
        per_class_acc[label] = (all_preds[:, idx] == all_targets[:, idx]).mean()
        
    # Output results
    print("\n" + "=" * 70)
    print(f"      KẾT QUẢ ĐỘ CHÍNH XÁC (ACCURACY) TRÊN TEST SET ({args.model.upper()})")
    print("=" * 70)
    print(f"  * Số lượng mẫu kiểm thử: {len(all_targets)}")
    print(f"  * Chế độ ngưỡng:         {args.threshold_mode.upper()} (Ngưỡng: {[round(t, 2) for t in thresholds]})")
    print("-" * 70)
    print(f"  🏆 Tỷ lệ đoán đúng từng bệnh (Hamming Acc):  {hamming_acc * 100:.2f}%")
    print(f"  🏆 Tỷ lệ khớp hoàn toàn cả 8 bệnh lý (Subset Acc):      {subset_acc * 100:.2f}%")
    print("-" * 70)
    print("  Độ chính xác (Accuracy) chi tiết từng bệnh:")
    for label in LABELS:
        name = LABEL_NAMES[label]
        acc = per_class_acc[label]
        print(f"    - {name:<32} ({label}): {acc * 100:.2f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
