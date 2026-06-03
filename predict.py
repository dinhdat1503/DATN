#!/usr/bin/env python3
"""
Single-image inference script for ODIR-5K Multi-task Learning.

Runs the complete pipeline:
  1. Image preprocessing (ROI Crop -> Ben Graham -> CLAHE)
  2. Image standardization (ImageNet normalization)
  3. MTL deep learning model inference (CNN / Swin)
  4. Outputs 8-class disease probabilities & predicted retinal age

Usage:
    python predict.py --image path/to/raw/image.jpg --checkpoint results/exp_3_cnn_preprocess_with_aug/best.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

# Add current directory to path to resolve imports
sys.path.insert(0, str(Path(__file__).parent))

from src.models import build_model
from src.utils import LABELS, LABEL_NAMES


def crop_image_from_gray(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """ROI Cropping: Removes black background border surrounding the eyeball."""
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # Image is too dark or empty
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img
    return img


def ben_graham_normalization(
    img: np.ndarray,
    sigma_ratio: float = 1 / 6,
    scale: int = 128,
) -> np.ndarray:
    """Ben Graham color normalization to align illumination and reduce device bias."""
    h, w = img.shape[:2]
    sigma = int(max(h, w) * sigma_ratio)
    if sigma % 2 == 0:
        sigma += 1
    local_avg = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigmaX=sigma)
    result = img.astype(np.float32) - local_avg + scale
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Applies CLAHE on the L channel in LAB color space to enhance blood vessels."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return result


def preprocess_image(
    image_path: str,
    img_size: int = 384,
    apply_ben: bool = True,
    apply_cl: bool = True,
) -> tuple[np.ndarray, torch.Tensor]:
    """Loads a raw fundus image, applies ROI cropping, Ben Graham, CLAHE, and standardizes."""
    # 1. Load image
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise FileNotFoundError(f"Không thể đọc file ảnh: {image_path}")

    # 2. ROI Cropping
    img = crop_image_from_gray(raw_img, tol=7)

    # 3. Ben Graham Normalization
    if apply_ben:
        img = ben_graham_normalization(img, sigma_ratio=1/6)

    # 4. CLAHE
    if apply_cl:
        img = apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))

    # Keep a copy of preprocessed BGR image for save/display if needed
    prep_bgr = cv2.resize(img, (img_size, img_size))

    # 5. Standardization for PyTorch
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(prep_bgr, cv2.COLOR_BGR2RGB)
    
    # Scale to [0, 1]
    img_float = img_rgb.astype(np.float32) / 255.0

    # ImageNet Mean & Std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_float - mean) / std

    # Transpose to Channel-First [3, H, W]
    img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1))
    
    # Add Batch Dimension [1, 3, H, W]
    img_tensor = img_tensor.unsqueeze(0)

    return prep_bgr, img_tensor


def draw_prob_bar(prob: float, width: int = 20) -> str:
    """Helper to draw a visual probability bar on console."""
    filled_len = int(round(width * prob))
    bar = "█" * filled_len + "░" * (width - filled_len)
    return bar


def main():
    parser = argparse.ArgumentParser(
        description="Dự đoán đa nhiệm ODIR-5K từ 1 ảnh võng mạc thô"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Đường dẫn tới ảnh võng mạc thô cần test (jpg/png)"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="results/exp_3_cnn_preprocess_with_aug/best.pth",
        help="Đường dẫn file trọng số checkpoint (.pth)"
    )
    parser.add_argument(
        "--config", type=str,
        default="results/exp_3_cnn_preprocess_with_aug/config.yaml",
        help="Đường dẫn file cấu hình YAML (.config hoặc .yaml)"
    )
    parser.add_argument(
        "--save-prep", type=str, default="",
        help="Đường dẫn để lưu ảnh sau khi tiền xử lý (nếu muốn xem kết quả)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Sử dụng thiết bị: {device}")

    # 1. Load config
    config_path = Path(args.config)
    if not config_path.exists():
        # Fallback to look in root configs/
        config_path = Path("configs/exp_3_cnn_preprocess_with_aug.yaml")

    print(f"[Config] Đang tải cấu hình từ: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_type = cfg.get("model_type", "cnn")
    img_size = cfg.get("training", {}).get("img_size", 384)
    print(f"[Model] Kiến trúc: {model_type.upper()} | Kích thước ảnh đầu vào: {img_size}x{img_size}")

    # 2. Preprocess raw image
    print(f"[Data] Đang thực hiện Tiền xử lý (ROI Crop -> Ben Graham -> CLAHE) cho: {args.image}...")
    try:
        prep_bgr, img_tensor = preprocess_image(
            image_path=args.image,
            img_size=img_size,
            apply_ben=True,
            apply_cl=True
        )
        print("  → Tiền xử lý hoàn tất thành công.")
    except Exception as e:
        print(f"[ERROR] Tiền xử lý thất bại: {e}")
        sys.exit(1)

    if args.save_prep:
        cv2.imwrite(args.save_prep, prep_bgr)
        print(f"  → Lưu ảnh tiền xử lý tại: {args.save_prep}")

    # 3. Build & Load model
    print(f"[Model] Đang khởi tạo mô hình và nạp trọng số từ: {args.checkpoint}...")
    model = build_model(
        model_type=model_type,
        pretrained=False,
        img_size=img_size,
    )
    
    try:
        ckpt = torch.load(args.checkpoint, map_location=device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt)
        model = model.to(device)
        model.eval()
        print("  → Nạp trọng số mô hình thành công.")
    except Exception as e:
        print(f"[ERROR] Không thể nạp checkpoint: {e}")
        sys.exit(1)

    # 4. Load metadata for un-normalizing age & loading thresholds
    age_mean = 58.141
    age_std = 11.2583
    thresholds = [0.5] * 8

    # Attempt to load metadata from archive/splits_clean/metadata.json
    splits_clean_dir = Path(cfg.get("splits_dir", "archive/splits_clean"))
    metadata_path = splits_clean_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                meta = json.load(f)
            age_mean = meta["age_stats"]["mean"]
            age_std = meta["age_stats"]["std"]
            print(f"[Metadata] Nạp thống kê tuổi: mean={age_mean:.2f}, std={age_std:.2f}")
        except Exception:
            pass

    # Attempt to find optimal thresholds from test_results.json
    ckpt_dir = Path(args.checkpoint).parent
    test_results_path = ckpt_dir / "test_results.json"
    if test_results_path.exists():
        try:
            with open(test_results_path) as f:
                tr_data = json.load(f)
            if "optimal_thresholds" in tr_data:
                thresholds = tr_data["optimal_thresholds"]
                print(f"[Thresholds] Phát hiện ngưỡng tối ưu động từ test_results: {[round(t, 2) for t in thresholds]}")
        except Exception:
            pass
    else:
        # Check if thresholds are printed or fallback
        print(f"[Thresholds] Sử dụng ngưỡng mặc định [0.5] cho tất cả các lớp.")

    # 5. Model inference
    print("\n[Inference] Đang chạy dự đoán thô...")
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        logits = outputs["logits"].cpu().squeeze(0)  # Shape [8]
        age_pred_norm = outputs["age_pred"].cpu().item()   # Float scalar

    # Apply Sigmoid to logits to get probabilities
    probs = torch.sigmoid(logits).numpy()

    # Un-normalize Age
    pred_age = age_pred_norm * age_std + age_mean

    # 6. Output beautiful formatted results
    print("\n" + "=" * 70)
    print("                KẾT QUẢ CHẨN ĐOÁN VÕNG MẠC ĐA NHIỆM ODIR-5K")
    print("=" * 70)
    print(f"  Tệp tin kiểm thử:  {Path(args.image).name}")
    print(f"  Mô hình sử dụng:   {model_type.upper()}")
    print(f"  Thiết bị chạy:     {device.type.upper()}")
    print("-" * 70)
    print("  [A] CHẨN ĐOÁN 8 LỚP BỆNH LÝ VÕNG MẠC:")
    print("  " + "─" * 66)
    
    for idx, lbl in enumerate(LABELS):
        prob = probs[idx]
        thresh = thresholds[idx]
        decision = "DƯƠNG TÍNH (YES)" if prob >= thresh else "Âm tính (No)"
        tag = "🔴" if prob >= thresh else "🟢"
        if lbl == "N" and prob >= thresh:
            tag = "🟢"  # Normal positive is healthy
            decision = "BÌNH THƯỜNG (HEALTHY)"
        elif lbl == "N" and prob < thresh:
            tag = "🟡"
            decision = "Có dấu hiệu bất thường"

        bar = draw_prob_bar(prob, width=15)
        lbl_full = LABEL_NAMES[lbl]
        
        print(f"  {tag} {lbl_full:<32} {bar} {prob*100:6.2f}% (Ngưỡng: {thresh:.2f}) → {decision}")

    print("-" * 70)
    print("  [B] ƯỚC LƯỢNG ĐỘ TUỔI VÕNG MẠC:")
    print(f"  🕒 Dự đoán Retinal Age: {pred_age:.1f} tuổi (Giá trị normalized: {age_pred_norm:+.4f})")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
