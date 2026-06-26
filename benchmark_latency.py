"""
Script đo đạc hiệu năng định lượng (Benchmark Latency & Resource) cho Chương 4 - Mục 4.6.
Đo thời gian chạy tiền xử lý và suy luận của EfficientNet-B0 vs Swin-Tiny trên CPU.

Cách chạy:
    d:\\DOANTOTNGHIEP\\DOANTOTNGHIEP\\.venv_win\\Scripts\\python.exe benchmark_latency.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np
import torch

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from webapp.inference import crop_roi, ben_graham, apply_clahe, prepare_tensor
from src.models import build_model

def benchmark():
    print("================================================================")
    print(" BẮT ĐẦU ĐO HIỆU NĂNG HỆ THỐNG (BENCHMARK LATENCY)")
    print("================================================================")

    # 1. Tạo ảnh dummy giả lập ảnh đáy mắt (1000 x 1000)
    print("📷 Đang khởi tạo ảnh giả lập (1000x1000)...")
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    # Vẽ hình tròn màu cam ở giữa giả lập nhãn cầu đáy mắt
    cv2_cv = None
    try:
        import cv2
        cv2.circle(img, (500, 500), 450, (15, 75, 150), -1)
        # Thêm một ít nhiễu/chi tiết
        cv2.circle(img, (600, 450), 30, (80, 200, 240), -1)
    except Exception:
        # Fallback nếu không có cv2
        pass

    # 2. Đo thời gian tiền xử lý ảnh (100 lần chạy để lấy trung bình)
    print("\n⏳ 1. Đo thời gian các bước Tiền xử lý ảnh (100 lần chạy)...")
    n_runs = 100
    
    t_crop = []
    t_bg = []
    t_clahe = []
    t_total_pre = []

    for _ in range(n_runs):
        # Đo Crop ROI
        t0 = time.time()
        cropped = crop_roi(img)
        t_crop.append((time.time() - t0) * 1000) # ms

        # Đo Ben Graham
        t0 = time.time()
        bg = ben_graham(cropped)
        t_bg.append((time.time() - t0) * 1000) # ms

        # Đo CLAHE
        t0 = time.time()
        enh = apply_clahe(bg)
        t_clahe.append((time.time() - t0) * 1000) # ms

        # Đo tổng thể
        t0 = time.time()
        _ = apply_clahe(ben_graham(crop_roi(img)))
        t_total_pre.append((time.time() - t0) * 1000) # ms

    print(f"   - Crop ROI: {np.mean(t_crop):.2f} ± {np.std(t_crop):.2f} ms")
    print(f"   - Ben Graham Normalization: {np.mean(t_bg):.2f} ± {np.std(t_bg):.2f} ms")
    print(f"   - CLAHE: {np.mean(t_clahe):.2f} ± {np.std(t_clahe):.2f} ms")
    print(f"   👉 Tổng tiền xử lý (1 mắt): {np.mean(t_total_pre):.2f} ms")
    print(f"   👉 Tổng tiền xử lý (song nhãn - 2 mắt): {np.mean(t_total_pre)*2:.2f} ms")

    # 3. Đo thời gian Model Forward Pass trên CPU (10 lần chạy ấm máy + 20 lần đo)
    print("\n⏳ 2. Đo thời gian Mô hình Forward Pass (CPU)...")
    device = torch.device("cpu")
    
    # Chuẩn bị Tensor đầu vào
    left_tensor = prepare_tensor(img).to(device)
    right_tensor = prepare_tensor(img).to(device)
    left_missing = torch.tensor([False], dtype=torch.bool).to(device)
    right_missing = torch.tensor([False], dtype=torch.bool).to(device)
    missing_one_eye = torch.tensor([True], dtype=torch.bool).to(device) # Cho ca khuyết 1 mắt
    zero_tensor = torch.zeros(1, 3, 384, 384).to(device)

    # 2.1. Đánh giá EfficientNet-B0 (CNN)
    print("   [Mô hình CNN Siamese - EfficientNet-B0]")
    model_cnn = build_model("cnn", pretrained=False, img_size=384).to(device)
    model_cnn.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model_cnn(left_tensor, right_tensor, left_missing, right_missing)

    t_cnn_full = []
    t_cnn_masked = []
    with torch.no_grad():
        for _ in range(20):
            t0 = time.time()
            _ = model_cnn(left_tensor, right_tensor, left_missing, right_missing)
            t_cnn_full.append((time.time() - t0) * 1000) # ms
            
            t0 = time.time()
            _ = model_cnn(left_tensor, zero_tensor, left_missing, missing_one_eye)
            t_cnn_masked.append((time.time() - t0) * 1000) # ms

    print(f"   - Đầy đủ 2 mắt: {np.mean(t_cnn_full):.2f} ± {np.std(t_cnn_full):.2f} ms")
    print(f"   - Khuyết 1 mắt (Zero Masking): {np.mean(t_cnn_masked):.2f} ± {np.std(t_cnn_masked):.2f} ms")

    # 2.2. Đánh giá Swin-Tiny (Transformer)
    print("   [Mô hình Transformer Siamese - Swin-Tiny]")
    model_swin = build_model("swin", pretrained=False, img_size=384).to(device)
    model_swin.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model_swin(left_tensor, right_tensor, left_missing, right_missing)

    t_swin_full = []
    t_swin_masked = []
    with torch.no_grad():
        for _ in range(20):
            t0 = time.time()
            _ = model_swin(left_tensor, right_tensor, left_missing, right_missing)
            t_swin_full.append((time.time() - t0) * 1000) # ms
            
            t0 = time.time()
            _ = model_swin(left_tensor, zero_tensor, left_missing, missing_one_eye)
            t_swin_masked.append((time.time() - t0) * 1000) # ms

    print(f"   - Đầy đủ 2 mắt: {np.mean(t_swin_full):.2f} ± {np.std(t_swin_full):.2f} ms")
    print(f"   - Khuyết 1 mắt (Zero Masking): {np.mean(t_swin_masked):.2f} ± {np.std(t_swin_masked):.2f} ms")

    # 4. In bảng tóm tắt lý thuyết cho GPU (ước tính dựa trên benchmarks tiêu chuẩn của Tesla T4)
    print("\n" + "="*64)
    print(" BẢNG THAM KHẢO HIỆU NĂNG HỆ THỐNG (ĐỊNH LƯỢNG)")
    print("="*64)
    print(f"| Giai đoạn hệ thống | Tác vụ | Latency trên CPU (Đo được) | Latency trên GPU T4 (Tham khảo) |")
    print(f"|:---|:---|:---:|:---:|")
    print(f"| **Tiền xử lý** | ROI Crop (1 ảnh) | {np.mean(t_crop):.1f} ms | ~3-5 ms |")
    print(f"| | Ben Graham (1 ảnh) | {np.mean(t_bg):.1f} ms | ~15-20 ms |")
    print(f"| | CLAHE (1 ảnh) | {np.mean(t_clahe):.1f} ms | ~1-2 ms |")
    print(f"| | **Tổng tiền xử lý song nhãn (2 ảnh)** | **{np.mean(t_total_pre)*2:.1f} ms** | **~40-50 ms** |")
    print(f"| **Suy luận (Inference)** | EfficientNet-B0 (Đầy đủ 2 mắt) | {np.mean(t_cnn_full):.1f} ms | ~10-15 ms |")
    print(f"| | EfficientNet-B0 (Khuyết 1 mắt) | {np.mean(t_cnn_masked):.1f} ms | ~10-15 ms |")
    print(f"| | Swin-Tiny (Đầy đủ 2 mắt) | {np.mean(t_swin_full):.1f} ms | ~20-30 ms |")
    print(f"| | Swin-Tiny (Khuyết 1 mắt) | {np.mean(t_swin_masked):.1f} ms | ~20-30 ms |")
    print("="*64)

if __name__ == "__main__":
    benchmark()
