# 🔬 ODIR-5K — Chẩn Đoán Bệnh Lý Nhãn Khoa Song Nhãn bằng Học Sâu

> **Đề tài tốt nghiệp:** Nghiên cứu và ứng dụng học sâu hỗ trợ chẩn đoán bệnh lý nhãn khoa và dự đoán tuổi sinh học từ ảnh đáy mắt  
> **Sinh viên:** Ngô Đình Đạt — Trường Đại học Thủy Lợi  
> **GVHD:** TS. Lê Thị Tú Kiên  

---

## 📋 Tổng Quan

Hệ thống sử dụng kiến trúc **Mạng Siamese Song Nhãn Đa Nhiệm** để phân tích đồng thời ảnh đáy mắt trái và mắt phải của bệnh nhân, thực hiện hai nhiệm vụ:

- **Phân loại nhị phân:** Bình thường (Normal) vs. Bệnh lý (Pathological)
- **Hồi quy tuổi sinh học:** Ước lượng tuổi võng mạc (Retinal Age) và tính chỉ số Retinal Age Gap

**Bộ dữ liệu:** [ODIR-5K](https://odir2019.grand-challenge.org/) — 3.343 bệnh nhân, 6.686 ảnh đáy mắt, bao gồm 7 nhóm bệnh lý: Diabetic Retinopathy (D), Glaucoma (G), Cataract (C), AMD (A), Hypertension (H), Myopia (M), Other (O).

---

## 🏗️ Kiến Trúc Hệ Thống

```
Mắt Trái [B,3,384,384]    Mắt Phải [B,3,384,384]
        │                           │
        ▼                           ▼
┌─────────────────────────────────────────┐
│        SHARED WEIGHTS BACKBONE          │
│  EfficientNet-B0 (CNN) / Swin-Tiny      │
└──────────┬──────────────────┬───────────┘
           │                  │
    Feat_L [B,D]        Feat_R [B,D]
           │                  │
     (×~missing_mask)   (×~missing_mask)
           └──────────┬───────┘
                      │ Concat [B, 2D]
                      ▼
             ┌────────────────┐
             │   Fusion MLP   │ → 512 chiều
             │ (LayerNorm+SiLU│
             │   + Dropout)   │
             └───────┬────────┘
                ┌────┴────┐
                ▼         ▼
          CLS Head    REG Head
         (Logits)   (Age Z-score)
```

**2 Backbone được so sánh:**
| Backbone | Params | Feature Dim |
|---|---|---|
| EfficientNet-B0 (CNN) | ~5.3M | 1280 |
| Swin Transformer-Tiny | ~28M | 768 |

---

## 🔬 Pipeline Tiền Xử Lý Ảnh

```
Ảnh gốc → ROI Crop (512×512) → Ben Graham Normalization → CLAHE → Model Input (384×384)
```

1. **ROI Crop:** Loại bỏ viền đen, chuẩn hóa kích thước 512×512
2. **Ben Graham:** Trừ Gaussian Blur local + cộng 128 → chuẩn hóa ánh sáng không đồng đều
3. **CLAHE:** Tăng tương phản cục bộ trên kênh L (không gian LAB)

---

## 📊 Kết Quả Ablation Study

| EXP | Kiến trúc | Ảnh | Aug | Accuracy | AUC-ROC | F1 | Sensitivity | Specificity | Age MAE |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| EXP 1 | EfficientNet-B0 | Raw | ❌ | 0.6614 | 0.7546 | 0.7099 | 0.6118 | 0.7654 | 8.02y |
| EXP 2 | EfficientNet-B0 | Enhanced | ❌ | 0.6693 | 0.7911 | 0.6993 | 0.5676 | 0.8827 | 7.92y |
| EXP 3 | EfficientNet-B0 | Enhanced | ✅ | 0.6912 | 0.7840 | 0.7446 | 0.6647 | 0.7469 | 7.74y |
| EXP 4 | Swin-Tiny | Raw | ❌ | 0.7311 | 0.8200 | 0.7900 | 0.7471 | 0.6975 | 7.41y |
| **EXP 5** ⭐ | **Swin-Tiny** | **Enhanced** | **❌** | **0.7629** | **0.8563** | **0.8046** | **0.7206** | **0.8519** | **7.96y** |
| EXP 6 | Swin-Tiny | Enhanced | ✅ | 0.7291 | 0.8012 | 0.7695 | 0.6676 | 0.8580 | 7.58y |

> ⭐ **Model tốt nhất:** EXP 5 (Swin-Tiny + Enhanced) — AUC-ROC = **0.856**, Accuracy = **76.3%**

**Phân chia dữ liệu:** Train 2.340 / Val 501 / Test 502 bệnh nhân (theo Patient ID, không bị data leakage)

---

## 🗂️ Cấu Trúc Thư Mục

```
DOANTOTNGHIEP/
├── train.py                    # Entry-point huấn luyện (CLI)
├── evaluate.py                 # So sánh 6 thực nghiệm → Ablation Study
│
├── configs/                    # YAML config cho 6 thực nghiệm
│   ├── exp_1_cnn_binary_raw.yaml
│   ├── exp_2_cnn_binary_enhanced.yaml
│   ├── exp_3_cnn_binary_enhanced_aug.yaml
│   ├── exp_4_swin_binary_raw.yaml
│   ├── exp_5_swin_binary_enhanced.yaml
│   └── exp_6_swin_binary_enhanced_aug.yaml
│
├── src/                        # Logic Deep Learning
│   ├── config.py               # Load YAML, set_seed
│   ├── dataset.py              # BinocularDataset (ghép cặp theo Patient ID)
│   ├── transforms.py           # Albumentations augmentation
│   ├── augment.py              # Binocular MixUp/CutMix đồng bộ
│   ├── losses.py               # BinaryFocalLoss + MultiTaskLoss
│   ├── metrics.py              # AUC/F1/Sensitivity/Specificity + Youden
│   ├── engine.py               # Training loop + Early Stopping
│   └── models/
│       ├── backbone.py         # EfficientNet-B0 / Swin-Tiny (via timm)
│       └── siamese.py          # BinocularClassifier (Siamese architecture)
│
├── scripts/                    # Tiền xử lý dữ liệu (chạy 1 lần)
│   ├── preprocess_enhance.py   # ROI Crop → Ben Graham → CLAHE
│   └── build_patient_splits.py # Chia Train/Val/Test theo Patient ID
│
├── webapp/                     # Ứng dụng web Streamlit
│   ├── app.py                  # Giao diện chính
│   └── inference.py            # Pipeline inference + Grad-CAM
│
├── notebooks/
│   └── odir5k_binocular_kaggle.ipynb  # Notebook Kaggle
│
├── results/                    # Kết quả training
│   ├── exp_1_cnn_binary_raw/
│   │   ├── best_model.pth
│   │   ├── test_results.json
│   │   └── config.yaml
│   ├── ... (exp_2 đến exp_6)
│   └── comparison_table.md
│
└── archive/                    # Dữ liệu (không push lên Git)
    ├── ODIR-5K/.../Training Images/
    ├── enhanced_images/
    └── splits_clean/
        ├── train.csv
        ├── val.csv
        ├── test.csv
        └── metadata.json
```

---

## ⚙️ Thiết Kế Kỹ Thuật Nổi Bật

### 1. Xử lý thiếu mắt (Missing Eye Handling)
Bệnh nhân chỉ có 1 mắt → ảnh mắt thiếu được thay bằng **zero tensor**, kèm cờ `left_missing`/`right_missing`. Đặc trưng từ mắt thiếu được nhân với `(~missing).float()` trước khi concat, đảm bảo không có nhiễu.

### 2. Hàm mất mát đa nhiệm
```
L_total = L_focal + 0.05 × L_age
```
- `L_focal`: Binary Focal Loss với α tự động theo tỷ lệ lớp, γ=2.0
- `L_age`: Smooth L1 Loss trên tuổi Z-score

### 3. Chiến lược huấn luyện Two-Stage
- **Stage 1 (5 epoch):** Freeze backbone, chỉ train Fusion MLP + 2 head (LR=0.001)
- **Stage 2 (35 epoch):** Unfreeze toàn bộ (LR=0.0001), Early Stopping theo AUC-ROC val

### 4. Tăng cường dữ liệu đồng bộ song nhãn
MixUp/CutMix được áp dụng **đồng thời** cho cả cặp ảnh (mắt trái + mắt phải) với cùng tham số λ, đảm bảo tính nhất quán sinh học.

### 5. Hiệu chỉnh ngưỡng Youden Index
```
J = Sensitivity + Specificity - 1
```
Ngưỡng tối ưu được tìm trong một lượt validation, không cần inference lại.

---

## 🚀 Hướng Dẫn Sử Dụng

### Cài đặt môi trường
```bash
pip install torch torchvision timm albumentations opencv-python streamlit pandas scikit-learn
```

### Huấn luyện mô hình
```bash
# Dry-run kiểm tra lỗi (1 epoch)
PYTHONPATH=./ python train.py --config configs/exp_5_swin_binary_enhanced.yaml --dry-run

# Huấn luyện chính thức
PYTHONPATH=./ python train.py --config configs/exp_5_swin_binary_enhanced.yaml
```

### Đánh giá và so sánh
```bash
PYTHONPATH=./ python evaluate.py --exps results/exp_4_swin_binary_raw results/exp_5_swin_binary_enhanced results/exp_6_swin_binary_enhanced_aug
```

### Chạy Web App
```bash
streamlit run webapp/app.py
```

Truy cập: `http://localhost:8501`

---

## 🌐 Web Application

Ứng dụng Streamlit hỗ trợ:
- **Upload ảnh:** Tải ảnh đáy mắt trái/phải
- **Chọn model:** 6 thực nghiệm + 2 loại trọng số (best/last)
- **Kết quả:** Xác suất bệnh lý, tuổi võng mạc, Retinal Age Gap
- **Grad-CAM:** Bản đồ nhiệt trực quan hóa vùng bất thường
- **Debug mode:** Hỗ trợ URL params `?left_file=...&right_file=...`

---

## 📚 Công Nghệ Sử Dụng

| Thư viện | Mục đích |
|---|---|
| **PyTorch** | Framework học sâu chính |
| **timm** | EfficientNet-B0, Swin-Tiny pretrained |
| **Albumentations** | Tăng cường dữ liệu ảnh |
| **OpenCV** | Tiền xử lý ROI, Ben Graham, CLAHE |
| **scikit-learn** | Tính AUC-ROC, Youden Index |
| **Streamlit** | Web app demo |
| **Pandas** | Quản lý dataset splits |

---

## 📖 Tài Liệu Tham Khảo

- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for CNNs. *ICML*.
- Liu, Z., et al. (2021). Swin Transformer: Hierarchical Vision Transformer. *ICCV*.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *ICCV*.
- Zhang, H., et al. (2018). MixUp: Beyond Empirical Risk Minimization. *ICLR*.
- Yun, S., et al. (2019). CutMix: Training Strategy using Guide to Cut and Paste. *ICCV*.
- ODIR-5K Dataset: Ocular Disease Intelligent Recognition, Peking University.

---

## 📄 Giấy Phép

Dự án phục vụ mục đích học thuật và nghiên cứu.  
© 2026 Ngô Đình Đạt — Trường Đại học Thủy Lợi.
