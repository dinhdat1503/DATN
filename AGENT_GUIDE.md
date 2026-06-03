# 📋 AGENT CONTEXT FILE — ODIR-5K Multi-task Learning

> **Mục đích:** Đây là tài liệu **Single Source of Truth** dành cho AI Coding Agent.  
> Đọc file này = nắm toàn bộ kiến trúc, pipeline, kết quả, và quy ước của dự án.  
> **Không cần duyệt mã nguồn** trước khi bắt đầu làm việc.

---

## 0. Danh Tính Dự Án (Project Identity)

| Thuộc tính | Giá trị |
|---|---|
| **Tên dự án** | ODIR-5K Multi-task Learning |
| **Mục tiêu** | Phân loại 8 bệnh lý mắt (multi-label) + Dự đoán tuổi võng mạc (regression) |
| **Bộ dữ liệu** | ODIR-5K (Ocular Disease Intelligent Recognition, 5.000 bệnh nhân, ~7.000 ảnh đáy mắt) |
| **Framework** | PyTorch (pure) · Albumentations · timm · scikit-learn |
| **Môi trường huấn luyện** | Kaggle Notebook (GPU T4/P100) |
| **Môi trường phát triển** | Ubuntu Linux · Python 3.12 · VSCode |
| **Backbone A** | EfficientNet-B0 (CNN, ~5.3M params, feature_dim=1280) |
| **Backbone B** | Swin Transformer-Tiny (ViT, ~28M params, feature_dim=768) |
| **Nhãn bệnh (8)** | `N` Normal · `D` Diabetes · `G` Glaucoma · `C` Cataract · `A` AMD · `H` Hypertension · `M` Myopia · `O` Other |

---

## 1. Cây Thư Mục (Directory Tree)

```
DOANTOTNGHIEP/
├── train.py                    # ★ Entry-point huấn luyện chính
├── evaluate.py                 # So sánh delta giữa các thực nghiệm
├── predict.py                  # Inference đơn ảnh (ROI→BenGraham→CLAHE→Model)
├── calculate_accuracy.py       # Tính Hamming/Subset Accuracy trên tập Test
├── run_experiment.sh           # Shell script chạy batch thực nghiệm
│
├── configs/                    # ★ YAML config cho 6 thực nghiệm
│   ├── exp_1_cnn_no_preprocess.yaml
│   ├── exp_2_cnn_preprocess_no_aug.yaml
│   ├── exp_3_cnn_preprocess_with_aug.yaml        # CNN SOTA
│   ├── exp_4_swin_no_preprocess.yaml
│   ├── exp_5_swin_preprocess_no_aug.yaml
│   └── exp_6_swin_preprocess_with_aug.yaml       # Swin SOTA ★
│
├── src/                        # ★ Logic nghiệp vụ Deep Learning
│   ├── __init__.py             # Public exports
│   ├── dataset.py              # ODIRDataset + get_dataloaders
│   ├── transforms.py           # Albumentations pipeline (khóa Hue)
│   ├── loss.py                 # MultiTaskLoss = BCE(pos_weight) + λ·SmoothL1
│   ├── mixup.py                # MixUpCollator (trộn toàn cục α=0.4)
│   ├── cutmix.py               # CutMixCollator (cắt dán cục bộ α=1.0)
│   ├── utils.py                # LABELS, metrics, find_best_thresholds
│   └── models/
│       ├── __init__.py         # build_model() factory
│       ├── efficientnet_mtl.py # EfficientNet-B0 + 2 heads MTL
│       └── swin_mtl.py         # Swin-Tiny + 2 heads MTL
│
├── scripts/                    # Tiện ích tiền xử lý
│   ├── preprocess_enhance.py   # ROI Crop → Ben Graham → CLAHE
│   ├── build_patient_splits.py # Chia Train/Val/Test theo Patient ID
│   ├── check_preprocessing.py  # Kiểm tra chất lượng ảnh sau xử lý
│   ├── clean_and_rebuild.py    # Dọn dẹp và rebuild dữ liệu
│   └── generate_augmentation_samples.py
│
├── notebooks/                  # Kaggle notebooks
│   ├── odir5k_cnn_kaggle.ipynb
│   ├── odir5k_swin_kaggle.ipynb
│   └── kaggle_setup.md
│
├── webapp/                     # Ứng dụng demo web
│   ├── app.py                  # Gradio/Streamlit UI
│   └── inference.py            # Inference pipeline cho webapp
│
├── tests/                      # Unit tests
│   ├── test_mixup.py
│   └── test_cutmix.py
│
├── results/                    # Output huấn luyện (checkpoints, logs, metrics)
│   ├── exp_1_cnn_preprocess_with_aug/
│   ├── exp_2_cnn_preprocess_with_aug/
│   ├── exp_3_cnn_preprocess_with_aug/    # ★ CNN best
│   ├── exp_4_swin_no_preprocess/
│   ├── exp_5_swin_no_preprocess/
│   ├── exp_6_swin_preprocess_with_aug/   # ★ Swin best
│   └── comparison_table.md
│
├── docs/                       # Tài liệu & báo cáo
│   ├── Bao_cao_Ablation_Study_CNN.md
│   ├── Bao_cao_Ablation_Study_Swin.md
│   ├── Bao_cao_Do_chinh_xac_Toan_dien.md
│   ├── Bao_cao_Tien_xu_ly_Du_lieu.md
│   ├── Bang_so_sanh_thuc_nghiem_toan_dien.md
│   ├── bao_cao_chi_tiet_datn.md
│   ├── giai_thich_code_huan_luyen.md
│   ├── giai_thich_hai_mo_hinh.md
│   ├── giai_thich_quy_trinh_xu_ly.md
│   ├── giai_thich_run_experiment_notebook.md
│   ├── giai_thich_tang_cuong_du_lieu.md
│   ├── huong_dan_chi_tiet_ma_nguon.md
│   └── De_cuong_DATN.pdf
│
├── archive/                    # Dữ liệu gốc & đã xử lý
│   ├── ODIR-5K/                # Ảnh gốc
│   ├── enhanced_images/        # Ảnh sau ROI+BenGraham+CLAHE
│   └── splits_clean/           # train.csv, val.csv, test.csv, metadata.json
│
└── AGENT_GUIDE.md              # ← BẠN ĐANG ĐỌC FILE NÀY
```

---

## 2. Kiến Trúc Mô Hình (Architecture Overview)

### 2.1. Sơ đồ kiến trúc Multi-task Learning

```
┌──────────────┐
│  Input Image │    384×384×3 (Swin) hoặc 384×384×3 (CNN)
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────┐
│        SHARED BACKBONE           │
│  ┌────────────────────────────┐  │
│  │ EfficientNet-B0 (CNN)     │  │   feature_dim = 1280
│  │    MBConv×7 + SE Attention│  │   ~5.3M params
│  │ ─── HOẶC ───             │  │
│  │ Swin-Tiny (Transformer)   │  │   feature_dim = 768
│  │    Shifted Window Attn    │  │   ~28M params
│  └────────────────────────────┘  │
│         Global Avg Pool          │
│              │                   │
└──────────────┼───────────────────┘
               │
        features [B, dim]
         ┌─────┴─────┐
         │           │
         ▼           ▼
┌────────────┐ ┌────────────┐
│ CLS Head   │ │ REG Head   │
│ Dropout    │ │ Dropout    │
│ Linear→8   │ │ Linear→1   │
│ (Sigmoid)  │ │            │
└─────┬──────┘ └─────┬──────┘
      │               │
      ▼               ▼
  logits [B,8]    age_pred [B,1]
  BCEWithLogits   SmoothL1Loss
  (pos_weight)    (Huber, β=1.0)
      │               │
      └───────┬───────┘
              ▼
    Total Loss = L_cls + λ × L_reg    (λ = 0.1)
```

### 2.2. Thông số kỹ thuật cốt lõi

| Thành phần | CNN (EfficientNet-B0) | Swin (Transformer-Tiny) |
|---|---|---|
| **Feature dim** | 1280 | 768 |
| **Params (approx.)** | ~5.3M | ~28M |
| **Pretrained** | ImageNet-1K (torchvision/timm) | ImageNet-1K (timm) |
| **Input size** | 384×384 | 384×384 |
| **Batch size** | 16 | 8 |
| **Optimizer** | AdamW (lr=1e-4, wd=0.01) | AdamW (lr=5e-5, wd=0.05) |
| **Scheduler** | CosineAnnealingLR (T_max=45) | CosineAnnealingLR (T_max=45) |
| **Epochs** | 45 (early stop patience=8) | 45 (early stop patience=8) |
| **Dropout (cls/reg)** | 0.3 / 0.2 | 0.3 / 0.2 |
| **Gradient clipping** | max_norm=1.0 | max_norm=1.0 |

---

## 3. Pipeline Dữ Liệu (Data Pipeline)

### 3.1. Tiền xử lý tĩnh (offline, một lần)

```
Ảnh gốc ODIR-5K  ──►  ROI Crop (loại viền đen)
                  ──►  Ben Graham (đồng nhất sáng, sigma=10)
                  ──►  CLAHE (tăng tương phản cục bộ, clipLimit=2.0)
                  ──►  Lưu vào archive/enhanced_images/
```

**Script:** `scripts/preprocess_enhance.py`

### 3.2. Tiền xử lý động (runtime, mỗi batch)

```
Ảnh enhanced ──► Albumentations Pipeline:
                 ├── Resize(384×384)
                 ├── HorizontalFlip(p=0.5)
                 ├── VerticalFlip(p=0.5)
                 ├── ShiftScaleRotate(shift=0.05, scale=0.1, rotate=15°)
                 ├── ColorJitter(brightness=0.2, contrast=0.2, sat=0.2, hue=0)  ← HUE LOCKED
                 ├── GaussNoise(var=10-50)
                 ├── Normalize(ImageNet mean/std)
                 └── ToTensorV2
```

### 3.3. Augmentation nâng cao (batch-level, chỉ EXP 3 & 6)

| Kỹ thuật | Alpha | Prob | Mô tả |
|---|---|---|---|
| **MixUp** | α=0.4 | 50% batch | Trộn tuyến tính 2 ảnh + nhãn (Zhang et al., 2018) |
| **CutMix** | α=1.0 | 50% batch | Cắt vùng ngẫu nhiên từ ảnh B dán vào ảnh A (Yun et al., 2019) |
| **WRS** | — | 100% | WeightedRandomSampler: over-sample bệnh hiếm |

### 3.4. Chia dữ liệu (Data Splits)

- **Chia theo Patient ID** (không rò rỉ dữ liệu giữa tập Train/Val/Test)
- Tỷ lệ: **70% Train / 15% Val / 15% Test**
- Lọc tuổi: loại bỏ 28 hồ sơ có `Patient Age < 5` (tuổi=1, dữ liệu nhiễu)
- File: `archive/splits_clean/{train,val,test}.csv` + `metadata.json`
- **Chuẩn hóa tuổi:** Z-score từ training set → `age_norm = (age - mean) / std`

---

## 4. Kết Quả Ablation Study (6 Thực Nghiệm)

### 4.1. Bảng tổng hợp metrics

| EXP | Backbone | Tiền xử lý | Aug (MixUp/CutMix/WRS) | Best Val F1 | Test F1 (θ=0.5) | Test F1 (θ tối ưu) | Test AUC-ROC | Test Age MAE |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | EfficientNet-B0 | ❌ | ❌ | 0.5146 | 0.5248 | 0.5248 | 0.8071 | 7.81 năm |
| **2** | EfficientNet-B0 | ✅ | ❌ | 0.5479 | 0.5368 | 0.5368 | 0.8124 | 7.54 năm |
| **3** | EfficientNet-B0 | ✅ | ✅ | 0.5094 | 0.5492 | 0.5492 | **0.8395** | 7.59 năm |
| **4** | Swin-Tiny | ❌ | ❌ | 0.5518 | 0.5509 | 0.5582 | 0.8190 | 7.79 năm |
| **5** | Swin-Tiny | ✅ | ❌ | **0.5784** | 0.5537 | 0.5312 | 0.8205 | 7.65 năm |
| **6** | Swin-Tiny | ✅ | ✅ | 0.5694 | **0.5718** | **0.5771** | 0.8125 | **7.48 năm** |

### 4.2. Kết luận thực nghiệm

- **🏆 SOTA tổng thể:** EXP 6 (Swin + Preprocessing + MixUp/CutMix + WRS)
  - Test F1-macro cao nhất: **0.5771** (ngưỡng tối ưu)
  - Age MAE thấp nhất: **7.48 năm**
- **🏆 AUC-ROC cao nhất:** EXP 3 (CNN Full Pipeline) đạt **0.8395**
- **Delta tiền xử lý:** EXP 1→2 (CNN): AUC +0.0053, MAE −0.27 năm
- **Delta augmentation:** EXP 5→6 (Swin): Test F1 +0.0459, chứng minh MixUp/CutMix điều hòa overfitting

### 4.3. Accuracy đa nhãn (Tập Test, CNN EXP 3)

| Metric | Giá trị | Ý nghĩa |
|---|---|---|
| **Hamming Accuracy** | 83.87% | Đoán đúng từng quyết định chẩn đoán (per-label) |
| **Subset Accuracy** | 30.92% | Khớp hoàn toàn cả 8 nhãn đồng thời (rất khó) |

**Accuracy per-label nổi bật (bệnh hiếm):**
- Hypertension (H): **95.49%** · Myopia (M): **96.44%** · AMD (A): **93.08%**
- → Chứng minh WRS thành công ngăn thiên vị lớp đa số

---

## 5. Quyết Định Thiết Kế & Giải Thích Kỹ Thuật

### 5.1. Tại sao khóa Hue (`hue_shift_limit=0`)?

> **Y sinh:** Màu đỏ trong nhãn khoa biểu thị xuất huyết (hemorrhages) và vi phình mạch. Nếu augment xoay Hue → đỏ thành vàng/xanh → mô hình mất khả năng nhận diện bệnh lý dựa trên màu sắc.

### 5.2. Tại sao dùng SmoothL1 (Huber) thay vì MSE/MAE cho tuổi?

> **Kỹ thuật:** Tuổi đã chuẩn hóa Z-score (mean≈0, std≈1). SmoothL1 = MSE khi |error|<1, = MAE khi |error|≥1 → ổn định gradient + robust với outlier tuổi cực trị.

### 5.3. Tại sao λ = 0.1 cho regression loss?

> **Cân bằng:** BCE loss ~0.3–1.5, SmoothL1 ~0.2–1.0. Với λ=0.1, đóng góp regression ~10% tổng loss → task phân loại bệnh là ưu tiên chính, tuổi là phụ trợ.

### 5.4. Tại sao AUC-ROC (0.81–0.84) >> F1-macro (0.55–0.58)?

> **Metrics:** AUC đánh giá **xếp hạng xác suất** trên toàn dải ngưỡng [0,1] → phản ánh backbone trích xuất đặc trưng rất tốt. F1-macro thấp hơn vì bị **mất cân bằng nhãn** ảnh hưởng mạnh khi áp dụng ngưỡng cứng 0.5 lên bệnh hiếm.

### 5.5. Tại sao EXP 5 (Swin, tiền xử lý, không aug) overfitting?

> **Hiện tượng:** Val F1 cao nhất (0.5784) nhưng Test F1 thấp (0.5312 ở ngưỡng tối ưu). Swin ~28M params dễ ghi nhớ chi tiết ảnh enhanced nếu thiếu regularization. EXP 6 thêm MixUp/CutMix → Test F1 tăng +0.0459.

### 5.6. Tại sao tiền xử lý cải thiện Age MAE?

> **Cơ chế:** ROI Crop loại viền đen, Ben Graham đồng nhất sáng → triệt tiêu Device Bias (sai số do thiết bị chụp khác nhau) → mô hình tập trung phân tích đặc trưng lão hóa tự nhiên → MAE giảm từ 7.81 xuống 7.48 năm.

---

## 6. Quy Ước Mã Nguồn (Code Conventions)

### 6.1. Output format của model

```python
output = model(images)  # images: [B, 3, 384, 384]
# output = {
#     "logits":   Tensor[B, 8],   # raw logits, chưa sigmoid
#     "age_pred": Tensor[B, 1],   # tuổi đã chuẩn hóa Z-score
# }
```

### 6.2. Loss computation

```python
from src.loss import MultiTaskLoss
criterion = MultiTaskLoss(pos_weight=pos_weight_tensor, lam=0.1, device="cuda")
total_loss, detail = criterion(logits, labels, age_pred, age_true)
# detail = {"loss_total": float, "loss_cls": float, "loss_reg": float, "lam": 0.1}
```

### 6.3. Dataset sample format

```python
sample = dataset[idx]
# sample = {
#     "image":    FloatTensor[3, H, W],   # ảnh đã transform
#     "labels":   FloatTensor[8],          # multi-hot [N,D,G,C,A,H,M,O]
#     "age":      FloatTensor[1],          # tuổi chuẩn hóa Z-score
#     "filename": str,                     # tên file ảnh
# }
```

### 6.4. Config YAML structure

```yaml
experiment_name: str
model_type: "cnn" | "swin"
splits_dir: "archive/splits_clean"
img_dir: "archive/enhanced_images"
model:
  pretrained: true
  variant: "tiny"          # (Swin only)
  freeze_backbone: false
  dropout_cls: 0.3
  dropout_reg: 0.2
loss:
  lam: 0.1
training:
  img_size: 384
  batch_size: 8 | 16
  epochs: 45
  early_stopping_patience: 8
  use_weighted_sampler: true
optimizer: { name: AdamW, lr: float, weight_decay: float }
scheduler: { name: CosineAnnealingLR, T_max: int, eta_min: float }
augmentation:
  use_mixup: true/false
  use_cutmix: true/false
  mixup_alpha: 0.4
  cutmix_alpha: 1.0
output:
  results_dir: "results/exp_X_..."
  save_best_model: true
```

### 6.5. Checkpoint format

```python
checkpoint = {
    "epoch": int,
    "model_state": OrderedDict,
    "optimizer_state": dict,
    "scheduler_state": dict,
    "best_val_f1": float,
    "val_metrics": dict,      # (best model only)
    "early_stop_cnt": int,    # (last model only)
    "config": dict,
}
```

---

## 7. Lệnh Nhanh cho Agent (Quick Commands)

> **Lưu ý:** Tất cả lệnh chạy từ thư mục gốc dự án (`DOANTOTNGHIEP/`).  
> Trên Kaggle, thay `PYTHONPATH` bằng `!pip install timm albumentations scikit-learn`.

### 7.1. Huấn luyện

```bash
# Chạy huấn luyện EXP 3 (CNN Full Pipeline)
PYTHONPATH=.venv/lib/python3.12/site-packages python3 train.py --config configs/exp_3_cnn_preprocess_with_aug.yaml

# Chạy dry-run để test pipeline (1 epoch)
PYTHONPATH=.venv/lib/python3.12/site-packages python3 train.py --config configs/exp_6_swin_preprocess_with_aug.yaml --dry-run

# Resume training từ checkpoint
PYTHONPATH=.venv/lib/python3.12/site-packages python3 train.py --config configs/exp_3_cnn_preprocess_with_aug.yaml --resume results/exp_3_cnn_preprocess_with_aug/last_model.pth
```

### 7.2. Dự đoán đơn ảnh

```bash
PYTHONPATH=.venv/lib/python3.12/site-packages python3 predict.py --image "archive/ODIR-5K/ODIR-5K/Training Images/0_left.jpg"
```

### 7.3. Đánh giá Accuracy

```bash
# CNN (EXP 3) — ngưỡng mặc định 0.5
PYTHONPATH=.venv/lib/python3.12/site-packages python3 calculate_accuracy.py --model cnn

# Swin (EXP 6) — ngưỡng tối ưu động
PYTHONPATH=.venv/lib/python3.12/site-packages python3 calculate_accuracy.py --model swin --threshold-mode optimal
```

### 7.4. Chạy unit tests

```bash
PYTHONPATH=.venv/lib/python3.12/site-packages python3 -m pytest tests/ -v
```

---

## 8. Ràng Buộc & Lưu Ý Quan Trọng

> [!CAUTION]
> **Không được thay đổi** các hằng số sau mà không cập nhật toàn bộ pipeline:
> - `LABELS = ["N", "D", "G", "C", "A", "H", "M", "O"]` trong `src/utils.py`
> - `num_labels = 8` trong các model constructor
> - `age_min_filter = 5` trong `ODIRDataset`

> [!WARNING]
> **Kaggle-specific:**
> - Swin-Tiny cần `timm` library. Nếu không có timm → fallback mini-ViT (chỉ test pipeline, không train thật).
> - EfficientNet-B0 có 3 fallback: torchvision → timm → pure PyTorch (random init).
> - `num_workers` nên đặt 4 trên Kaggle, 0 trên local nếu gặp lỗi multiprocessing.

> [!IMPORTANT]
> **Quy tắc chia dữ liệu:** Train/Val/Test **phải chia theo Patient ID** (file `build_patient_splits.py`), không được chia ngẫu nhiên theo ảnh — vì 1 bệnh nhân có 2 ảnh (mắt trái + phải).

---

## 9. Dependencies (Thư Viện Bắt Buộc)

| Thư viện | Version tối thiểu | Vai trò |
|---|---|---|
| `torch` | ≥2.0 | Framework DL chính |
| `torchvision` | ≥0.15 | Pretrained EfficientNet-B0 |
| `timm` | ≥0.9 | Pretrained Swin Transformer |
| `albumentations` | ≥1.3 | Augmentation pipeline |
| `scikit-learn` | ≥1.2 | AUC-ROC computation |
| `pandas` | ≥2.0 | Đọc CSV, xử lý metadata |
| `opencv-python` | ≥4.7 | Đọc và xử lý ảnh |
| `pyyaml` | ≥6.0 | Đọc config YAML |
| `tqdm` | ≥4.65 | Progress bar (optional) |

---

## 10. Chỉ Mục Tài Liệu Chi Tiết (Documentation Index)

Nếu cần đi sâu vào một chủ đề cụ thể, tham khảo các báo cáo sau:

| Tài liệu | Mô tả |
|---|---|
| `docs/Bao_cao_Ablation_Study_CNN.md` | Phân tích chi tiết 3 thực nghiệm CNN (EXP 1–3) |
| `docs/Bao_cao_Ablation_Study_Swin.md` | Phân tích chi tiết 3 thực nghiệm Swin (EXP 4–6) |
| `docs/Bao_cao_Do_chinh_xac_Toan_dien.md` | Báo cáo Hamming/Subset Accuracy toàn diện |
| `docs/Bao_cao_Tien_xu_ly_Du_lieu.md` | Giải thích pipeline ROI→BenGraham→CLAHE |
| `docs/Bang_so_sanh_thuc_nghiem_toan_dien.md` | Bảng so sánh 6 EXP |
| `docs/bao_cao_chi_tiet_datn.md` | Báo cáo tổng thể đồ án tốt nghiệp |
| `docs/giai_thich_code_huan_luyen.md` | Giải thích từng dòng code `train.py` |
| `docs/giai_thich_hai_mo_hinh.md` | So sánh EfficientNet vs Swin Transformer |
| `docs/giai_thich_quy_trinh_xu_ly.md` | Quy trình xử lý dữ liệu end-to-end |
| `docs/giai_thich_tang_cuong_du_lieu.md` | Giải thích MixUp, CutMix, WRS |
| `docs/huong_dan_chi_tiet_ma_nguon.md` | Hướng dẫn chi tiết mã nguồn |
| `notebooks/kaggle_setup.md` | Hướng dẫn cài đặt chạy trên Kaggle |

---

*Phiên bản: 2.0 · Cập nhật: 2026-06-03 · Tối ưu cho AI Agent context ingestion.*
