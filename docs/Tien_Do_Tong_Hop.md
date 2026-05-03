# TỔNG HỢP TIẾN ĐỘ DỰ ÁN — ODIR-5K Multi-task Learning
**Cập nhật lần cuối**: 28/04/2026  
**Sinh viên**: Ngô Đình Đạt — MSSV 2251161965 — Lớp 64HTTT2  
**GVHD**: TS. Lê Thị Tú Kiên  
**Đề tài**: Nghiên cứu và ứng dụng học sâu hỗ trợ chẩn đoán bệnh lý nhãn khoa và dự đoán tuổi sinh học từ ảnh đáy mắt

---

## TIẾN ĐỘ THEO ĐỀ CƯƠNG

| TT | Thời gian | Nội dung | Trạng thái |
|----|-----------|----------|------------|
| 1 | Tuần 1 | Lý thuyết CNN, Swin Transformer, Multi-task Learning, thu thập ODIR-5K | ✅ Hoàn thành |
| 2 | Tuần 2 | Tiền xử lý: ROI Cropping, CLAHE, Ben Graham | ✅ Hoàn thành |
| **3** | **Tuần 3–6** | **Xử lý mất cân bằng (MixUp, CutMix), splits, môi trường** | **✅ Hoàn thành** |
| 4 | Tuần 7–10 | CNN (ResNet/EfficientNet) + Swin Transformer — huấn luyện song song | ⏳ Chưa bắt đầu |
| 5 | Tuần 11 | Đánh giá, so sánh 2 mô hình, Ablation Study | ⏳ Chưa bắt đầu |
| 6 | Tuần 12–13 | Web App (Streamlit/Flask) | ⏳ Chưa bắt đầu |
| 7 | Tuần 14 | Hoàn thiện báo cáo, slide | ⏳ Chưa bắt đầu |

---

## CHI TIẾT TỪNG PHẦN ĐÃ HOÀN THÀNH

---

### ✅ MỤC 1 — Lý Thuyết & Thu Thập Dữ Liệu (Tuần 1)

**Dữ liệu gốc ODIR-5K:**
- Tổng ảnh: **6,392** fundus images (3,198 mắt trái + 3,194 mắt phải)
- Tổng bệnh nhân unique: **3,358**
- 8 nhãn bệnh: `N, D, G, C, A, H, M, O`
- Bài toán: **Multi-task Learning** = Classification (8 nhãn) + Regression (tuổi)

**Tài liệu lý thuyết đã có:**
- `Giai_thich_Dataset_ODIR.md` — ý nghĩa từng trường dữ liệu
- `Y_nghia_Tien_xu_ly_Anh_Y_te.md` — giải thích 3 bước preprocessing
- `Cong_nghe_Mo_hinh_Du_an.md` — tổng quan công nghệ, mô hình, pipeline

---

### ✅ MỤC 2 — Tiền Xử Lý Dữ Liệu (Tuần 2)

#### 2.1 Pipeline Xử Lý Ảnh (`scripts/preprocess_enhance.py`)

| Bước | Kỹ thuật | Output |
|------|----------|--------|
| B1 | ROI Cropping (phát hiện viền đen, cắt vùng nhãn cầu) | `archive/preprocessed_images/` |
| B2 | Ben Graham Color Normalization (chuẩn hóa ánh sáng) | |
| B3 | CLAHE trên kênh L (tăng tương phản cục bộ) | `archive/enhanced_images/` |

**Kết quả:**
- **6,392/6,392** ảnh đã được xử lý và lưu vào `archive/enhanced_images/`
- Kích thước đồng nhất: tất cả **512×512 pixel**
- File size trung bình: 60.9 KB/ảnh

#### 2.2 Phân Chia Dữ Liệu Patient-Level (`scripts/build_patient_splits.py`)

- Chiến lược: **Patient-level stratified split** — 2 ảnh trái/phải của cùng bệnh nhân KHÔNG bị tách ra 2 tập
- Kết quả ban đầu: Train 4,496 / Val 949 / Test 947
- **Zero patient leakage** đã xác nhận giữa cả 3 tập

#### 2.3 Kiểm Tra Chất Lượng (`scripts/check_preprocessing.py`)

8 nhóm kiểm tra:
- ✅ Schema CSV đầy đủ (20 cột)
- ✅ Labels hợp lệ (không có NULL, 1001 dòng multi-label)
- ⚠️ 28 hồ sơ tuổi bất thường (Age=1) → đã xử lý ở Mục 3
- ✅ Không trùng lặp filename
- ✅ 6,392/6,392 ảnh có trên disk khớp CSV
- ✅ Kích thước ảnh đồng nhất 512×512
- ✅ Zero patient leakage

**Tài liệu:** `Bao_cao_Tien_xu_ly_Du_lieu.md` (163 dòng, đầy đủ lý thuyết)

---

### ✅ MỤC 3 — Xử Lý Mất Cân Bằng Dữ Liệu (Tuần 3–6)

#### 3.1 Làm Sạch & Rebuild Splits (`scripts/clean_and_rebuild.py`)

Chạy ngày 28/04/2026 → tạo `archive/splits_clean/`:

| Tập | Số ảnh | Số bệnh nhân |
|-----|--------|-------------|
| Train | 4,462 | 2,340 |
| Val | 948 | 501 |
| Test | 954 | 502 |
| **Tổng** | **6,364** | **3,343** |

- Loại bỏ: 15 bệnh nhân (28 dòng) có tuổi = 1 (dữ liệu lỗi gốc ODIR-5K)
- Zero patient leakage đã xác nhận

**Thống kê tuổi (train set, sau làm sạch):**

```
mean = 58.14  |  std = 11.26  |  min = 14  |  max = 89
```

#### 3.2 Pos_weight cho BCEWithLogitsLoss (`archive/splits_clean/metadata.json`)

Giải pháp **TRỰC TIẾP** và CHÍNH cho class imbalance:

```
pos_weight[i] = số mẫu âm [i] / số mẫu dương [i]
Loss(i) = -[ pos_weight[i] × y × log(σ(x)) + (1-y) × log(1-σ(x)) ]
```

| Nhãn | Bệnh | Mẫu (+) | pos_weight |
|------|------|---------|------------|
| N | Normal | 1,478 | 2.02 |
| D | Diabetes Retinopathy | 1,486 | 2.00 |
| G | Glaucoma | 285 | 14.66 |
| C | Cataract | 288 | 14.49 |
| A | Age-related Macular Deg. | 215 | 19.75 |
| **H** | **Hypertension** | **132** | **32.80** ← thiểu số nặng nhất |
| M | Pathological Myopia | 199 | 21.42 |
| O | Other Diseases | 1,110 | 3.02 |

#### 3.3 MixUp Augmentation (`src/mixup.py`)

**Nguồn:** Zhang et al. (ICLR 2018) — *"mixup: Beyond Empirical Risk Minimization"*

**Lý thuyết:** Vicinal Risk Minimization — nội suy tuyến tính toàn bộ pixel:
```
x̃ = λ × xᵢ + (1-λ) × xⱼ     (ảnh)
ỹ = λ × yᵢ + (1-λ) × yⱼ     (nhãn — soft labels)
ã = λ × aᵢ + (1-λ) × aⱼ     (tuổi)
λ ~ Beta(α=0.4, α=0.4),  λ = max(λ, 1-λ) ≥ 0.5
```

**Kiểm tra:** `scripts/test_mixup.py` → **26/26 tests PASS**

Các test đã qua:
- ✅ Khởi tạo + validation input (alpha ≤ 0, prob ngoài [0,1])
- ✅ Shape output: image [N,3,H,W], labels [N,8], age [N,1]
- ✅ Lambda ≥ 0.5, soft labels trong (0,1)
- ✅ Ảnh đã bị trộn (khác ảnh gốc)
- ✅ Prob behavior: prob=0.01 → 1/50 batch bị trộn; prob=1.0 → 10/10 batch bị trộn
- ✅ Seed reproducibility
- ✅ Batch size 2, 4, 16, 32
- ✅ Multi-task age mixing nội suy tuyến tính
- ✅ repr chứa alpha và prob

#### 3.4 CutMix Augmentation (`src/cutmix.py`)

**Nguồn:** Yun et al. (ICCV 2019) — *"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"*

**Lý thuyết:** Cắt vùng hình chữ nhật từ ảnh B dán đè lên ảnh A, λ thực tế theo diện tích:
```
W_cut = W × √(1-λ₀),  H_cut = H × √(1-λ₀),  λ₀ ~ Beta(α=1.0, α=1.0)
image_mix[:, y1:y2, x1:x2] ← image_B[:, y1:y2, x1:x2]
λ = 1 - (x2-x1)(y2-y1) / (W×H)
labels_mix = λ × labels_A + (1-λ) × labels_B
age_mix    = λ × age_A    + (1-λ) × age_B
```

**Khác biệt cốt lõi với MixUp:** pixel bên ngoài vùng cắt **nguyên vẹn** (không có pixel trung gian) → bảo toàn cấu trúc cục bộ của ảnh fundus (mạch máu, microaneurysms).

**Kiểm tra:** `scripts/test_cutmix.py` → **37/37 tests PASS**

Các test đã qua:
- ✅ Khởi tạo + validation input
- ✅ Shape output đúng: image [N,3,H,W], labels [N,8], age [N,1]
- ✅ `_rand_bbox` hợp lệ với mọi λ (0.1→0.9), tọa độ clip vào [0,W]×[0,H]
- ✅ λ thực tế = 1 - area_ratio
- ✅ Prob behavior (prob=0.01 và prob=1.0)
- ✅ Seed reproducibility
- ✅ Batch size 2, 4, 16, 32
- ✅ Image size 224×224 và 384×384 (CNN và Swin-T)
- ✅ **Đặc trưng CutMix**: pixel ngoài vùng cắt = 0 (ảnh A) và pixel trong vùng cắt = 1 (ảnh B) — KHÔNG có pixel trung gian

#### 3.5 Input Pipeline hoàn chỉnh (`src/`)

| File | Vai trò | Trạng thái |
|------|---------|------------|
| `src/__init__.py` | Export 18 symbols cho training script | ✅ |
| `src/dataset.py` | `ODIRDataset` + `get_dataloaders()` | ✅ |
| `src/transforms.py` | Albumentations train/val/test (224 & 384) | ✅ |
| `src/utils.py` | `compute_class_weights`, metrics, age stats | ✅ |
| `src/mixup.py` | `MixUpCollator` + `get_mixup_dataloader` | ✅ |
| `src/cutmix.py` | `CutMixCollator` + `get_cutmix_dataloader` | ✅ |

---

## CẤU TRÚC THƯ MỤC HIỆN TẠI

```
DOANTOTNGHIEP/
├── archive/
│   ├── ODIR-5K/                  ← ảnh gốc (raw)
│   ├── preprocessed_images/      ← sau ROI Crop (512×512)
│   ├── enhanced_images/          ← sau Ben Graham + CLAHE (6,392 ảnh) ✅
│   ├── full_df.csv               ← CSV gốc toàn bộ (6,392 dòng)
│   ├── splits/                   ← splits gốc (còn 28 dòng tuổi=1)
│   └── splits_clean/             ← splits sạch (6,364 dòng) ✅ MỚI
│       ├── train.csv             (4,462 dòng)
│       ├── val.csv               (948 dòng)
│       ├── test.csv              (954 dòng)
│       ├── full_df_clean.csv     (6,364 dòng)
│       └── metadata.json         (class_weights + age_stats) ✅
├── scripts/
│   ├── preprocess_enhance.py     ← Bước 2: preprocessing pipeline ✅
│   ├── build_patient_splits.py   ← Bước 2: tạo splits ✅
│   ├── check_preprocessing.py    ← Bước 2: QA 8 nhóm kiểm tra ✅
│   ├── clean_and_rebuild.py      ← Bước 3: làm sạch + rebuild ✅ ĐÃ CHẠY
│   ├── test_mixup.py             ← Bước 3: 26/26 PASS ✅ MỚI
│   └── test_cutmix.py            ← Bước 3: 37/37 PASS ✅ MỚI
├── src/
│   ├── __init__.py               ← 18 symbols exported ✅ CẬP NHẬT
│   ├── dataset.py                ← ODIRDataset + get_dataloaders ✅
│   ├── transforms.py             ← Augmentation pipeline ✅
│   ├── utils.py                  ← Metrics + weights + stats ✅
│   ├── mixup.py                  ← MixUpCollator ✅
│   └── cutmix.py                 ← CutMixCollator ✅ MỚI
├── Bao_cao_Tien_xu_ly_Du_lieu.md ← Báo cáo Mục 2+3 (163 dòng) ✅ CẬP NHẬT
├── Cong_nghe_Mo_hinh_Du_an.md    ← Tổng quan công nghệ ✅
├── Giai_thich_Code_Tien_xu_ly.md ← Giải thích code ✅
├── Giai_thich_Dataset_ODIR.md    ← Giải thích dataset ✅
├── Y_nghia_Tien_xu_ly_Anh_Y_te.md ← Lý thuyết preprocessing ✅
└── Tac_dung_File_Python_va_Muc_do_Quan_trong.md ← Vai trò từng file ✅
```

---

## KIỂM TRA TÍCH HỢP (`src` PACKAGE)

```python
from src import MixUpCollator, CutMixCollator   # ✅
from src import ODIRDataset, get_dataloaders     # ✅
from src import LABELS, compute_class_weights    # ✅
from src import get_transforms                   # ✅
# 18/18 symbols export thành công
```

---

## VIỆC CÒN LẠI (THEO ĐỀ CƯƠNG)

### Mục 4 — Model CNN + Swin Transformer (Tuần 7–10)

Cần tạo mới:

| File cần tạo | Nội dung |
|---|---|
| `src/models/resnet_mtl.py` | ResNet50/EfficientNet với 2 đầu ra (classification + regression) |
| `src/models/swin_mtl.py` | Swin-Transformer-Tiny/Small với 2 đầu ra |
| `src/loss.py` | Joint Loss = BCE(pos_weight) + λ×MAE — cân bằng 2 nhiệm vụ |
| `train.py` | Training loop đầy đủ (AdamW, CosineAnnealing, early stopping) |
| `evaluate.py` | Tính F1, AUC-ROC (8 nhãn), MAE/Pearson (tuổi) |
| `configs/` | Config file cho 8 thực nghiệm (2 model × 2 preprocessing × 2 imbalance) |

### Mục 5 — Đánh Giá & Ablation Study (Tuần 11)

- Chạy 8 thực nghiệm: CNN/Swin × Raw/Enhanced × NoBal/Bal
- Bảng so sánh F1-Score, AUC-ROC, MAE, Pearson Correlation
- Ablation Study: tắt từng component, đo delta performance

### Mục 6 — Web App (Tuần 12–13)

- Streamlit hoặc Flask
- Upload ảnh fundus → dự đoán 8 nhãn + tuổi + Retinal Age Gap
- Heatmap visualization (Grad-CAM)

---

## TỔNG KẾT SỐ LIỆU KEY

| Chỉ số | Giá trị |
|--------|---------|
| Tổng ảnh đã xử lý | 6,392 ảnh (512×512) |
| Dữ liệu sạch (sau loại tuổi lỗi) | 6,364 ảnh / 3,343 bệnh nhân |
| Train / Val / Test | 4,462 / 948 / 954 ảnh |
| Nhãn thiểu số nhất (H) | 132 mẫu — pos_weight = **32.80** |
| Tests MixUp PASS | **26/26** |
| Tests CutMix PASS | **37/37** |
| Symbols export từ `src` | **18/18** |
| Files Python đã code | **13 files** |
| Tỉ lệ hoàn thành đề cương | **~3/7 tuần (Mục 1→3)** |
