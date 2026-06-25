# 📋 AGENT CONTEXT FILE — ODIR-5K Siamese Binocular Binary Classification (Phase 1)

> **Mục đích:** Đây là tài liệu **Single Source of Truth** dành cho AI Coding Agent.  
> Đọc file này = nắm toàn bộ kiến trúc, pipeline, kết quả, và quy ước của dự án ở Giai đoạn 1 (Phase 1).  
> **Không cần duyệt mã nguồn** trước khi bắt đầu làm việc.

---

## 0. Danh Tính Dự Án (Project Identity)

| Thuộc tính | Giá trị |
21. |---|---|
22. | **Tên dự án** | ODIR-5K Siamese Binocular Binary Classification |
23. | **Mục tiêu** | Phân loại nhị phân Bình thường vs Bệnh lý (Normal vs Pathological) + Hồi quy tuổi võng mạc (Age Regression) |
24. | **Kiến trúc** | Siamese Network (Mạng Xiêm) ghép cặp hai mắt đáy mắt (đầu vào song nhãn) |
25. | **Bộ dữ liệu** | ODIR-5K (~7.000 ảnh đáy mắt, ghép cặp thành ~3.500 bệnh nhân có đủ mắt trái/mắt phải) |
26. | **Tiêu chí thành công** | Đạt chỉ số kiểm thử trên Test Set: **Accuracy ≥ 90%**, **AUC-ROC ≥ 95%**, **F1-Score ≥ 90%**, **Sensitivity (Recall) ≥ 90%**, **Specificity ≥ 85%** |
27. | **Framework** | PyTorch (pure) · Albumentations · timm · scikit-learn |
28. | **Môi trường huấn luyện** | Kaggle Notebook (GPU T4 hoặc GPU P100) |
29. | **Môi trường phát triển** | Ubuntu Linux · Python 3.12 · VSCode |
30. | **Backbone A (CNN)** | EfficientNet-B0 (CNN, ~5.3M params, feature_dim=1280) |
31. | **Backbone B (Swin)** | Swin Transformer-Tiny (ViT, ~28M params, feature_dim=768) |
32. | **Nhãn phân loại (Binary)** | **0: Normal (Bình thường)** (Nếu cả hai mắt không có bệnh lý nào) · **1: Pathological (Bệnh lý)** (Nếu có ít nhất một mắt mắc bất kỳ bệnh lý nào trong 7 bệnh đáy mắt) |

---

## 1. Cây Thư Mục (Directory Tree)

```
DOANTOTNGHIEP/
├── train.py                    # ★ Entry-point huấn luyện (CLI mỏng: load → fit → test)
├── evaluate.py                 # So sánh 6 thực nghiệm → bảng Ablation Study
│
├── configs/                    # ★ YAML config cho 6 thực nghiệm nhị phân song nhãn
│   ├── exp_1_cnn_binary_raw.yaml            # CNN,  ảnh gốc,    không MixUp/CutMix
│   ├── exp_2_cnn_binary_enhanced.yaml       # CNN,  enhanced,   không MixUp/CutMix
│   ├── exp_3_cnn_binary_enhanced_aug.yaml   # CNN SOTA: enhanced + MixUp + CutMix ★
│   ├── exp_4_swin_binary_raw.yaml           # Swin, ảnh gốc,    không MixUp/CutMix
│   ├── exp_5_swin_binary_enhanced.yaml      # Swin, enhanced,   không MixUp/CutMix
│   └── exp_6_swin_binary_enhanced_aug.yaml  # Swin SOTA: enhanced + MixUp + CutMix ★
│
├── src/                        # ★ Logic Deep Learning (Phase 1 sạch — mỗi file 1 nhiệm vụ)
│   ├── __init__.py             # Public exports
│   ├── config.py               # Load YAML, set_seed (reproducible), resolve path Kaggle/local
│   ├── dataset.py              # ★ BinocularDataset (ghép cặp theo Patient ID, xử lý thiếu mắt) + build_dataloaders
│   ├── transforms.py           # Albumentations (khóa Hue, không CLAHE online)
│   ├── augment.py              # ★ Binocular MixUp/CutMix ĐỒNG BỘ trên cả 2 mắt
│   ├── losses.py               # BinaryFocalLoss (soft-label) + MultiTaskLoss = Focal + λ_age·SmoothL1
│   ├── metrics.py              # Acc/Precision/Sens/Spec/F1/AUC + find_best_threshold (Youden)
│   ├── engine.py               # ★ run_epoch + fit (two-stage, early-stop AUC) + evaluate_test
│   └── models/
│       ├── __init__.py         # build_model() → BinocularClassifier
│       ├── backbone.py         # build_backbone('cnn'|'swin') qua timm → (module, feature_dim)
│       └── siamese.py          # ★ BinocularClassifier (Siamese: backbone chia sẻ → fusion → 2 head)
│
├── scripts/                    # Tiền xử lý dữ liệu (chạy 1 lần — đã tạo enhanced_images)
│   ├── preprocess_enhance.py   # ROI Crop → Ben Graham → CLAHE
│   └── build_patient_splits.py # Chia Train/Val/Test theo Patient ID
│
├── archive/                    # DỮ LIỆU (giữ nguyên)
│   ├── ODIR-5K/.../Training Images   # Ảnh gốc raw
│   ├── enhanced_images/             # Ảnh ROI+BenGraham+CLAHE
│   └── splits_clean/                # train/val/test.csv + metadata.json
│
├── notebooks/
│   └── odir5k_binocular_kaggle.ipynb # ★ Notebook chạy toàn bộ pipeline trên Kaggle
│
├── kaggle_upload/upload_code.sh # Upload src/configs/splits/train.py/evaluate.py lên Kaggle Dataset
│
├── results/                    # Output huấn luyện (checkpoints, logs, test_results.json, comparison_table.md)
│
├── legacy_phase0/              # Code Phase 0 cũ (đa nhãn 8 bệnh) — lưu trữ tham khảo, KHÔNG import
│
├── docs/                       # Báo cáo & tài liệu học thuật
│
└── AGENT_GUIDE.md              # ← BẠN ĐANG ĐỌC FILE NÀY
```

---

## 2. Kiến Trúc Mạng Siamese (Architecture Overview)

Kiến trúc ghép cặp song nhãn trích xuất đặc trưng độc lập từ hai mắt sử dụng chung một Backbone chia sẻ trọng số (Weight Sharing Backbone), sau đó xử lý khuyết thiếu, ghép nối đặc trưng và đi qua Fusion MLP đa nhiệm.

```
                  ┌───────────────┐        ┌───────────────┐
                  │  Mắt Trái     │        │  Mắt Phải     │
                  │  [B, 3, H, W] │        │  [B, 3, H, W] │
                  └───────┬───────┘        └───────┬───────┘
                          │                        │
                          ▼                        ▼
                  ┌────────────────────────────────────────┐
                  │      SHARED WEIGHTS BACKBONE           │
                  │  (EfficientNet-B0 HOẶC Swin-Tiny)      │
                  └───────┬────────────────────────┬───────┘
                          │                        │
               Đặc trưng  ▼             Đặc trưng  ▼
               trái: [B, feature_dim]   phải: [B, feature_dim]
                          │                        │
                          ▼ (Áp dụng mặt nạ)       ▼ (Áp dụng mặt nạ)
                   [Xử lý mắt thiếu: nhân với cờ ~missing]
                          │                        │
                          └───────────┬────────────┘
                                      │
                                      ▼ (Ghép nối đặc trưng)
                             Concat: [B, 2 * feature_dim]
                                      │
                                      ▼
                             ┌──────────────────┐
                             │    Fusion MLP    │  -> Chiếu xuống 512 chiều
                             │ (LayerNorm/SiLU) │  -> Dropout (0.3)
                             └────────┬─────────┘
                                      │
                         Tích hợp đặc trưng: [B, 512]
                               ┌──────┴──────┐
                               │             │
                               ▼             ▼
                        ┌───────────┐ ┌───────────┐
                        │ CLS Head  │ │ REG Head  │  -> Nhánh phụ trợ
                        │  Linear   │ │  Linear   │  -> Hồi quy tuổi võng mạc
                        └─────┬─────┘ └─────┬─────┘
                              │             │
                              ▼             ▼
                         Logits [B,1]   Age Pred [B,1]
                         Binary Focal   Smooth L1 Loss (λ_age = 0.05)
```

### Xử lý khuyết thiếu (Missing Eyes):
Trong ODIR-5K, một số bệnh nhân bị thiếu ảnh một mắt (chỉ có mắt trái hoặc mắt phải). 
* Để xử lý, hệ thống nạp ảnh mắt bị khuyết là ảnh đen (zero tensor) và cung cấp cờ boolean `left_missing` / `right_missing`.
* Khi lan truyền tiến, đặc trưng trích xuất từ mắt bị khuyết sẽ nhân với `(~missing).float()` để ép hoàn toàn về `0` trước khi thực hiện ghép nối (concatenate), loại bỏ hoàn toàn nhiễu do ảnh giả gây ra.

---

## 3. Pipeline Dữ Liệu & Tăng Cường Đồng Bộ

### 3.1. Tiền xử lý nâng cao (Offline)
Tất cả ảnh đáy mắt gốc được chuẩn hóa thông qua bộ tiền xử lý tĩnh:
1. **ROI Crop:** Định vị võng mạc, cắt bỏ viền đen không chứa thông tin chẩn đoán, đưa về kích thước 512x512.
2. **Ben Graham Color Normalization:** Trừ trung bình cục bộ bằng Gaussian Blur rồi cộng 128 để triệt tiêuDevice Bias (sự khác biệt về ánh sáng của thiết bị chụp).
3. **CLAHE:** Tăng tương phản cục bộ trên không gian màu LAB (kênh L) giúp nổi rõ mạch máu và đĩa thị giác.

### 3.2. Tăng cường song nhãn đồng bộ (Binocular MixUp & CutMix)
Để áp dụng MixUp và CutMix cho mạng Siamese, việc trộn ảnh phải diễn ra **đồng bộ**. Nghĩa là:
* Nếu bệnh nhân A trộn với bệnh nhân B với tỷ lệ $\lambda$ (hoặc một vùng mặt nạ CutMix), thì **phép trộn tương tự phải được áp dụng đồng thời** cho cả cặp mắt trái của A/B và cặp mắt phải của A/B.
* Module `src/binocular_augment.py` quản lý tiến trình này để đảm bảo tính nhất quán không gian sinh học giữa mắt trái và mắt phải của cùng một bệnh nhân sau khi trộn.

---

## 4. Danh Sách 6 Thực Nghiệm Phân Loại Nhị Phân

Bộ thử nghiệm Ablation Study của Giai đoạn 1 gồm 6 cấu hình tương đương 2 nhóm kiến trúc:

### Nhóm 1: Backbone CNN (EfficientNet-B0)
1. **`exp_1_cnn_binary_raw`:** Chạy trên ảnh gốc, không dùng tăng cường MixUp/CutMix.
2. **`exp_2_cnn_binary_enhanced`:** Chạy trên ảnh đã tiền xử lý nâng cao, không dùng MixUp/CutMix.
3. **`exp_3_cnn_binary_enhanced_aug`:** Chạy trên ảnh tiền xử lý nâng cao + trộn MixUp/CutMix đồng bộ.

### Nhóm 2: Backbone Swin Transformer (Swin-Tiny)
4. **`exp_4_swin_binary_raw`:** Chạy trên ảnh gốc, không dùng tăng cường MixUp/CutMix.
5. **`exp_5_swin_binary_enhanced`:** Chạy trên ảnh đã tiền xử lý nâng cao, không dùng MixUp/CutMix.
6. **`exp_6_swin_binary_enhanced_aug`:** Chạy trên ảnh tiền xử lý nâng cao + trộn MixUp/CutMix đồng bộ.

---

## 5. Quyết Định Thiết Kế & Giải Thích Kỹ Thuật

### 5.1. Tự động tính Focal Loss Alpha (`focal_alpha: auto`)
* Để giải quyết vấn đề mất cân bằng lớp (756 mẫu Normal / 1584 mẫu Pathological trong tập Train), hệ thống tự động đếm số lượng mẫu của hai lớp trên tập huấn luyện thực tế và gán:
  $$\alpha = \frac{N_{\text{normal}}}{N_{\text{normal}} + N_{\text{pathological}}}$$
  Cho lớp Pathological và $1-\alpha$ cho lớp Normal. Điều này giúp loại bỏ hoàn toàn việc chỉnh tay $\alpha$ thủ công.

### 5.2. Cân chỉnh ngưỡng Youden Index trong một lượt chạy (Single Pass Calibration)
* Ngưỡng phân loại nhị phân tối ưu được tính toán động cuối mỗi epoch trên tập Validation bằng cách tối đa hóa chỉ số Youden:
  $$J = \text{Sensitivity} + \text{Specificity} - 1$$
* Nhằm tối ưu hiệu năng, việc thu thập dự đoán được thực hiện trực tiếp trong luồng validation của `run_epoch`. Sau khi kết thúc epoch, hàm `find_best_binary_threshold` sẽ tính toán và ghi đè các chỉ số tối ưu mà không cần chạy lại suy diễn (inference) một lần nữa.

### 5.3. Giám sát AUC-ROC tập Validation cho Early Stopping
* F1-score phụ thuộc vào một ngưỡng phân loại cụ thể và dễ dao động mạnh giữa các epoch đầu. Hệ thống chuyển sang giám sát **AUC-ROC tập Validation (`val_auc_roc`)** làm chỉ số quyết định dừng sớm (Early Stopping) giúp tiến trình huấn luyện học sâu ổn định hơn.

### 5.4. Khóa kênh màu Hue trong ColorJitter (`hue_shift_limit=0`)
* Các tổn thương y khoa võng mạc như xuất huyết có màu đỏ đặc trưng. Việc xoay góc Hue có thể đổi màu đỏ thành màu xanh/vàng, phá hủy đặc trưng lâm sàng khiến mô hình bị nhiễu.

---

## 6. Quy Ước Mã Nguồn (Code Conventions)

### 6.1. Định dạng Output của Mô hình
Mô hình `BinocularClassifier` luôn trả về một Python dictionary chứa logits phân loại nhị phân và age regression:
```python
output = model(
    left_image=left_img,    # Tensor [B, 3, 384, 384]
    right_image=right_img,  # Tensor [B, 3, 384, 384]
    left_missing=left_miss, # Tensor [B] (boolean)
    right_missing=right_miss# Tensor [B] (boolean)
)
# output = {
#     "logits": Tensor[B, 1],    # Logit thô của lớp Pathological (chưa qua sigmoid)
#     "age_pred": Tensor[B, 1]   # Tuổi dự đoán đã chuẩn hóa Z-score
# }
```

### 6.2. Định dạng mẫu dữ liệu đầu ra từ DataLoader
Mỗi batch dữ liệu tải từ `BinocularDataset` (file `src/dataset.py`) có dạng:
```python
batch = next(iter(train_loader))
# batch = {
#     "left_image": Tensor[B, 3, H, W],
#     "right_image": Tensor[B, 3, H, W],
#     "left_missing": Tensor[B] (boolean),
#     "right_missing": Tensor[B] (boolean),
#     "labels": Tensor[B, 1],           # Nhãn nhị phân: 0 (Normal) hoặc 1 (Pathological)
#     "age": Tensor[B, 1],              # Tuổi đã chuẩn hóa Z-score
#     "patient_id": list[str]           # ID bệnh nhân tương ứng
# }
```

---

## 7. Các Lệnh Huấn Luyện & Đánh Giá Nhanh (Quick CLI Commands)

*Lưu ý: Luôn chạy lệnh từ thư mục gốc dự án.*

### 7.1. Chạy thử nghiệm nhanh (Dry-Run 1 Epoch để kiểm tra lỗi)
```bash
PYTHONPATH=./ python3 train.py --config configs/exp_6_swin_binary_enhanced_aug.yaml --dry-run
```

### 7.2. Huấn luyện chính thức
```bash
PYTHONPATH=./ python3 train.py --config configs/exp_6_swin_binary_enhanced_aug.yaml
```

### 7.3. Khôi phục huấn luyện từ checkpoint
```bash
PYTHONPATH=./ python3 train.py --config configs/exp_6_swin_binary_enhanced_aug.yaml --resume results/exp_6_swin_binary_enhanced_aug/last_model.pth
```

### 7.4. Đánh giá và so sánh Ablation Study giữa các thực nghiệm
```bash
PYTHONPATH=./ python3 evaluate.py --exps results/exp_4_swin_binary_raw results/exp_5_swin_binary_enhanced results/exp_6_swin_binary_enhanced_aug --results-dir results
```

---

## 8. Hằng số bắt buộc khóa cứng (Hard Constraints)

> [!CAUTION]
> **Không được tự ý sửa đổi các hằng số sau:**
> * Định dạng nhãn nhị phân: Lớp `0` là Normal, Lớp `1` là Pathological.
> * Tuổi tối thiểu để lọc dữ liệu: `age_min_filter = 5` (để tránh nhiễu do trẻ sơ sinh chụp võng mạc).
> * Hệ số liên kết hàm loss đa nhiệm: `lam_age = 0.05` trong các file cấu hình YAML.

---

*Phiên bản: 4.0 (Giai đoạn 1 — viết lại sạch từ đầu) · Cập nhật: 2026-06-05 · Cấu trúc: `src/{config,dataset,transforms,augment,losses,metrics,engine}.py` + `src/models/{backbone,siamese}.py`. Code Phase 0 cũ lưu tại `legacy_phase0/`.*
