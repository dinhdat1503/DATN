# Giải thích chi tiết code huấn luyện mô hình CNN & Swin Transformer

> Tài liệu giải thích **toàn bộ code huấn luyện** của đồ án ODIR-5K Phase 1 (Siamese nhị phân: Normal vs Pathological).
> Bổ trợ cho tài liệu kiến trúc model: [`GIAI_THICH_MO_HINH_CNN_SWIN.md`](GIAI_THICH_MO_HINH_CNN_SWIN.md).

> **⭐ ĐIỀU QUAN TRỌNG NHẤT:** Code huấn luyện cho **CNN và Swin là CHUNG MỘT bộ, hoàn toàn giống nhau**. Sự khác biệt duy nhất là một dòng `model_type: cnn` hay `model_type: swin` trong file config. Vòng lặp huấn luyện, hàm loss, optimizer, early stopping... đều **dùng chung**. Vì vậy tài liệu này giải thích một lần cho cả hai mô hình.

---

## Mục lục
1. [Bản đồ các file liên quan](#1-bản-đồ-các-file-liên-quan)
2. [Bức tranh tổng thể: ai gọi ai](#2-bức-tranh-tổng-thể-ai-gọi-ai)
3. [`train.py` — Nhạc trưởng lắp ráp](#3-trainpy--nhạc-trưởng-lắp-ráp)
4. [`engine.py` — Động cơ huấn luyện](#4-enginepy--động-cơ-huấn-luyện)
   - [4.1 `run_epoch()` — chạy 1 epoch](#41-run_epoch--chạy-1-epoch-forwardbackward)
   - [4.2 `build_optimizer_scheduler()`](#42-build_optimizer_scheduler--adamw--cosine-theo-giai-đoạn)
   - [4.3 `fit()` — vòng lặp chính](#43-fit--vòng-lặp-huấn-luyện-chính-)
   - [4.4 `evaluate_test()`](#44-evaluate_test--đánh-giá-trên-test)
5. [`losses.py` — Hàm mất mát đa nhiệm](#5-lossespy--hàm-mất-mát-đa-nhiệm)
6. [`metrics.py` — Chỉ số đánh giá](#6-metricspy--chỉ-số-đánh-giá)
7. [Config — siêu tham số huấn luyện](#7-config--siêu-tham-số-huấn-luyện)
8. [CNN và Swin khác nhau ở đâu khi huấn luyện?](#8-cnn-và-swin-khác-nhau-ở-đâu-khi-huấn-luyện)
9. [Sơ đồ luồng huấn luyện end-to-end](#9-sơ-đồ-luồng-huấn-luyện-end-to-end)
10. [Các "vũ khí" chống overfit & mất cân bằng](#10-các-vũ-khí-chống-overfit--mất-cân-bằng)
11. [Câu hỏi thường gặp khi bảo vệ](#11-câu-hỏi-thường-gặp-khi-bảo-vệ)

---

## 1. Bản đồ các file liên quan

| File | Vai trò trong huấn luyện |
|------|--------------------------|
| `train.py` | **Entry point** — đọc config, lắp ráp dữ liệu/model/loss, gọi `fit()` rồi `evaluate_test()` |
| `src/engine.py` | **Động cơ chính** — `run_epoch`, `fit`, `build_optimizer_scheduler`, `evaluate_test` |
| `src/losses.py` | Hàm loss đa nhiệm: `BinaryFocalLoss` + `MultiTaskLoss` |
| `src/metrics.py` | Tính AUC/F1/Sensitivity/Specificity + ngưỡng Youden + MAE tuổi |
| `src/config.py` | Đọc YAML, cố định seed, tự dò đường dẫn (local/Kaggle) |
| `configs/exp_*.yaml` | Siêu tham số mỗi thực nghiệm |
| `src/models/` | Mạng Siamese (CNN/Swin) — xem tài liệu model riêng |
| `src/dataset.py`, `src/augment.py` | Nạp dữ liệu + tăng cường (MixUp/CutMix) |

---

## 2. Bức tranh tổng thể: ai gọi ai

```
            python train.py --config configs/exp_3_cnn_binary_enhanced_aug.yaml
                                   │
                                   ▼
                            train.py : main()         ← lắp ráp mọi thứ
                                   │
        ┌──────────────┬──────────┼───────────────┬──────────────┐
        ▼              ▼          ▼                ▼              ▼
  load_config     build_       build_model    MultiTaskLoss   compute_
  set_seed        dataloaders  (cnn/swin)     (losses.py)     focal_alpha
  (config.py)     (dataset.py) (models/)                      (train.py)
                                   │
                                   ▼
                          engine.fit()  ◄────────── VÒNG LẶP HUẤN LUYỆN
                                   │  (gọi run_epoch mỗi epoch)
                                   ▼
                       engine.evaluate_test()  ◄──── đánh giá Test cuối cùng
```

---

## 3. `train.py` — Nhạc trưởng lắp ráp

File này **không chứa vòng lặp epoch**. Nhiệm vụ của `main()` ([train.py:62](../train.py#L62)) là chuẩn bị "nguyên liệu" theo đúng thứ tự rồi giao cho engine.

### Thứ tự khởi tạo trong `main()`
```python
cfg = load_config(config_path)          # 1. Đọc YAML
set_seed(seed)                          # 2. Cố định seed (tái lập)
device = "cuda" if available else "cpu" # 3. Chọn thiết bị
splits_dir, img_dir, results_dir = ...  # 4. Tự dò đường dẫn (local/Kaggle)
focal_alpha = compute_focal_alpha(...)  # 5. Tự tính alpha chống mất cân bằng
dataloaders = build_dataloaders(...)    # 6. Tạo loader train/val/test
model = build_model(model_type=...)     # 7. Dựng Siamese (cnn HOẶC swin)
criterion = MultiTaskLoss(...)          # 8. Hàm loss đa nhiệm
best_path = fit(...)                    # 9. ⭐ HUẤN LUYỆN
evaluate_test(...)                      # 10. Đánh giá Test
```

### Điểm đặc sắc: `compute_focal_alpha()` ([train.py:43](../train.py#L43))
Tự động tính `alpha` cho Focal Loss = **tỉ lệ Normal trên tổng số ở mức BỆNH NHÂN** của tập Train:
```python
g = df.groupby("ID").first()          # gộp về mức bệnh nhân (không tính trùng theo ảnh)
n_normal = (g["N"] == 1).sum()        # số bệnh nhân Normal
n_path   = (g["N"] == 0).sum()        # số bệnh nhân Pathological
alpha = n_normal / (n_normal + n_path)  # ≈ 0.323
```
Vì lớp Pathological (~68%) áp đảo lớp Normal (~32%), `alpha ≈ 0.323` sẽ **giảm trọng số lớp đa số, tăng trọng số lớp thiểu số** → chống "sụp đổ lớp" (model lười đoán hết về Pathological). Đặt `focal_alpha: auto` trong config để bật tính năng này.

### Các cờ dòng lệnh ([train.py:163-168](../train.py#L163-L168))
- `--config / -c`: file YAML (bắt buộc).
- `--dry-run`: chạy 1 epoch / 1 batch để kiểm tra pipeline không lỗi.
- `--resume / -r`: tiếp tục huấn luyện từ checkpoint.

---

## 4. `engine.py` — Động cơ huấn luyện

Đây là **trái tim** của code huấn luyện. Có 4 hàm chính.

### 4.1 `run_epoch()` — chạy 1 epoch (forward/backward)
[engine.py:56](../src/engine.py#L56). Một hàm dùng chung cho cả **train, val, test** (phân biệt qua tham số `mode`).

**Luồng xử lý mỗi batch** ([engine.py:111-152](../src/engine.py#L111-L152)):
```python
# 1) Đưa dữ liệu lên GPU
left, right        = batch["left_image"], batch["right_image"]   # ảnh 2 mắt
left_m, right_m    = batch["left_missing"], batch["right_missing"] # cờ mắt thiếu
labels, age_true   = batch["label"], batch["age"]

# 2) Forward (trong autocast — mixed precision)
with make_autocast(use_amp):
    output = model(left, right, left_m, right_m)     # → {"logits", "age_pred"}
    loss, detail = criterion(output["logits"], labels, output["age_pred"], age_true)
    loss_back = loss / accum_steps                    # chia cho số bước tích lũy

# 3) Backward + cập nhật (chỉ khi train)
scaler.scale(loss_back).backward()                    # backward có AMP
if is_step:                                           # đủ accum_steps mới step
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), max_norm=1.0) # chống gradient nổ
    scaler.step(optimizer); scaler.update()
    optimizer.zero_grad()

# 4) Gom dự đoán → tính metrics cuối epoch
all_probs.extend(sigmoid(logits))                     # xác suất để tính AUC/F1
```

**3 kỹ thuật quan trọng trong hàm này:**
1. **AMP (Automatic Mixed Precision)** — `make_autocast` + `GradScaler` ([engine.py:31-49](../src/engine.py#L31-L49)): tính ở FP16 cho nhanh & tiết kiệm VRAM, `GradScaler` chống underflow gradient. Có lớp tương thích cho cả `torch.amp` mới lẫn `torch.cuda.amp` cũ trên Kaggle.
2. **Gradient Accumulation** ([engine.py:125-128](../src/engine.py#L125-L128)): batch_size nhỏ (8) nhưng tích lũy gradient qua `accum_steps=2` → **batch hiệu dụng = 16**, giả lập batch lớn mà không tràn VRAM.
3. **Gradient Clipping** (`max_norm=1.0`): cắt chuẩn gradient, ổn định huấn luyện (đặc biệt quan trọng với Transformer/Swin).

> Khi **eval** (`mode="val"/"test"`): `optimizer=None`, bọc trong `torch.no_grad()`, không backward. Cùng một hàm, chỉ tắt phần cập nhật.

### 4.2 `build_optimizer_scheduler()` — AdamW + Cosine theo giai đoạn
[engine.py:177](../src/engine.py#L177). Tạo optimizer/scheduler **khác nhau theo giai đoạn** two-stage:

| | Stage 1 (`frozen`) | Stage 2 (`unfrozen`) |
|---|---|---|
| Tham số train | chỉ phần `requires_grad=True` (head) | toàn mạng |
| Learning rate | `frozen_lr = 1e-3` (lớn) | `unfrozen_lr = 1e-4` (nhỏ) |
| `T_max` của Cosine | `freeze_epochs` | `epochs - freeze_epochs` |

- **Optimizer:** `AdamW` (`weight_decay` để regularize — chống overfit).
- **Scheduler:** `CosineAnnealingLR` — LR giảm mượt theo hình cos từ giá trị đầu xuống `eta_min=1e-6`.

### 4.3 `fit()` — vòng lặp huấn luyện chính ⭐
[engine.py:216](../src/engine.py#L216). Đây là hàm `train.py` gọi tới. Gồm các cơ chế:

**(a) Two-stage training (đóng băng → mở khóa)**
```python
# Khởi đầu: đóng băng backbone, chỉ train head
model.freeze_backbone(); stage = "frozen"          # [engine.py:246-248]

# Khi sang epoch (freeze_epochs+1): mở khóa toàn mạng
if epoch == freeze_epochs + 1 and stage == "frozen":
    model.unfreeze_backbone()                       # [engine.py:292-296]
    optimizer, scheduler = build_optimizer_scheduler(..., "unfrozen", ...)
    stage = "unfrozen"
```
→ 5 epoch đầu chỉ "làm nóng" head (không phá pretrained), từ epoch 6 mới fine-tune cả backbone với LR nhỏ.

**(b) Mỗi epoch** ([engine.py:303-328](../src/engine.py#L303-L328)):
```python
train_m = run_epoch(..., "train", ...)        # huấn luyện
val_m, val_probs, val_targets = run_epoch(..., "val", ...)   # đánh giá
cur_thresh = find_best_threshold(val_probs, val_targets)     # ngưỡng Youden trên val
# tính lại metrics theo ngưỡng tối ưu, ghi log CSV
scheduler.step()
```

**(c) Cân chỉnh ngưỡng (calibration) bằng Youden** ([engine.py:313-318](../src/engine.py#L313-L318)):
Thay vì cứng nhắc dùng ngưỡng 0.5, model tìm ngưỡng tối ưu trên tập **val** (cân bằng độ nhạy/đặc hiệu — quan trọng trong y tế). Dùng lại dự đoán val đã có, **không forward lại** → tiết kiệm.

**(d) Early Stopping theo `val_AUC`** ([engine.py:331-363](../src/engine.py#L331-L363)):
```python
if val_auc > best_auc:        # cải thiện → lưu best_model.pth
    best_auc = val_auc; best_thresh = cur_thresh; no_improve = 0
    torch.save({...}, best_path)
else:                         # không cải thiện
    no_improve += 1
    if no_improve >= patience:    # đủ kiên nhẫn → dừng sớm
        break
```
Chọn AUC làm tiêu chí vì nó **độc lập ngưỡng** và bền với mất cân bằng lớp. Mỗi epoch cũng lưu `last_model.pth` để resume.

**(e) Resume** ([engine.py:259-279](../src/engine.py#L259-L279)): nạp lại model + optimizer + scheduler + best_auc, tự xác định lại đang ở Stage 1 hay 2 theo epoch.

### 4.4 `evaluate_test()` — đánh giá trên Test
[engine.py:373](../src/engine.py#L373). Nạp `best_model.pth`, rồi:
1. Tìm lại ngưỡng Youden tối ưu **trên val** ([engine.py:394-399](../src/engine.py#L394-L399)).
2. Chạy **1 forward pass** trên Test, báo cáo metrics ở **cả 2 ngưỡng**: 0.5 và ngưỡng Youden ([engine.py:402-415](../src/engine.py#L402-L415)).
3. Lưu `test_results.json` (AUC, F1, Sensitivity, Specificity, Accuracy, Age MAE/Pearson).

> Lưu ý phương pháp luận chuẩn mực: ngưỡng được chọn trên **val**, rồi áp lên **test** — không "rò rỉ" nhãn test.

---

## 5. `losses.py` — Hàm mất mát đa nhiệm

Tổng loss = **Loss phân loại bệnh (Focal)** + λ × **Loss hồi quy tuổi (SmoothL1)**.

### `BinaryFocalLoss` ([losses.py:18](../src/losses.py#L18))
Công thức (cho từng mẫu):
```
p = sigmoid(logit)
loss_pos = -alpha     · (1-p)^gamma · log(p)        # khi nhãn y=1 (Pathological)
loss_neg = -(1-alpha) ·    p^gamma  · log(1-p)      # khi nhãn y=0 (Normal)
loss     = y·loss_pos + (1-y)·loss_neg              # tổ hợp tuyến tính theo nhãn
```
- **`gamma=2.0`** (focusing): mẫu **dễ** (p gần đúng) bị nhân `(1-p)^gamma` rất nhỏ → giảm đóng góp, model **tập trung vào mẫu khó**.
- **`alpha≈0.323`** (auto): cân bằng hai lớp.
- **Tương thích nhãn mềm**: vì tổ hợp tuyến tính theo `targets`, nên đúng cả khi nhãn là số thực trong [0,1] do **MixUp/CutMix** sinh ra ([losses.py:55-56](../src/losses.py#L55-L56)).

### `MultiTaskLoss` ([losses.py:60](../src/losses.py#L60))
```python
cls_loss = BinaryFocalLoss(alpha, gamma)(logits, labels)   # nhiệm vụ chính
reg_loss = SmoothL1Loss()(age_pred, age_true)              # nhiệm vụ phụ (tuổi)
total    = cls_loss + lam_age · reg_loss                   # lam_age = 0.05
```
- **`SmoothL1`** (Huber) cho tuổi: bền với outlier hơn MSE.
- **`lam_age=0.05`** nhỏ: tuổi chỉ là nhiệm vụ **phụ trợ** giúp backbone học biểu diễn võng mạc giàu hơn, không lấn át nhiệm vụ chính.
- Trả về `(total, detail)` — `detail` chứa từng thành phần loss để ghi log.

---

## 6. `metrics.py` — Chỉ số đánh giá

### `compute_binary_metrics()` ([metrics.py:25](../src/metrics.py#L25))
Từ confusion matrix (TP/FP/FN/TN) tính:
| Chỉ số | Công thức | Ý nghĩa lâm sàng |
|---|---|---|
| Accuracy | (TP+TN)/N | tỉ lệ đúng chung |
| Precision | TP/(TP+FP) | dự đoán bệnh thì đúng bao nhiêu |
| **Sensitivity** (Recall) | TP/(TP+FN) | **bắt được bao nhiêu ca bệnh** (rất quan trọng y tế) |
| **Specificity** | TN/(TN+FP) | nhận đúng người khỏe |
| F1 | 2·P·Se/(P+Se) | cân bằng precision–recall |
| AUC-ROC | (sklearn) | **độc lập ngưỡng**, bền với mất cân bằng |

### `find_best_threshold()` ([metrics.py:71](../src/metrics.py#L71))
Quét ngưỡng 0.05→0.95 (bước 0.01), chọn ngưỡng tối đa **Chỉ số Youden**:
```
J = Sensitivity + Specificity − 1
```
→ điểm cân bằng tốt nhất giữa bắt bệnh và tránh báo động giả.

### `compute_age_metrics()` ([metrics.py:104](../src/metrics.py#L104))
Giải chuẩn hóa Z-score về **đơn vị năm** rồi tính **MAE** (sai số tuổi trung bình) và **Pearson** (tương quan tuổi dự đoán vs thực).

---

## 7. Config — siêu tham số huấn luyện

Ví dụ [`configs/exp_3_cnn_binary_enhanced_aug.yaml`](../configs/exp_3_cnn_binary_enhanced_aug.yaml):

```yaml
model_type: cnn            # ← ĐỔI thành "swin" là chuyển mô hình
loss:
  focal_alpha: auto        # tự tính chống mất cân bằng
  focal_gamma: 2.0
  lam_age: 0.05            # trọng số nhiệm vụ tuổi
training:
  img_size: 384
  batch_size: 8
  gradient_accumulation_steps: 2   # batch hiệu dụng = 16
  epochs: 40
  early_stopping_patience: 10
  two_stage: true          # bật đóng băng→mở khóa
  freeze_epochs: 5         # 5 epoch đầu đóng băng backbone
  frozen_lr: 0.001         # LR Stage 1
  unfrozen_lr: 0.0001      # LR Stage 2
  weight_decay: 0.01
  eta_min: 0.000001
augmentation:
  use_mixup: true
  use_cutmix: true
```

> **Triết lý thiết kế** ([config.py:11](../src/config.py#L11)): *không hardcode siêu tham số trong code* — mọi thứ đọc từ YAML để dễ làm thực nghiệm so sánh.

`config.py` còn lo **tự dò đường dẫn** cho cả local lẫn Kaggle (`resolve_splits_dir`, `resolve_img_dir`, `resolve_results_dir` — [config.py:74-151](../src/config.py#L74-L151)) và **cố định seed** (`set_seed` — [config.py:36](../src/config.py#L36)) để tái lập kết quả.

---

## 8. CNN và Swin khác nhau ở đâu khi huấn luyện?

**Gần như KHÔNG khác gì.** Toàn bộ engine, loss, metrics, vòng lặp đều dùng chung. Khác biệt chỉ nằm ở:

| | CNN (EXP 1–3) | Swin (EXP 4–6) |
|---|---|---|
| Dòng config | `model_type: cnn` | `model_type: swin` |
| Backbone được dựng | EfficientNet-B0 (D=1280) | Swin-Tiny (D=768) |
| `img_size` cho backbone | không bắt buộc | cần truyền để nội suy vị trí |
| Batch size thực tế | có thể lớn hơn | thường nhỏ hơn (Swin nặng VRAM hơn) → tăng `gradient_accumulation_steps` |

Mọi cơ chế khác — focal loss, two-stage, Youden, early stopping, AMP, grad clip — **y hệt nhau**. Đây chính là điểm mạnh thiết kế: **một bộ code, so sánh công bằng hai kiến trúc** trên cùng quy trình.

Bộ 6 config tạo thành **ma trận ablation**:
```
            Raw (ảnh gốc)   Enhanced       Enhanced + Aug
CNN          EXP 1           EXP 2          EXP 3 ★
Swin         EXP 4           EXP 5          EXP 6 ★
```
→ Bóc tách đóng góp của **tiền xử lý ảnh** và **tăng cường dữ liệu (MixUp/CutMix)** cho từng kiến trúc.

---

## 9. Sơ đồ luồng huấn luyện end-to-end

```
 ┌─────────────────────────── train.py main() ───────────────────────────┐
 │ load_config → set_seed → resolve paths → compute_focal_alpha           │
 │ build_dataloaders → build_model(cnn/swin) → MultiTaskLoss              │
 └───────────────────────────────────┬───────────────────────────────────┘
                                     ▼
 ┌──────────────────────────── engine.fit() ─────────────────────────────┐
 │                                                                        │
 │  STAGE 1 (epoch 1..5): freeze_backbone, LR=1e-3, chỉ train head        │
 │  STAGE 2 (epoch 6..N): unfreeze_backbone, LR=1e-4, fine-tune toàn mạng │
 │                                                                        │
 │  MỖI EPOCH:                                                            │
 │    ┌─ run_epoch("train") ──────────────────────────────┐              │
 │    │   for batch:                                        │              │
 │    │     forward → MultiTaskLoss (Focal + 0.05·SmoothL1)│              │
 │    │     backward (AMP) → grad accum → clip → step      │              │
 │    └─────────────────────────────────────────────────────┘            │
 │    ┌─ run_epoch("val") → val_probs ─────────────────────┐              │
 │    │   find_best_threshold (Youden) trên val            │              │
 │    │   compute_binary_metrics (AUC, F1, Sens, Spec)     │              │
 │    └─────────────────────────────────────────────────────┘            │
 │    scheduler.step() (Cosine)                                           │
 │    if val_AUC tốt hơn → lưu best_model.pth                             │
 │    else no_improve++ ; nếu ≥ patience → EARLY STOP                     │
 │    ghi training_log.csv ; lưu last_model.pth                           │
 └───────────────────────────────────┬───────────────────────────────────┘
                                     ▼
 ┌─────────────────────── engine.evaluate_test() ────────────────────────┐
 │ nạp best_model → tìm ngưỡng Youden trên val → forward Test            │
 │ báo cáo ở ngưỡng 0.5 VÀ ngưỡng Youden → lưu test_results.json         │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Các "vũ khí" chống overfit & mất cân bằng

Tổng hợp các kỹ thuật trong code huấn luyện (rất hữu ích khi bảo vệ):

| Kỹ thuật | Ở đâu | Chống vấn đề gì |
|---|---|---|
| **Focal Loss + alpha auto** | `losses.py`, `train.py:compute_focal_alpha` | mất cân bằng lớp (32/68), sụp đổ lớp |
| **Two-stage (freeze→unfreeze)** | `engine.fit` | phá hỏng pretrained do gradient sốc |
| **Early stopping theo AUC** | `engine.fit` | overfit khi train quá lâu |
| **Ngưỡng Youden (calibration)** | `metrics.find_best_threshold` | ngưỡng 0.5 không tối ưu cho y tế |
| **Weight decay (AdamW)** | `build_optimizer_scheduler` | overfit (regularize trọng số) |
| **Dropout 0.3** | Fusion MLP (model) | overfit |
| **MixUp / CutMix** | `augment.py` (bật qua config) | overfit, tăng đa dạng dữ liệu |
| **Gradient clipping** | `run_epoch` | gradient nổ (nhất là Swin) |
| **AMP mixed precision** | `run_epoch` | tiết kiệm VRAM, tăng tốc |
| **Gradient accumulation** | `run_epoch` | giả lập batch lớn trên GPU nhỏ |
| **Cosine LR schedule** | `build_optimizer_scheduler` | hội tụ mượt, thoát điểm yên ngựa |
| **Seed cố định** | `config.set_seed` | tái lập kết quả |

---

## 11. Câu hỏi thường gặp khi bảo vệ

1. **"Code huấn luyện CNN và Swin có khác nhau không?"**
   → Không. Chung một bộ engine, chỉ khác `model_type` trong config. Đây là chủ đích để so sánh công bằng.

2. **"Vì sao dùng Focal Loss thay vì BCE thường?"**
   → Dữ liệu mất cân bằng (32% Normal / 68% Pathological). Focal (alpha + gamma) cân bằng lớp và tập trung vào mẫu khó, chống model "đoán bừa" về lớp đa số.

3. **"Two-stage training là gì, vì sao cần?"**
   → 5 epoch đầu đóng băng backbone chỉ train head (LR lớn) để head ổn định; sau đó mở khóa fine-tune cả mạng (LR nhỏ). Tránh gradient lớn lúc đầu phá trọng số pretrained.

4. **"Vì sao chọn AUC làm tiêu chí early stopping mà không phải accuracy?"**
   → AUC độc lập ngưỡng và bền với mất cân bằng lớp; accuracy dễ "ăn gian" khi lớp lệch.

5. **"Ngưỡng Youden để làm gì?"**
   → Tìm ngưỡng cân bằng Sensitivity + Specificity tối ưu trên val (thay vì cứng 0.5), phù hợp bài toán y tế cần độ nhạy cao. Chọn trên val rồi áp lên test để không rò rỉ nhãn.

6. **"Nhánh dự đoán tuổi ảnh hưởng huấn luyện thế nào?"**
   → Là nhiệm vụ phụ trợ (`lam_age=0.05`), thêm tín hiệu giúp backbone học biểu diễn võng mạc tốt hơn, gián tiếp hỗ trợ phân loại bệnh.

7. **"Batch size chỉ 8 có nhỏ quá không?"**
   → Dùng `gradient_accumulation_steps=2` → batch hiệu dụng 16; cộng AMP để vừa VRAM GPU Kaggle (T4/P100). Swin nặng hơn nên kỹ thuật này càng cần thiết.

8. **"Làm sao đảm bảo kết quả tái lập?"**
   → `set_seed(42)` cố định random/numpy/torch; lưu cả `config.yaml` vào thư mục kết quả để truy vết.
