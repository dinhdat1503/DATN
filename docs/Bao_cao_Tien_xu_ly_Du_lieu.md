# Báo Cáo: Tiến Độ và Nội Dung Tiền Xử Lý Dữ Liệu (Data Preprocessing)

Theo như đề cương dự án mạng Neural đa nhiệm (Multi-task Learning) trên tập dữ liệu y tế nhãn khoa **ODIR-5K**, toàn bộ module nền tảng "Tiền Xử Lý Dữ Liệu" đã được thực thi và xác nhận hoàn tất thành công. Dưới đây là chi tiết các nội dung đã được ghi nhận lưu trữ lại.

## 1. Tổng Quan Cấu Trúc Dữ Liệu ODIR-5K Đã Làm Sạch
Thông tin được kết xuất từ file báo cáo kiểm tra (`scripts/check_result_utf8.txt`):
- **Tổng số lượng ảnh (Fundus images)**: `6,392` bức ảnh.
    - Mắt trái (Left): `3,198` ảnh.
    - Mắt phải (Right): `3,194` ảnh.
- **Tổng số bệnh nhân (Unique Patients)**: `3,358` hồ sơ bệnh án (Tỉ lệ Nam `3,424`/ Nữ `2,968`).
- **Xử lý dị thường thuộc tính (Anomalies Control)**:
    - Bắt lỗi chuẩn độ tuổi ngoại lệ (28 hồ sơ bất thường có độ tuổi bằng 1).
    - Hệ thống nhãn bệnh (8 Labels): Đa số phân bố ở Normal (N), Diabetic Retinopathy (D) và Other (O).
    - Đã xác thực không có sự tồn tại của file null rác giữa CSV và hệ thống file trên đĩa cứng.

## 2. Kỹ Thuật Chia Tách Dữ Liệu (Data Splitting)
Dữ liệu đã được phân tách và lưu trữ thành 3 tập tin `.csv` độc lập bên trong thư mục `archive/splits`:
1. **Train Set (Huấn luyện)**: `4,496` ảnh (Tỉ trọng ~70%).
2. **Validation Set (Kiểm định)**: `949` ảnh (Tỉ trọng ~15%).
3. **Test Set (Đánh giá độc lập)**: `947` ảnh (Tỉ trọng ~15%).

> **[LƯU Ý QUAN TRỌNG VỀ LEAKAGE]**: Dataset phân chia bằng phương pháp cô lập chéo mã bệnh nhân (ID Isolation). Đảm bảo 100% hai bức ảnh cơ địa mắt trái/phải của cùng một bệnh nhân **tuyệt đối không bị rò rỉ (Patient Leakage)** chéo nhau giữa tập Train/Val/Test. Điều này giúp loại bỏ thiên kiến, đánh giá khách quan được sức mạnh thực sự của mô hình AI.

## 3. Pipeline Xử Lý Đồ Hoạ Y Tế Chi Tiết (Medical Image Enhancements)
Pipeline lập trình (`scripts/preprocess_enhance.py`) quy định toàn bộ ảnh quét đáy mắt phải qua 3 công đoạn tiêu chuẩn nghiêm ngặt trước khi truyền cho mô hình AI.

### Bước 3.1: ROI Cropping (Đã xử lý ra `archive/preprocessed_images/`)
- Mọi hình ảnh y tế thô đa độ phân giải đều được phát hiện viền đen tối và tiến hành cô lập vùng chữ nhật bao chứa thuần túy mô mềm của nhãn cầu (Region of Interest).
- Sau cắt, định dạng ép chuẩn hoá kích thước `512x512` pixel.

### Bước 3.2: Ben Graham Color Normalization
- Kỹ thuật **Chuẩn hoá ánh sáng của Ben Graham** (Nhà vô địch Kaggle Diabetic Retinopathy).
- **Thuật toán**: Tạo mặt nạ vùng làm mờ (Gaussian Blur Sigma=1/6), sau đó trừ vào ma trận ảnh gốc và tịnh tiến lên scale trung bình (128). 
- **Tác dụng**: Cân bằng ánh sáng viền mô ảnh, loại bỏ sạch hiện tượng mờ sáng/tối không đồng đều (Device Bias) do khác chế độ flash của các loại máy soi màng cứng võng mạc khác nhau gây ra.

### Bước 3.3: Contrast Limited Adaptive Histogram Equalization (CLAHE)
- **Thuật toán**: Ảnh tịnh tiến sang hệ màu LAB, tiến hành áp dụng CLAHE lên rãnh kênh sáng (kênh L) với `clip_limit=2.0` và lưới chia ma trận cục bộ (Tile Grid) là `8x8`.
- **Tác dụng**: Nâng cao cấp số nhân độ sắc nét và tương phản cục bộ, tô đậm các cụm vạch **Vi mạch máu võng mạc (Blood vessels)**, hiển lộ rõ **Đĩa thị giác (Optic Disc)** và nhận diện chính xác các vết xuất huyết, ổ viêm rách siêu nhỏ cực trị (Microaneurysms).
- **Đầu ra**: Xuất thành công hơn 6,300 tệp tin đã cường hoá vào `archive/enhanced_images/`.

## 4. Giai Đoạn Đợi: Tăng Cường Dữ Liệu Random Augmentation (`src/transforms.py`)
Mã nguồn thiết lập cho AI (Dùng thư viện GPU Albumentations định tuyến Transform) đã chuẩn bị xong các bộ lọc biến đổi ngẫu nhiên tại kì Huấn Luyện (Training State) nhằm triêt tiêu Overfitting và phục vụ cân bằng tập dữ liệu:

- **Bộ Scale kích thước**: `224x224` (dành cho backbone CNN tiêu chuẩn) và `384x384` (phục vụ đặc chế cho Swin Transformer).
- **Biến hình hình học (Geometric Transformation)**: Horizontal/Vertical Flips (Lật ảnh), Rotation90 (Xoay 90 độ), ShiftScaleRotate (Sự biến thiên tịnh tiến góc tới).
- **Biến màu ảnh (Color Adjustments)**: RandomBrightnessContrast, Random HueSaturationValue, Gaussian Blur.
- **Regularization Mới**: Bổ dụng đặc tính `CoarseDropout` (Xóa bỏ che khuất ngẫu nhiên một mảng nhiễu điểm ảnh) đè bù thay cho kĩ thuật CutMix/Cutout đơn giảm, tránh sự học quá mức của Neural.
- Đóng gói hoá `ImageNet Statistics Normalization (Mean/Std)` và xuất Tensor `ToTensorV2()` cho PyTorch chuẩn bị kết nạp.

## 5. Xử Lý Mất Cân Bằng Dữ Liệu

Bộ dữ liệu ODIR-5K có mất cân bằng nhãn nghiêm trọng: nhãn đa số (N, D) chiếm ~33% mỗi loại trong khi nhãn thiểu số như Hypertension (H) chỉ chiếm **3.2%** — tỉ lệ chênh lệch gần **1:10**. Nếu không xử lý, mô hình sẽ thiên về dự đoán nhãn đa số và bỏ qua các bệnh thiểu số vốn có giá trị lâm sàng cao.

Hệ thống triển khai **hai lớp xử lý song song**:

---

### 5.1 Lớp 1 (Chính): `pos_weight` trong Hàm Mất Mát BCEWithLogitsLoss

Đây là cơ chế **trực tiếp và chính** để xử lý mất cân bằng trong bài toán multi-label classification.

**Nguyên lý toán học:**

Hàm mất mát `BCEWithLogitsLoss` có tham số `pos_weight` cho phép gán trọng số lớn hơn cho mẫu dương thiểu số:

```
Loss(i) = -[ pos_weight[i] × y × log(σ(x)) + (1-y) × log(1 - σ(x)) ]
```

Trọng số được tính theo công thức:

```
pos_weight[i] = số mẫu âm [i] / số mẫu dương [i]
```

**Giá trị pos_weight thực tế (từ `archive/splits_clean/metadata.json`):**

| Nhãn | Tên bệnh | Số mẫu (+) | pos_weight |
|------|----------|-----------|------------|
| N | Normal | 1.478 | 2.02 |
| D | Diabetes Retinopathy | 1.486 | 2.00 |
| G | Glaucoma | 285 | 14.66 |
| C | Cataract | 288 | 14.49 |
| A | Age-related Macular Deg. | 215 | 19.75 |
| **H** | **Hypertension** | **132** | **32.80** |
| M | Pathological Myopia | 199 | 21.42 |
| O | Other Diseases | 1.110 | 3.02 |

Ý nghĩa: mỗi mẫu dương của nhãn H được tính nặng **gấp 32.8 lần** trong hàm loss so với mẫu âm — buộc mô hình phải chú ý đến nhãn thiểu số ngay từ lúc huấn luyện.

---

### 5.2 Lớp 2 (Bổ trợ): MixUp Augmentation

**Nguồn gốc lý thuyết:**

MixUp được đề xuất bởi Zhang et al. (2018) trong bài báo *"mixup: Beyond Empirical Risk Minimization"* (ICLR 2018). Kỹ thuật này giải quyết hạn chế của **Empirical Risk Minimization (ERM)** — nguyên lý huấn luyện truyền thống chỉ tối thiểu hóa loss trên đúng các điểm dữ liệu thực tế:

```
ERM: min_f (1/n) × Σ L(f(xᵢ), yᵢ)
```

ERM khiến mô hình không biết cách xử lý các điểm nằm **giữa** hai mẫu dữ liệu → dễ overfit. MixUp thay thế bằng **Vicinal Risk Minimization (VRM)**: học cả trên các điểm nội suy tuyến tính giữa hai mẫu ngẫu nhiên.

**Công thức nội suy:**

```
x̃ = λ × xᵢ + (1 - λ) × xⱼ       (hình ảnh)
ỹ = λ × yᵢ + (1 - λ) × yⱼ       (nhãn bệnh — soft labels)
ã = λ × aᵢ + (1 - λ) × aⱼ       (tuổi — regression)
```

với hệ số trộn `λ ~ Beta(α, α)`, `λ ∈ [0, 1]`.

**Vai trò của tham số α = 0.4:**

| Giá trị α | Hành vi của λ | Mức độ trộn |
|-----------|--------------|-------------|
| α → 0 | λ gần 0 hoặc 1 | Hầu như không trộn |
| α = 0.2 | λ thường ở 0.8–0.9 | Trộn nhẹ |
| **α = 0.4** | **λ phân tán vừa phải** | **Trộn vừa ← dự án dùng** |
| α = 1.0 | λ đều trong [0,1] | Trộn mạnh |

Giá trị α = 0.4 được chọn vì: đủ mạnh để regularize nhưng không làm "nhoà" cấu trúc y tế đặc thù (mạch máu, vùng tổn thương) của ảnh fundus.

Ngoài ra, code bổ sung `λ = max(λ, 1-λ)` để đảm bảo ảnh A luôn chiếm tỉ lệ **≥ 0.5** — nhãn hỗn hợp luôn gần với nhãn gốc hơn, ổn định quá trình huấn luyện.

**MixUp can thiệp ở cấp độ Batch (Collate Function)**, không phải từng ảnh đơn lẻ:

```
image_mix  = λ × image_A  + (1-λ) × image_B      [3 × H × W]
labels_mix = λ × labels_A + (1-λ) × labels_B     [8]  ← soft labels
age_mix    = λ × age_A    + (1-λ) × age_B         [1]
```

**Ví dụ minh họa** (λ = 0.7):
- Bệnh nhân A: Glaucoma `[0,0,1,0,0,0,0,0]`, 55 tuổi
- Bệnh nhân B: Normal `[1,0,0,0,0,0,0,0]`, 70 tuổi
- Kết quả: nhãn `[0.3, 0, 0.7, 0, 0, 0, 0, 0]`, tuổi `0.7×55 + 0.3×70 = 59.5`

Mô hình học: *"ảnh này 70% là Glaucoma, 30% là Normal"* → **decision boundary mềm hơn, ít overfit hơn**.

**Tại sao MixUp hỗ trợ xử lý mất cân bằng (gián tiếp)?**

- Khi trộn 1 ảnh H (thiểu số) với ảnh bất kỳ, tạo ra **mẫu tổng hợp mang đặc trưng H** → mô hình được "thấy" đặc trưng thiểu số nhiều hơn theo cách gián tiếp.
- Soft labels ngăn mô hình overfit hoàn toàn vào nhãn đa số → giảm hiện tượng bỏ qua nhãn thiểu số.
- Cải thiện **calibration**: xác suất dự đoán khớp thực tế hơn.

> **Lưu ý quan trọng**: MixUp là kỹ thuật *regularization/augmentation* — vai trò chính là giảm overfitting và cải thiện tổng quát hóa. Giải pháp **trực tiếp** cho mất cân bằng là `pos_weight` ở Lớp 1. Hai lớp này bổ trợ nhau để đạt hiệu quả tối ưu.

---

### 5.3 Lớp 3 (Bổ trợ): CutMix Augmentation

**Nguồn gốc lý thuyết:**

CutMix được đề xuất bởi Yun et al. (2019) trong bài báo *"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"* (ICCV 2019). Kỹ thuật này là sự phát triển của MixUp: thay vì trộn pixel **toàn bộ ảnh**, CutMix **cắt một vùng hình chữ nhật** từ ảnh B và **dán đè** lên ảnh A.

**Cơ chế hoạt động:**

```
1. Lấy λ₀ ~ Beta(α, α)  →  tính kích thước vùng cắt:
       W_cut = W × √(1 - λ₀)
       H_cut = H × √(1 - λ₀)

2. Chọn tâm (cx, cy) ngẫu nhiên trong ảnh
       x1 = max(0, cx - W_cut/2),  x2 = min(W, cx + W_cut/2)
       y1 = max(0, cy - H_cut/2),  y2 = min(H, cy + H_cut/2)

3. Dán vùng B vào A:
       image_mix[:, y1:y2, x1:x2] ← image_B[:, y1:y2, x1:x2]

4. Tính λ THỰC TẾ từ diện tích vùng cắt:
       λ = 1 - (x2-x1)(y2-y1) / (W × H)

5. Trộn nhãn và tuổi theo λ thực tế:
       labels_mix = λ × labels_A + (1-λ) × labels_B
       age_mix    = λ × age_A    + (1-λ) × age_B
```

**Tham số α = 1.0** (chuẩn theo paper gốc): λ₀ phân bố đều trong [0,1] → kích thước vùng cắt ngẫu nhiên đa dạng, mô hình tiếp xúc với nhiều quy mô cắt khác nhau.

**Ví dụ minh họa** (λ thực tế = 0.7, tức vùng B chiếm 30% diện tích):
- Ảnh A (Hypertension): vùng ngoài hình chữ nhật vẫn nguyên vẹn
- Ảnh B (Normal): vùng hình chữ nhật được dán vào
- Nhãn: `[N=0.3, H=0.7, ...]` — 70% Hypertension, 30% Normal
- Tuổi: `0.7×55 + 0.3×65 = 57.0`

**Đặc điểm phân biệt với MixUp:**

| Tiêu chí | MixUp | CutMix |
|---|---|---|
| Cơ chế ảnh | Trộn pixel toàn ảnh | Cắt-dán vùng chữ nhật |
| Pixel ngoài vùng cắt | Là hỗn hợp (trung gian) | Nguyên vẹn từ ảnh A |
| Cấu trúc cục bộ | Bị làm mờ | Được bảo toàn |
| Phù hợp ảnh y tế | Tốt | **Tốt hơn** (mạch máu, tổn thương không bị mờ) |
| Mô hình học | Soft boundary toàn ảnh | Localizable features |
| α chuẩn | 0.4 | 1.0 |

**Tại sao CutMix phù hợp hơn với ảnh fundus?**

Trong ảnh đáy mắt, các đặc trưng bệnh lý như vi mạch máu, xuất huyết điểm (microaneurysms), đĩa thị giác (optic disc) đều có **cấu trúc không gian cục bộ**. MixUp làm mờ pixel toàn ảnh có thể làm nhòa các cấu trúc này. CutMix giữ nguyên từng vùng → mô hình vẫn học được đặc trưng cục bộ từ vùng không bị cắt của ảnh A, đồng thời học vị trí (localization) của đặc trưng.

**Tại sao CutMix hỗ trợ mất cân bằng (gián tiếp)?**

- Vùng bệnh H (thiểu số) được "nhúng" vào bối cảnh ảnh khác → mô hình học nhận diện đặc trưng H ngay cả khi nó chỉ chiếm một phần nhỏ của ảnh
- Tương tự MixUp: soft labels giảm bias về nhãn đa số

---

### 5.4 Trạng Thái Triển Khai

| Thành phần | File | Trạng thái |
|---|---|---|
| `pos_weight` cho BCEWithLogitsLoss | `archive/splits_clean/metadata.json` | ✅ Đã tính, sẵn sàng nạp |
| MixUp Collator (`MixUpCollator`) | `src/mixup.py` | ✅ Hoàn chỉnh, 26/26 tests PASS |
| CutMix Collator (`CutMixCollator`) | `src/cutmix.py` | ✅ Hoàn chỉnh, 37/37 tests PASS |
| `get_mixup_dataloader` | `src/mixup.py` | ✅ Sẵn sàng tích hợp |
| `get_cutmix_dataloader` | `src/cutmix.py` | ✅ Sẵn sàng tích hợp |
| Export package (`__init__.py`) | `src/__init__.py` | ✅ Export đầy đủ tất cả modules |

**Tương thích Multi-task**: cả MixUp, CutMix và `pos_weight` đều hỗ trợ đồng thời Classification (8 nhãn bệnh) và Regression (tuổi sinh học) — phù hợp với kiến trúc Multi-task Learning của dự án.

**Chiến lược sử dụng trong training**:
- `pos_weight` → luôn áp dụng (tích hợp trong hàm loss)
- MixUp hoặc CutMix → chọn 1 hoặc luân phiên theo epoch (thực nghiệm Ablation Study sẽ so sánh)
