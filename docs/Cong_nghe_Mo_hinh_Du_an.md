# CÔNG NGHỆ VÀ MÔ HÌNH ĐANG SỬ DỤNG TRONG DỰ ÁN ODIR-5K

Tài liệu này tổng hợp các công nghệ, kỹ thuật, mô hình học máy và vai trò của từng thành phần trong dự án, dựa trên code hiện có trong:
- `scripts/`
- `src/`
- các file kết quả/ghi chú đi kèm (`check_result_utf8.txt`, `decuong_extracted.txt`)

---

## 1) Tổng quan pipeline kỹ thuật của dự án

Pipeline hiện tại có 3 lớp chính:

1. **Tiền xử lý offline (data engineering + image enhancement)**
   - `scripts/preprocess_enhance.py`
   - `scripts/build_patient_splits.py`
   - `scripts/check_preprocessing.py`
   - `scripts/clean_and_rebuild.py`

2. **Nạp dữ liệu cho huấn luyện (training input pipeline)**
   - `src/dataset.py`
   - `src/transforms.py`

3. **Tiện ích cho huấn luyện/đánh giá**
   - `src/utils.py`

Ý nghĩa: dự án đang ở trạng thái rất mạnh về phần **data pipeline** và **input pipeline**, là nền tảng để train mô hình deep learning cho bài toán multi-label + regression tuổi.

---

## 2) Công nghệ (libraries/frameworks) đang được sử dụng

## 2.1 Nhóm xử lý dữ liệu

- **Python 3**
  - Ngôn ngữ chính cho toàn bộ scripts và modules.
- **pandas**
  - Đọc/ghi CSV, groupby theo bệnh nhân (`ID`), thống kê labels/tuổi.
  - Dùng nhiều trong `build_patient_splits.py`, `check_preprocessing.py`, `clean_and_rebuild.py`, `dataset.py`, `utils.py`.
- **numpy**
  - Xử lý mảng số học, tính sigma, mean/std, random generator.
- **scikit-learn** (`train_test_split`)
  - Chia tập patient-level có stratify trong `clean_and_rebuild.py`.

## 2.2 Nhóm xử lý ảnh y tế

- **OpenCV (cv2)**
  - Đọc/ghi ảnh, đổi hệ màu (BGR/RGB, BGR/LAB), GaussianBlur, CLAHE.
  - Đây là công nghệ trung tâm của file `scripts/preprocess_enhance.py`.
- **Pillow (PIL)**
  - Dùng để kiểm tra kích thước ảnh trong `check_preprocessing.py`.

## 2.3 Nhóm deep learning

- **PyTorch (`torch`)**
  - Định nghĩa tensor labels/age, DataLoader, Dataset.
  - Được dùng rõ trong `src/dataset.py`, `src/utils.py`.
- **Albumentations**
  - Augmentation online cho train/val/test (`src/transforms.py`).
- **albumentations.pytorch.ToTensorV2**
  - Chuyển nhanh từ ndarray sang tensor PyTorch.

## 2.4 Nhóm hệ thống/chạy script

- **argparse**: quản lý CLI arguments cho scripts.
- **pathlib**: quản lý đường dẫn file/folder an toàn.
- **ProcessPoolExecutor**: song song hóa tiền xử lý ảnh (`preprocess_enhance.py`).
- **tqdm**: progress bar khi xử lý hàng loạt.
- **json**: lưu metadata/summary.

---

## 3) Các mô hình học máy trong dự án (theo mức độ bằng chứng)

## 3.1 Đã có code hỗ trợ trực tiếp (implemented)

### A. Bài toán multi-label classification (8 nhãn)
- Nhãn: `N, D, G, C, A, H, M, O`.
- Bằng chứng:
  - `LABELS` trong `src/utils.py`, `src/dataset.py`, `scripts/*.py`.
  - `compute_class_weights()` tính `pos_weight = neg/pos` cho `BCEWithLogitsLoss`.
- Vai trò:
  - Dự đoán đồng thời nhiều bệnh trên cùng 1 ảnh fundus (multi-label), thay vì single-class.

### B. Bài toán age regression (dự đoán tuổi)
- Bằng chứng:
  - `ODIRDataset.__getitem__()` trả về trường `age`.
  - Có chuẩn hóa tuổi bằng `age_mean`, `age_std`.
  - `compute_age_stats()` trong `src/utils.py` và `clean_and_rebuild.py`.
- Vai trò:
  - Tạo bối cảnh multi-task: mô hình học đặc trưng hình ảnh vừa cho bệnh, vừa cho tuổi.

### C. Multi-task learning (classification + regression)
- Bằng chứng:
  - Docstring `src/dataset.py` nêu rõ "Multi-task learning".
  - Mẫu dữ liệu trả về gồm `image`, `labels`, `age`.
- Vai trò:
  - Khai thác shared representation trên fundus để cải thiện khả năng tổng quát.

## 3.2 Đã được thiết kế hỗ trợ, nhưng chưa thấy file model/train cụ thể trong workspace

### A. CNN backbones (ResNet, EfficientNet)
- Bằng chứng:
  - `src/transforms.py` ghi chú `224x224` cho CNN (ResNet/EfficientNet).
- Trạng thái:
  - **Chưa thấy file model definition** (`nn.Module`) hoặc training loop trong workspace hiện có.

### B. Swin Transformer
- Bằng chứng:
  - `src/transforms.py` ghi chú `384x384` cho Swin-T.
  - `decuong_extracted.txt` nêu hướng so sánh CNN vs Swin.
- Trạng thái:
  - **Chưa thấy code model/train của Swin** trong các file `.py` hiện tại.

=> Kết luận chính xác theo code: dự án đã sẵn sàng data/input cho các backbone này, nhưng phần "model huấn luyện" không nằm trong workspace hiện tại (hoặc chưa được thêm vào).

---

## 4) "Mô hình"/kỹ thuật tiền xử lý dữ liệu và vai trò

Lưu ý: trong preprocessing, "mô hình" ở đây là **thuật toán xử lý ảnh + chiến lược chia dữ liệu**, không phải neural network.

## 4.1 Ben Graham normalization (`ben_graham_normalization`)

- Vị trí: `scripts/preprocess_enhance.py`.
- Cơ chế:
  1. Tạo local average illumination bằng Gaussian blur.
  2. `result = image - local_avg + 128`.
  3. Clip về [0, 255].
- Vai trò trong dự án:
  - Chuẩn hóa ánh sáng/contrast giữa các máy chụp khác nhau.
  - Giảm domain bias (model học nhầm điều kiện chụp thay vì tổn thương).

## 4.2 CLAHE trên kênh L của LAB (`apply_clahe`)

- Vị trí: `scripts/preprocess_enhance.py`.
- Cơ chế:
  - Đổi BGR -> LAB, CLAHE chỉ trên kênh độ sáng L, giữ nguyên thông tin màu A/B.
- Vai trò:
  - Tăng tương phản cục bộ, làm rõ mạch máu và lesion nhỏ.
  - Hạn chế biến dạng màu so với việc can thiệp trực tiếp trên RGB/BGR.

## 4.3 Image enhancement pipeline toàn bộ (`process_all_images`)

- Vị trí: `scripts/preprocess_enhance.py`.
- Cơ chế:
  - Quét toàn bộ ảnh nguồn, xử lý từng ảnh, ghi output, có thể chạy đa tiến trình.
- Vai trò:
  - Biến bộ ảnh về chất lượng ổn định trước khi train.

## 4.4 Patient-level stratified split (`split_ids_by_stratum` / `create_patient_level_splits`)

- Vị trí:
  - `scripts/build_patient_splits.py`
  - `scripts/clean_and_rebuild.py`
- Cơ chế:
  - Gom theo `ID` bệnh nhân.
  - Xác định nhãn chủ đạo để stratify.
  - Chia train/val/test theo bệnh nhân, không chia theo từng ảnh.
- Vai trò:
  - Chặn **patient leakage** (left/right của cùng bệnh nhân rơi vào 2 tập khác nhau).
  - Bảo toàn phân bố nhãn giữa các tập.

## 4.5 Data cleaning (remove age outliers)

- Vị trí: `scripts/clean_and_rebuild.py`.
- Cơ chế:
  - Loại bỏ bệnh nhân có tuổi < ngưỡng (`min_age`, mặc định 5).
- Vai trò:
  - Cắt bỏ nhiễu nhãn regression không hợp lý (vd tuổi = 1) để age-task ổn định hơn.

## 4.6 Data QA script (`check_preprocessing.py`)

- 8 nhóm kiểm tra: schema, tuổi, labels, duplicate, left/right consistency, image existence, image size, split leakage.
- Vai trò:
  - Đảm bảo chất lượng data đầu vào trước huấn luyện.
  - Giảm rủi ro "train ra kết quả đẹp nhưng sai bản chất".

---

## 5) Trước và sau tiền xử lý: tác động thực tế đến dự án

Theo `scripts/check_result_utf8.txt` và `archive/splits/summary.json`:

### Đạt được
- 6392/6392 ảnh có mặt trên disk và khớp CSV.
- Mẫu kiểm tra kích thước cho thấy ảnh đồng nhất 512x512.
- Split train/val/test không có patient leakage.
- Tỉ lệ split gần 70/15/15 và phân bố nhãn giữa các tập tương đối sát.

### Vấn đề còn tồn tại
- Có 28 dòng tuổi bất thường (`Patient Age = 1`) trong bộ đang kiểm tra.
- Nghĩa là nếu dùng bộ `archive/splits/` hiện tại cho age-task thì chất lượng regression có thể bị ảnh hưởng.

### Hướng xử lý
- Chạy `clean_and_rebuild.py` để tạo bộ cleaned split (loại outlier tuổi, cập nhật metadata).
- Sau đó train trên bộ cleaned để kết quả multi-task ổn định hơn.

---

## 6) Vai trò của từng thành phần trong bức tranh tổng thể

- `preprocess_enhance.py`: nâng chất lượng tín hiệu ảnh y tế.
- `build_patient_splits.py`: tạo split đúng phương pháp khoa học, tránh leakage.
- `check_preprocessing.py`: đặt "quality gate" trước train.
- `clean_and_rebuild.py`: dọn dẹp metadata và tạo bộ dữ liệu train-ready.
- `src/dataset.py`: biến CSV + image thành mini-batch có cấu trúc multi-task.
- `src/transforms.py`: tạo đa dạng mẫu online, giảm overfitting.
- `src/utils.py`: cung cấp công cụ loss-weight, age normalize, metrics.

---

## 7) Kết luận cho mục "Công nghệ và mô hình"

1. Dự án hiện tại dùng bộ công nghệ "thực chiến" cho medical imaging: OpenCV + pandas + PyTorch + Albumentations.
2. Phần preprocessing và split được xây dựng khá đầy đủ, đúng hướng nghiệp vụ (patient-level, anti-leakage).
3. Bài toán đã được thiết kế theo hướng multi-task (bệnh + tuổi).
4. Backbone cụ thể (ResNet/EfficientNet/Swin) được nhắc đến và đã có support input-size, nhưng chưa thấy code model/train trong workspace hiện tại.
5. Nếu bổ sung module model + training script, nên ưu tiên tiếp tục sử dụng bộ cleaned split và metadata (class weights, age stats) để tối ưu hiệu quả.

---

## 8) Phụ lục: map nhanh "công nghệ -> file"

- `OpenCV`: `scripts/preprocess_enhance.py`, `src/dataset.py`
- `pandas/numpy`: hầu hết scripts + `src/utils.py`
- `scikit-learn`: `scripts/clean_and_rebuild.py`
- `PyTorch`: `src/dataset.py`, `src/utils.py`
- `Albumentations`: `src/transforms.py`
- `Pillow`: `scripts/check_preprocessing.py`
- `tqdm + multiprocessing`: `scripts/preprocess_enhance.py`

---

Tài liệu được tạo theo code trong workspace tại thời điểm hiện tại (27/04/2026). Nếu bạn thêm file model/train, có thể cập nhật phần 3 thành "mô hình đã implement" chi tiết hơn (backbone, head, loss, optimizer, scheduler, metric theo epoch).
