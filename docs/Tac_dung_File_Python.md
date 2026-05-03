# TÁC DỤNG VÀ MỨC ĐỘ QUAN TRỌNG CỦA CÁC FILE PYTHON
(Ngoại trừ `preprocess_enhance.py`)

Trong đồ án này, các file Python được chia làm 2 nhóm chính: **Nhóm chuẩn bị dữ liệu (Scripts)** và **Nhóm huấn luyện mô hình (Source - `src/`)**.

---

## 1. NHÓM CHUẨN BỊ DỮ LIỆU (`scripts/`)
*Mục đích: Chạy 1 lần duy nhất để tạo dữ liệu chuẩn trước khi train.*

### 1.1. `build_patient_splits.py`
- **Mức độ quan trọng:** ⭐⭐⭐⭐⭐ (Cực kỳ quan trọng)
- **Tác dụng:** Chia dữ liệu thành 3 tập Train (70%), Validation (15%), Test (15%).
- **Lý do quan trọng:** File này giải quyết vấn đề **Patient Leakage** (rò rỉ bệnh nhân). Nếu chia ngẫu nhiên từng ảnh, ảnh mắt trái ở tập Train, mắt phải ở tập Test → AI "học vẹt" đặc điểm cá nhân thay vì học bệnh lý → kết quả báo cáo bị sai lệch (ảo). File này gộp 2 mắt của 1 người lại và đảm bảo họ chỉ nằm ở 1 trong 3 tập. Nó cũng áp dụng phân tầng (Stratified) để đảm bảo tỉ lệ bệnh đồng đều giữa các tập.
- **Đầu vào:** `archive/full_df.csv`
- **Đầu ra:** `train.csv`, `val.csv`, `test.csv` trong `archive/splits/`

### 1.2. `check_preprocessing.py`
- **Mức độ quan trọng:** ⭐⭐⭐⭐ (Rất quan trọng)
- **Tác dụng:** Chạy 8 bài test tự động để kiểm định chất lượng dữ liệu sau khi tiền xử lý và chia split.
- **Lý do quan trọng:** Đóng vai trò như "người kiểm toán". Nó kiểm tra xem ảnh đã đúng 512x512 chưa, có ảnh nào bị lỗi không, có bị trùng lặp không, và đặc biệt là xác nhận 100% không có Patient Leakage. Đây là minh chứng kỹ thuật để đưa vào báo cáo gửi GVHD.
- **Đầu ra:** `check_result_utf8.txt`

### 1.3. `clean_and_rebuild.py`
- **Mức độ quan trọng:** ⭐⭐⭐ (Hữu ích)
- **Tác dụng:** Xóa sạch các thư mục dữ liệu đã sinh ra (`preprocessed_images/`, `enhanced_images/`, `splits/`) để chạy lại từ đầu nếu có lỗi.
- **Lý do quan trọng:** Giúp dọn dẹp workspace gọn gàng khi muốn làm lại mà không sợ sót file rác cũ.

---

## 2. NHÓM HUẤN LUYỆN MÔ HÌNH (`src/`)
*Mục đích: Được import vào file chính (ví dụ `train.py`) để chạy liên tục trong quá trình huấn luyện AI.*

### 2.1. `dataset.py`
- **Mức độ quan trọng:** ⭐⭐⭐⭐⭐ (Cực kỳ quan trọng - Trái tim của Data Pipeline)
- **Tác dụng:** Chuyển đổi dữ liệu từ dạng ổ cứng (ảnh JPEG + file CSV) thành Tensor (ma trận số) để GPU và mô hình PyTorch có thể hiểu được.
- **Chi tiết:**
  - Lớp `ODIRDataset`: Mỗi lần lấy 1 sample, nó đọc ảnh bằng OpenCV, áp dụng augmentation, đọc nhãn bệnh thành tensor `[8]`, đọc tuổi và chuẩn hóa (trừ mean chia std). (Đã thêm tính năng lọc tuổi < 5).
  - Hàm `get_dataloaders`: Đóng gói `ODIRDataset` thành các batch (ví dụ 32 ảnh/lần) để đẩy liên tục vào GPU lúc train.

### 2.2. `transforms.py`
- **Mức độ quan trọng:** ⭐⭐⭐⭐⭐ (Cực kỳ quan trọng - Chống Overfitting)
- **Tác dụng:** Định nghĩa các phép Augmentation (biến đổi hình ảnh ngẫu nhiên) dùng thư viện Albumentations.
- **Chi tiết:** Khi lấy ảnh vào tập Train, nó sẽ ngẫu nhiên: lật ngang/dọc, xoay, thay đổi độ sáng, làm mờ (Gaussian Blur), che khuất ngẫu nhiên (CoarseDropout), và chuẩn hóa theo phân bố ImageNet. Việc này giúp mô hình không bị "học vẹt" hình ảnh gốc mà học được bản chất bệnh lý.

### 2.3. `mixup.py` (Mới làm ở Mục 3)
- **Mức độ quan trọng:** ⭐⭐⭐⭐ (Nâng cao)
- **Tác dụng:** Kỹ thuật Augmentation mức Batch để xử lý mất cân bằng dữ liệu cực đoan (như bệnh Hypertension chỉ có 3.2%).
- **Chi tiết:** Nó lấy 2 ảnh bất kỳ và "trộn" chúng lại với nhau (cả hình ảnh, nhãn bệnh, và tuổi) theo tỷ lệ ngẫu nhiên. Giúp mô hình học được các "trạng thái chuyển tiếp" giữa bệnh hiếm và bệnh phổ biến.

### 2.4. `utils.py`
- **Mức độ quan trọng:** ⭐⭐⭐⭐ (Cần thiết)
- **Tác dụng:** Chứa các hàm toán học và tính toán chỉ số đánh giá (Metrics).
- **Chi tiết:**
  - `compute_class_weights`: Tính trọng số để bù đắp cho class hiếm (phạt nặng hơn nếu AI đoán sai bệnh hiếm).
  - Tính toán Mean/Std của Tuổi để chuẩn hóa (Z-score Normalization).
  - `compute_multilabel_metrics`: Tính chính xác các chỉ số như F1-Score, Precision, Recall cho bài toán có 8 bệnh.

---
**Tổng kết:** Nếu `scripts/` là "nhà bếp" chuẩn bị sơ chế nguyên liệu (1 lần), thì `src/` là "dây chuyền" liên tục đưa nguyên liệu sạch vào cỗ máy AI để học hỏi.
