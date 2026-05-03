# Giải Thích Chi Tiết Tệp Dữ Liệu ODIR-5K và Các Bước Tiền Xử Lý

Tài liệu này được lập ra để giúp bạn nắm vững cấu trúc của bộ dữ liệu gốc cũng như ý nghĩa của các bước tiền xử lý đã thực hiện, phục vụ cho quá trình làm đồ án và bảo vệ trước hội đồng.

---

## Phần 1: Ý nghĩa các trường (cột) trong tệp dữ liệu gốc (data.xlsx / full_df.csv)

Tệp dữ liệu lưu trữ thông tin chi tiết của các bệnh nhân trong bộ ODIR-5K. Dưới đây là ý nghĩa của từng cột:

### 1. Nhóm thông tin cơ bản của bệnh nhân
*   **`ID`**: Mã số định danh duy nhất của từng bệnh nhân.
*   **`Patient Age`**: Tuổi thực tế của bệnh nhân. (Dùng làm biến mục tiêu - *Target Variable* cho bài toán **Hồi quy** dự đoán tuổi).
*   **`Patient Sex`**: Giới tính của bệnh nhân (Male - Nam / Female - Nữ).

### 2. Nhóm thông tin hình ảnh và chẩn đoán ban đầu
*   **`Left-Fundus` / `Right-Fundus`**: Tên tệp tin ảnh chụp võng mạc (đáy mắt) của mắt trái và mắt phải tương ứng.
*   **`Left-Diagnostic Keywords` / `Right-Diagnostic Keywords`**: Chẩn đoán gốc của bác sĩ cho từng mắt bằng văn bản (ví dụ: *cataract* - đục thủy tinh thể, *normal fundus* - đáy mắt bình thường).

### 3. Nhóm các cột nhãn bệnh lý (Kỹ thuật One-hot Encoding)
Mỗi cột dưới đây đại diện cho một loại bệnh lý. Giá trị `1` nghĩa là có mắc bệnh, `0` là không mắc. Đây là biến mục tiêu cho bài toán **Phân loại đa nhãn (Multi-label Classification)**:
*   **`N` (Normal)**: Bình thường (Không mắc bệnh gì).
*   **`D` (Diabetic Retinopathy)**: Bệnh võng mạc đái tháo đường.
*   **`G` (Glaucoma)**: Bệnh cườm nước (tăng nhãn áp).
*   **`C` (Cataract)**: Đục thủy tinh thể.
*   **`A` (Age-related Macular Degeneration - AMD)**: Thoái hóa điểm vàng do tuổi tác.
*   **`H` (Hypertensive Retinopathy)**: Bệnh võng mạc do cao huyết áp.
*   **`M` (Pathological Myopia)**: Cận thị bệnh lý.
*   **`O` (Other)**: Các bệnh lý nhãn khoa hoặc bất thường khác.

### 4. Nhóm các cột phục vụ cho Lập trình (Đầu vào cho mô hình AI)
*   **`filepath`**: Đường dẫn trỏ tới vị trí lưu bức ảnh đó trên hệ thống file. Giúp mã nguồn (code) tự động load ảnh.
*   **`labels`**: Mảng chứa ký hiệu các bệnh mà ảnh đó mắc phải (ví dụ: `['N']` hoặc `['D', 'M']`).
*   **`target`**: Vectơ mục tiêu gộp 8 cột bệnh lý thành 1 mảng. Ví dụ `[1, 0, 0, 0, 0, 0, 0, 0]` nghĩa là nhãn `N`. Vectơ này được đưa trực tiếp vào hàm Loss của PyTorch (như `BCEWithLogitsLoss`) để AI tính toán mức độ lỗi.
*   **`filename`**: Tên file rút gọn của bức ảnh.

> **Tóm lại quá trình huấn luyện:** AI sẽ đọc ảnh từ đường dẫn ở cột `filepath` (Đầu vào **X**), và học cách dự đoán ra vectơ ở cột `target` cùng với số ở cột `Patient Age` (Đầu ra **Y**).

---

## Phần 2: Sự khác biệt Giữa Trước và Sau khi Tiền xử lý dữ liệu (Data Preprocessing)

Ảnh y tế gốc thường chứa rất nhiều nhiễu sáng, độ tương phản kém và không đồng nhất về kích thước. Quy trình tiền xử lý 3 bước cốt lõi đã biến dữ liệu "thô" thành dữ liệu "sạch" giúp AI học tập hiệu quả.

### Bước 1: ROI Cropping (Cắt vùng quan tâm)
*   **Trước khi xử lý:** Ảnh gốc có kích thước lộn xộn, bao quanh nhãn cầu là những viền đen (khoảng tối) rất lớn vô ích.
*   **Sau khi xử lý:** Thuật toán nhận diện ranh giới nhãn cầu để cắt bỏ viền đen. Ảnh được chuẩn hóa về kích thước `512x512`. 
*   **Tác dụng:** Ép mô hình AI chỉ tập trung (Attention) vào phần mô nhãn cầu, bỏ qua vùng đen dư thừa, giúp tiết kiệm bộ nhớ GPU và tăng tốc độ hội tụ của thuật toán.

### Bước 2: Ben Graham Color Normalization (Chuẩn hóa màu)
*   **Trước khi xử lý:** Ảnh bị nhiễu do thiết bị chụp (*Device Bias*). Có bức chụp flash quá sáng, bức lại quá tối, màu sắc võng mạc sai lệch giữa các phòng khám khác nhau.
*   **Sau khi xử lý:** Áp dụng bộ lọc mờ (Gaussian Blur) kết hợp trừ với ma trận ảnh gốc.
*   **Tác dụng:** Loại bỏ hoàn toàn bóng râm đổ sai và sự chênh lệch ánh sáng. Tất cả các bức ảnh được đưa về một thang đo ánh sáng cân bằng, màu sắc đồng nhất.

### Bước 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
*   **Trước khi xử lý:** Độ tương phản ảnh yếu. Các vi mạch máu (blood vessels) và tổn thương liti bị chìm vào phần nền, rất khó nhận diện.
*   **Sau khi xử lý:** Phân chia ảnh thành lưới `8x8` và cân bằng biểu đồ màu cục bộ (thường áp dụng trên rãnh L của hệ màu LAB).
*   **Tác dụng:** Hình ảnh trở nên vô cùng sắc nét. Mạng lưới mạch máu võng mạc nổi bật, đĩa thị giác (Optic Disc) sáng rõ. Việc làm rõ mạch máu đặc biệt có giá trị cực cao đối với nhánh mô hình giải quyết bài toán "Dự đoán tuổi sinh học".
