# GIỚI THIỆU CHI TIẾT BỘ DỮ LIỆU ODIR-5K
**Đồ Án Tốt Nghiệp** | Phân loại nhị phân Siamese song nhãn ODIR-5K

Tài liệu này cung cấp toàn bộ số liệu thống kê chi tiết, cấu trúc dữ liệu và phương pháp xử lý của bộ dữ liệu **ODIR-5K** trong đồ án, phục vụ trực tiếp cho việc viết chương **"Cơ sở dữ liệu và Tiền xử lý dữ liệu"** trong báo cáo đồ án tốt nghiệp của bạn.

---

## 1. Tổng Quan Về Bộ Dữ Liệu ODIR-5K
*   **Tên đầy đủ:** Ocular Disease Intelligent Recognition (ODIR-5K).
*   **Nguồn gốc:** Được thu thập bởi Công ty Công nghệ Y tế Thượng Công (Shanggong Medical Technology Co., Ltd.) từ nhiều bệnh viện mắt khác nhau tại Trung Quốc.
*   **Đặc điểm hình ảnh:** Ảnh màu đáy mắt (color fundus images) chụp không xâm lấn qua đồng tử, chứa các cấu trúc giải phẫu học võng mạc (đĩa thị, hoàng điểm, mạch máu) và các biểu hiện bệnh lý.

---

## 2. Số Liệu Thống Kê Sau Khi Làm Sạch
Tập dữ liệu gốc sau khi loại bỏ 28 bản ghi lỗi (ảnh mờ, hỏng, không thể chẩn đoán lâm sàng) còn lại **3.343 bệnh nhân**, được gộp theo Patient ID:

*   **Tổng số bệnh nhân:** 3.343 người.
    *   **Bệnh nhân có đủ 2 mắt:** 3.021 người (chiếm **90.4%**).
    *   **Bệnh nhân bị khuyết 1 mắt:** 322 người (chiếm **9.6%**) — chỉ có ảnh mắt trái hoặc mắt phải.
*   **Tổng số ảnh đáy mắt thực tế:** 6.364 ảnh (3.021 × 2 + 322).

---

## 3. Phân Chia Dữ Liệu (Splits) Theo Bệnh Nhân (Patient ID)
Để chống rò rỉ thông tin (data leakage) giữa các tập, dữ liệu được chia cố định theo Patient ID (đảm bảo ảnh của cùng 1 bệnh nhân không nằm ở 2 tập khác nhau):

| Tập dữ liệu | Số lượng Bệnh nhân | Số lượng Ảnh mắt | Tỷ lệ (%) |
| :--- | :---: | :---: | :---: |
| **Huấn luyện (Train Set)** | 2.340 | 4.462 | 70% |
| **Kiểm thử giao đoạn (Val Set)** | 501 | 948 | 15% |
| **Kiểm thử độc lập (Test Set)** | 502 | 954 | 15% |
| **TỔNG CỘNG** | **3.343** | **6.364** | **100%** |

*Lưu ý:* Số lượng ảnh không bằng đúng 2 lần số lượng bệnh nhân vì có **9.6%** ca bị thiếu 1 mắt.

---

## 4. Định Nghĩa Nhãn Nhị Phân & Phân Bố Lớp (Mức Bệnh Nhân)
Bài toán quy đổi từ 8 nhãn gốc về nhị phân mức bệnh nhân nhằm phục vụ mục tiêu sàng lọc tuyến đầu:
*   **Lớp 0 (Normal - Khỏe mạnh):** Cả hai mắt của bệnh nhân đều hoàn toàn khỏe mạnh (cờ N = 1).
*   **Lớp 1 (Pathological - Bệnh lý):** Ít nhất một trong hai mắt có bất kỳ dấu hiệu bất thường nào (cờ N = 0).

### Phân bố lớp trên tập Huấn luyện (Train Set):
*   **Normal (y = 0):** 756 bệnh nhân (chiếm **32.3%**).
*   **Pathological (y = 1):** 1.584 bệnh nhân (chiếm **67.7%**).
*   ➔ Hệ số cân bằng lớp y sinh tự động tính toán trên Train:
    $$\alpha = \frac{N_{\text{normal}}}{N_{\text{total}}} = \frac{756}{2.340} \approx 0.3231$$

---

## 5. Phân Bố 8 Lớp Bệnh Gốc Của ODIR-5K (Mức Bệnh Nhân)

Dưới đây là bảng thống kê chi tiết số lượng bệnh nhân mắc từng loại bệnh trong 8 lớp bệnh gốc của ODIR-5K, được chia theo các tập Train, Val, Test và Tổng số:

| Ký hiệu | Tên bệnh (English) | Tên bệnh (Tiếng Việt) | Train Set | Val Set | Test Set | Tổng số bệnh nhân | Tỷ lệ (%) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **N** | Normal | Võng mạc khỏe mạnh | 756 | 162 | 162 | **1.080** | 32.3% |
| **D** | Diabetes | Bệnh võng mạc tiểu đường | 773 | 166 | 166 | **1.105** | 33.1% |
| **G** | Glaucoma | Bệnh glôcôm (Tăng nhãn áp) | 147 | 28 | 31 | **206** | 6.2% |
| **C** | Cataract | Đục thể thủy tinh | 148 | 28 | 32 | **208** | 6.2% |
| **A** | AMD | Thoái hóa hoàng điểm tuổi già | 109 | 21 | 32 | **162** | 4.8% |
| **H** | Hypertension | Bệnh võng mạc do tăng huyết áp | 67 | 14 | 22 | **103** | 3.1% |
| **M** | Myopia | Cận thị bệnh lý | 112 | 23 | 23 | **158** | 4.7% |
| **O** | Other | Các bệnh lý khác | 632 | 143 | 128 | **903** | 27.0% |

*Lưu ý:* Tổng số bệnh nhân mắc các bệnh lớn hơn quy mô thực tế (3.343 bệnh nhân) vì đây là tập dữ liệu đa nhãn gốc (một bệnh nhân có thể đồng thời mắc nhiều bệnh lý ở cả hai mắt).

---

## 6. Thống Kê Độ Tuổi Lâm Sàng

Độ tuổi của bệnh nhân đóng vai trò quan trọng trong tác vụ học đa nhiệm (Multi-task Learning) - dự đoán tuổi võng mạc để bổ trợ tính năng phân loại bệnh lý chính:
*   **Độ tuổi trung bình (Mean age):** **58.14 tuổi**.
*   **Độ lệch chuẩn (Std age):** **11.26 tuổi**.
*   **Độ tuổi nhỏ nhất (Min age):** 5 tuổi (các hồ sơ bệnh án ghi nhận nhỏ hơn 5 tuổi đã được tự động loại bỏ để tránh nhiễu do lỗi nhập liệu hành chính).
*   **Độ tuổi lớn nhất (Max age):** 91 tuổi.
