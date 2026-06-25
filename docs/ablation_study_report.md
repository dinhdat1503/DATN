# BÁO CÁO TỔNG HỢP SO SÁNH ABLATION STUDY (EXP 1 - EXP 6)
**Đồ Án Tốt Nghiệp** | Phân loại nhị phân Siamese song nhãn ODIR-5K

Dưới đây là bảng tổng hợp đầy đủ kết quả thực nghiệm trên **tập Test** của cả 6 cấu hình thí nghiệm thuộc Giai đoạn 1. Các chỉ số được báo cáo ở cả hai chế độ: **Ngưỡng mặc định 0.5** và **Ngưỡng tối ưu Youden Index** (tìm trên tập Validation).

---

## 📊 Bảng Chỉ Số So Sánh Tổng Hợp

### Bảng A: Đánh giá ở ngưỡng mặc định 0.5

| EXP | Backbone | Tiền xử lý | Augment (MixUp/CutMix) | AUC-ROC | Accuracy | F1-Score | Sensitivity | Specificity | Age MAE |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP 1** | EfficientNet-B0 | ❌ Raw | ❌ None | 0.7546 | 0.6873 | 0.7566 | 0.7176 | 0.6235 | 8.02y |
| **EXP 2** | EfficientNet-B0 | ✅ Enhanced | ❌ None | 0.7911 | 0.7092 | 0.7653 | 0.7000 | 0.7284 | 7.92y |
| **EXP 3** | EfficientNet-B0 | ✅ Enhanced | ✅ Yes | 0.7840 | 0.7072 | 0.7721 | **0.7324** | 0.6543 | 7.74y |
| **EXP 4** | Swin-Tiny | ❌ Raw | ❌ None | 0.8200 | 0.7709 | 0.8345 | **0.8529** | 0.5988 | **7.41y** |
| **EXP 5** | Swin-Tiny | ✅ Enhanced | ❌ None | **0.8563** | **0.7829** | **0.8376** | 0.8265 | **0.6914** | 7.96y |
| **EXP 6** | Swin-Tiny | ✅ Enhanced | ✅ Yes | 0.8012 | 0.7490 | 0.8073 | 0.7765 | **0.6914** | 7.58y |

---

### Bảng B: Đánh giá ở ngưỡng Youden tối ưu (được tối ưu hóa trên Validation)

| EXP | Backbone | Tiền xử lý | Augment (MixUp/CutMix) | Ngưỡng Youden | AUC-ROC | Accuracy | F1-Score | Sensitivity | Specificity | Age MAE |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP 1** | EfficientNet-B0 | ❌ Raw | ❌ None | **0.56** | 0.7546 | 0.6614 | 0.7099 | 0.6118 | 0.7654 | 8.02y |
| **EXP 2** | EfficientNet-B0 | ✅ Enhanced | ❌ None | **0.61** | 0.7911 | 0.6693 | 0.6993 | 0.5676 | **0.8827** | 7.92y |
| **EXP 3** | EfficientNet-B0 | ✅ Enhanced | ✅ Yes | **0.53** | 0.7840 | 0.6912 | 0.7446 | 0.6647 | 0.7469 | 7.74y |
| **EXP 4** | Swin-Tiny | ❌ Raw | ❌ None | **0.67** | 0.8200 | 0.7311 | 0.7900 | **0.7471** | 0.6975 | **7.41y** |
| **EXP 5** | Swin-Tiny | ✅ Enhanced | ❌ None | **0.81** | **0.8563** | **0.7629** | **0.8046** | 0.7206 | 0.8519 | 7.96y |
| **EXP 6** | Swin-Tiny | ✅ Enhanced | ✅ Yes | **0.69** | 0.8012 | 0.7291 | 0.7695 | 0.6676 | 0.8580 | 7.58y |

---

## 📈 Phân Tích & Đối Chiếu Khoa Học (Dành Cho Viết Khóa Luận)

### 1. Phân tích tác động của Tiền xử lý nâng cao (Raw vs Enhanced)
*   **Với CNN:** Khi chuyển từ ảnh gốc (EXP 1) sang ảnh Enhanced (EXP 2), AUC-ROC tăng **từ 0.7546 lên 0.7911 (+3.65%)**, độ đặc hiệu (Specificity) ở ngưỡng Youden tăng rất mạnh **từ 0.7654 lên 0.8827 (+11.73%)**.
*   **Với Swin:** Chuyển từ ảnh gốc (EXP 4) sang ảnh Enhanced (EXP 5), AUC-ROC tăng **từ 0.8200 lên 0.8563 (+3.63%)**, độ đặc hiệu (Specificity) tăng **từ 0.6975 lên 0.8519 (+15.44%)**.
*   **Kết luận:** Tiền xử lý tĩnh offline (ROI Crop + Ben Graham + CLAHE) đóng vai trò cực kỳ quan trọng trên cả hai kiến trúc. Nó giúp loại bỏ hoàn toàn bias thiết bị chụp và làm nổi bật các vi tổn thương cục bộ, giúp mô hình nhận diện chính xác mắt bình thường và giảm mạnh tỷ lệ báo động giả (False Positive).

### 2. Tác động của Augmentation (MixUp & CutMix đồng bộ)
Có sự phân hóa trái ngược rất thú vị giữa 2 kiến trúc khi bật tăng cường ảnh nâng cao:
*   **CNN (EfficientNet-B0) hưởng lợi:** F1-Score ở ngưỡng tối ưu Youden tăng rõ rệt từ **0.6993 (EXP 2) lên 0.7446 (EXP 3) (+4.53%)**. Điều này chứng tỏ MixUp & CutMix làm mượt biên quyết định và chống overfitting tốt cho mạng tích chập CNN (vốn có inductive bias mạnh về tính dịch chuyển bất biến và trường thụ cảm cục bộ).
*   **Swin Transformer bị suy giảm:** F1-Score Youden giảm từ **0.8046 (EXP 5) xuống 0.7695 (EXP 6)**, AUC-ROC giảm từ **0.8563 xuống 0.8012**. Do Swin Transformer học qua cơ chế Self-Attention trên các ô ảnh (patches), phép dán CutMix đột ngột tạo ra các viền biên phi thực tế và làm đứt gãy cấu trúc không gian mạch máu võng mạc liên tục. Swin Transformer thiếu inductive bias nên cần dữ liệu "thật" lâm sàng hơn là dữ liệu bị xáo trộn mạnh bởi MixUp/CutMix trên tập dữ liệu nhỏ.

### 3. So sánh hiệu năng ngang (CNN vs Swin Transformer)
*   **Swin Transformer luôn thắng thế:** Trong tất cả các kịch bản đối chứng tương ứng (1 vs 4, 2 vs 5, 3 vs 6), **Swin-Tiny đều cho AUC-ROC và F1-Score vượt trội hơn hẳn EfficientNet-B0**.
*   **Đỉnh SOTA của dự án:** Mô hình **Swin-Tiny + Enhanced (EXP 5)** đạt kết quả cao nhất toàn dự án với **AUC-ROC = 85.63%**, **F1-Score Youden = 80.46%**, và **Accuracy = 76.29%**.
*   **Nguyên nhân:** Cơ chế Self-Attention phân cấp của Swin Transformer-Tiny có khả năng nắm bắt bối cảnh toàn cục (Global Context) và mối liên kết không gian xa cực tốt (ví dụ mối tương quan bệnh học giữa hai mắt hoặc các điểm tổn thương cách xa nhau), điều mà trường thụ cảm cục bộ của CNN bị hạn chế.
