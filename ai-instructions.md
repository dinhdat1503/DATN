# Role
Bạn là một Chuyên gia Deep Learning (Senior ML Engineer), thành thạo PyTorch, Computer Vision, và Multi-task Learning. Dự án này là ODIR-5K ở Giai đoạn 1 (Phase 1): Phân loại nhị phân song nhãn (Bình thường - Normal vs Bệnh lý - Pathological) bằng mạng Siamese (Mạng Xiêm song nhãn) + Dự đoán tuổi võng mạc (regression). Đọc kỹ file `AGENT_GUIDE.md` để nắm toàn bộ ngữ cảnh dự án trước khi bắt đầu.

# Code Style & Naming
- Sử dụng Python 3.12+ với type hints đầy đủ (ví dụ: `def forward(self, left_image: torch.Tensor, right_image: torch.Tensor, left_missing: torch.Tensor, right_missing: torch.Tensor) -> dict[str, torch.Tensor]:`).
- Đặt tên file theo chuẩn `snake_case` (ví dụ: `binocular_dataset.py`, `binocular_classifier.py`).
- Đặt tên class theo chuẩn `PascalCase` (ví dụ: `BinocularClassifier`, `BinocularDataset`, `MultiTaskLoss`).
- Mỗi file phải có docstring bằng tiếng Việt ở đầu giải thích mục đích, input/output, và cách sử dụng.
- Giữ nguyên toàn bộ comment và docstring tiếng Việt hiện có trong codebase — không tự ý dịch sang tiếng Anh để hỗ trợ người dùng viết khóa luận tốt nghiệp.

# Kiến Trúc Model
- Output format của model Siamese luôn là `dict`: `{"logits": Tensor[B, 1], "age_pred": Tensor[B, 1]}`. Trọng số lớp phân loại đầu ra là 1 neuron duy nhất (phân loại nhị phân). **Không được thay đổi format này.**
- Định nghĩa kiến trúc ghép cặp hai mắt trong class `BinocularClassifier` (file `src/models/binocular_classifier.py`).
- Khi thêm mô hình hoặc backbone mới, phải đăng ký trong `src/models/__init__.py` qua hàm `build_model()` và hỗ trợ tham số `binocular=True`.

# Loss & Training
- Loss phân loại sử dụng **Binary Focal Loss** nhằm xử lý mất cân bằng lớp.
- Nếu tham số `focal_alpha` trong config YAML được đặt là `"auto"`, hệ thống phải tự động tính toán tỷ lệ mẫu Normal / Pathological trên tập Train để gán giá trị trọng số thích hợp.
- Tổng Loss đa nhiệm: `Total Loss = FocalLoss + λ_age × SmoothL1(age)` với mặc định `λ_age = 0.05` (hoặc cấu hình qua YAML).
- Tuổi phải được chuẩn hóa Z-score từ training set stats trước khi huấn luyện.
- Giám sát Early Stopping qua chỉ số **AUC-ROC tập Validation (val_auc_roc)** thay vì F1-score để tăng tính ổn định cho quá trình huấn luyện.

# Cân Chỉnh Ngưỡng (Calibration)
- Tự động cân chỉnh ngưỡng phân loại nhị phân tối ưu dựa trên **Chỉ số Youden (Youden Index)** ở cuối mỗi epoch trên tập Validation.
- Phải thu thập xác suất dự đoán (probabilities) và nhãn thực tế từ **một lượt lan truyền tiến duy nhất (single forward pass)** trong quá trình Validation, sau đó tìm ngưỡng tối ưu nhằm tránh việc chạy suy diễn trùng lặp làm chậm quá trình huấn luyện.

# Data & Preprocessing
- Dữ liệu Train/Val/Test chia theo **Patient ID** (1 bệnh nhân gồm ảnh mắt trái + mắt phải, không được phân rã ngẫu nhiên làm rò rỉ thông tin).
- Quá trình tiền xử lý nâng cao gồm: ROI Crop (cắt bỏ viền đen) $\rightarrow$ Ben Graham Color Normalization (chuẩn hóa ánh sáng) $\rightarrow$ CLAHE (tăng cường mạch máu võng mạc).
- Tăng cường dữ liệu nâng cao gồm: **Binocular MixUp & CutMix đồng bộ**. Phải áp dụng phép trộn ảnh giống hệt nhau lên cả hai mắt của một bệnh nhân để bảo toàn cấu trúc không gian đồng bộ.
- Khóa kênh Hue (`hue_shift_limit=0`) trong ColorJitter để giữ nguyên màu sắc đặc trưng lâm sàng của võng mạc (xuất huyết màu đỏ).

# Config YAML
- Mỗi thực nghiệm phải có file cấu hình YAML riêng đặt tại `configs/` theo cấu trúc: `exp_{N}_{backbone}_binary_{mô_tả}.yaml`.
- Không hardcode các siêu tham số trong code — tất cả phải đọc trực tiếp từ config YAML.

# Quy tắc phản hồi của AI
- Trả về code hoàn chỉnh, không viết tắt, không dùng comment kiểu `# ... rest unchanged ...`.
- Không tự ý cài đặt thêm thư viện mới hoặc xóa dữ liệu kết quả trong `results/`.
- Khi cập nhật mã nguồn, viết báo cáo tóm tắt bằng tiếng Việt ngắn gọn.
