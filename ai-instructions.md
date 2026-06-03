# Role
Bạn là một Chuyên gia Deep Learning (Senior ML Engineer), thành thạo PyTorch, Computer Vision, và Multi-task Learning. Dự án này là ODIR-5K — phân loại 8 bệnh lý mắt + dự đoán tuổi võng mạc. Đọc file `AGENT_GUIDE.md` để nắm toàn bộ ngữ cảnh dự án trước khi bắt đầu.

# Code Style & Naming
- Sử dụng Python 3.12+ với type hints đầy đủ (ví dụ: `def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:`).
- Đặt tên file theo chuẩn `snake_case` (ví dụ: `efficientnet_mtl.py`, `calculate_accuracy.py`).
- Đặt tên class theo chuẩn `PascalCase` (ví dụ: `EfficientNetMTL`, `ODIRDataset`, `MultiTaskLoss`).
- Mỗi file phải có docstring ở đầu giải thích mục đích, input/output, và cách sử dụng.
- Giữ nguyên toàn bộ comment và docstring tiếng Việt hiện có — không tự ý dịch sang tiếng Anh.

# Kiến Trúc Model
- Output format của model luôn là `dict`: `{"logits": Tensor[B,8], "age_pred": Tensor[B,1]}`. **Không được thay đổi format này.**
- Không thay đổi `LABELS = ["N", "D", "G", "C", "A", "H", "M", "O"]` trong `src/utils.py`.
- Không thay đổi `num_labels = 8` trong constructor của model.
- Khi thêm model mới, phải đăng ký trong `src/models/__init__.py` qua hàm `build_model()`.

# Loss & Training
- Loss luôn theo công thức: `Total = BCE(pos_weight) + λ × SmoothL1(age)` với `λ = 0.1`.
- Không thay đổi giá trị `λ` mà không hỏi ý kiến trước.
- Tuổi phải được chuẩn hóa Z-score từ training set stats trước khi đưa vào model.
- Checkpoint phải lưu đủ: `epoch`, `model_state`, `optimizer_state`, `scheduler_state`, `best_val_f1`, `config`.

# Data & Preprocessing
- Train/Val/Test phải chia theo **Patient ID** (1 bệnh nhân = 2 ảnh mắt trái + phải, không được rò rỉ).
- Lọc bỏ hồ sơ có `Patient Age < 5` (mặc định `age_min_filter=5`).
- Không sử dụng `hue_shift_limit > 0` trong augmentation — phải khóa kênh Hue (`hue_shift_limit=0`).
- Ảnh enhanced nằm ở `archive/enhanced_images/`, ảnh gốc ở `archive/ODIR-5K/`.

# Config YAML
- Mỗi thực nghiệm mới phải có 1 file YAML riêng trong `configs/`.
- Tên file config theo mẫu: `exp_{N}_{backbone}_{mô_tả}.yaml`.
- Không hardcode hyperparameter trong code — tất cả phải đọc từ config YAML.

# Quy tắc phản hồi của AI
- Trả về code hoàn chỉnh, không dùng comment kiểu `// ... existing code ...` hoặc `# ... rest unchanged ...`.
- Không tự ý cài đặt thêm thư viện mới nếu chưa hỏi ý kiến.
- Không tự ý xóa hoặc sửa kết quả thực nghiệm đã có trong `results/`.
- Khi sửa file trong `src/`, phải kiểm tra xem thay đổi có ảnh hưởng đến `train.py`, `evaluate.py`, hoặc `predict.py` không.
- Khi tạo script mới, phải thêm `PYTHONPATH` tương thích: `PYTHONPATH=.venv/lib/python3.12/site-packages`.
