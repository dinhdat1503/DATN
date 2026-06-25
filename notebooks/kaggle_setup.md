# Hướng Dẫn Kết Nối Source Code Local → Kaggle và Chạy Thực Nghiệm

---

## PHƯƠNG PHÁP: Dùng Kaggle API để sync code lên Kaggle Dataset

**Ý tưởng:** Upload toàn bộ `src/`, `configs/`, `train.py`, `splits_clean/` lên Kaggle như một **Dataset riêng**. Notebook trên Kaggle sẽ `import` code từ đó — không cần copy paste thủ công.

```
[Máy local]                    [Kaggle]
  src/ ──────────────────────→  Dataset: odir5k-code
  configs/                           ↓ mount vào notebook
  splits_clean/              /kaggle/input/odir5k-code/
  train.py                           ↓
                              sys.path.insert(0, '/kaggle/input/odir5k-code')
                              from src.models import build_model  ← import được!
```

---

## BƯỚC 0 — Lấy Kaggle API Key (chỉ làm 1 lần)

1. Vào **https://www.kaggle.com** → click avatar góc phải → **"Settings"**
2. Kéo xuống mục **"API"** → click **"Create New Token"**
3. File `kaggle.json` tự động download (chứa username + key)
4. Copy file này vào đúng vị trí:

```bash
mkdir -p ~/.config/kaggle
cp ~/Downloads/kaggle.json ~/.config/kaggle/kaggle.json
chmod 600 ~/.config/kaggle/kaggle.json
```

5. Kiểm tra:
```bash
cd /media/dinhdat/OD/DOANTOTNGHIEP/DOANTOTNGHIEP
source .venv/bin/activate
kaggle config view
# Phải hiện: username: your_username
```

---

## BƯỚC 1 — Upload Source Code lên Kaggle (1 lệnh)

```bash
cd /media/dinhdat/OD/DOANTOTNGHIEP/DOANTOTNGHIEP
source .venv/bin/activate

# Chạy script tự động
bash kaggle_upload/upload_code.sh
```

**Script này sẽ tự động:**
- Copy `src/`, `configs/`, `train.py`, `evaluate.py`, `splits_clean/` vào `kaggle_upload/odir5k-code/`
- Tạo `dataset-metadata.json`
- Upload lên Kaggle Dataset tên `odir5k-code`

**Output mong đợi:**
```
[1/6] Làm sạch thư mục upload...
[2/6] Copy source code...
[3/6] Copy splits_clean...
[4/6] Tạo dataset-metadata.json...
       Dataset ID: your_username/odir5k-code
[5/6] Upload lên Kaggle...
[6/6] HOÀN THÀNH!
  Dataset URL: https://www.kaggle.com/datasets/your_username/odir5k-code
```

> **Lần sau muốn cập nhật code:** Chạy lại `bash kaggle_upload/upload_code.sh` — script tự tạo version mới.

---

## BƯỚC 2 — Tạo Kaggle Notebook

1. Vào **https://www.kaggle.com/code** → **"New Notebook"**
2. Notebook trống xuất hiện, tên mặc định là "notebook..."

---

## BƯỚC 3 — Gắn Datasets vào Notebook

**Bên phải màn hình** có panel **"Input"** → click **"Add data"**:

### Dataset 1: ODIR-5K Public (ảnh gốc)
1. Click **"Add data"** → tab **"Datasets"**
2. Tìm: `ocular disease recognition odir5k`
3. Chọn: **"Ocular Disease Recognition ODIR5K"** (by andrewmvd)
4. Click **"Add"** ✅

### Dataset 2: Code của bạn (vừa upload)
1. Click **"Add data"** → tab **"Your Datasets"** (hoặc search `odir5k-code`)
2. Chọn: **"odir5k-code"**
3. Click **"Add"** ✅

**Kiểm tra:** Panel Input phải hiện 2 datasets:
```
📁 ocular-disease-recognition-odir5k
📁 odir5k-code
```

---

## BƯỚC 4 — Bật GPU

**Bên phải** → **"Session options"** (biểu tượng ⚙️):
- **Accelerator**: **GPU T4 x2** (hoặc T4 x1)
- Click **"Save"**

> Kaggle cho **30 giờ GPU miễn phí/tuần**. 3 EXP CNN mất khoảng 3-4 giờ.

---

## BƯỚC 5 — Upload Notebook

1. Trong notebook → **"File"** → **"Import Notebook"**
2. Upload file: `notebooks/odir5k_cnn_kaggle.ipynb`

---

## BƯỚC 6 — Kiểm tra đường dẫn trước khi chạy

**Chạy Cell 4 riêng trước** (không Run All ngay):

Nếu `RAW_DIR` sai, Cell 4 sẽ in ra cấu trúc thư mục thực tế:
```
❌ RAW_DIR không tồn tại! Kiểm tra cấu trúc:
  ocular-disease-recognition-odir5k/
    ODIR-5K/
      ODIR-5K/
        Training Images/   ← đây rồi!
```

Sửa `RAW_DIR` trong Cell 3 cho khớp, ví dụ:
```python
RAW_DIR = '/kaggle/input/ocular-disease-recognition-odir5k/ODIR-5K/ODIR-5K/Training Images'
```

---

## BƯỚC 7 — Chạy toàn bộ notebook

Sau khi Cell 4 xác nhận đường dẫn OK:
- **"Run All"** (hoặc Shift+Enter từng cell)

### Timeline ước tính (GPU T4):

| Cell | Công việc | Thời gian |
|------|-----------|-----------|
| Cell 1 | pip install timm... | 2 phút |
| Cell 2 | Check GPU | <1 giây |
| Cell 3 | Setup paths | <1 giây |
| Cell 4 | Kiểm tra files | <1 giây |
| Cell 5 | ROI Crop 6,364 ảnh | 10-15 phút |
| Cell 6 | Ben Graham + CLAHE | 15-20 phút |
| Cell 7 | Load metadata | <1 giây |
| Cell 8 | Define training | <1 giây |
| **Cell 9** | **EXP 1 (20 epoch)** | **40-60 phút** |
| **Cell 10** | **EXP 2 (20 epoch)** | **40-60 phút** |
| **Cell 11** | **EXP 3 (20 epoch)** | **40-60 phút** |
| Cell 12 | Bảng so sánh | <1 giây |
| Cell 13 | Biểu đồ | <1 giây |
| **TỔNG** | | **~3-4 giờ** |

---

## BƯỚC 8 — Theo dõi training

Output của mỗi EXP sẽ hiện như sau:
```
=======================================================
  exp_1_cnn_no_preprocess
  img_dir=/kaggle/input/ocular-disease.../Training Images
=======================================================
[Aug] Không dùng MixUp/CutMix
Ep 01/20 | Train=0.4231 | Val F1=0.5124 AUC=0.7823 MAE=8.2y [52.1s]
Ep 02/20 | Train=0.3876 | Val F1=0.5436 AUC=0.8012 MAE=7.9y [51.8s]
...
TEST: F1=0.5623 AUC=0.8234 MAE=7.41y
```

---

## BƯỚC 9 — Lấy kết quả về

Sau khi xong, tab **"Output"** bên phải:
```
📁 results/
   exp_1_cnn_no_preprocess/
      results.json          ← Kết quả EXP 1
      best.pth              ← Model weights
   exp_2_cnn_preprocess_no_aug/
      results.json
   exp_3_cnn_preprocess_with_aug/
      results.json
   comparison_table.md      ← Bảng so sánh
   cnn_learning_curves.png  ← Biểu đồ
```

Download toàn bộ folder → giải nén vào `results/` trên máy local → chạy:
```bash
python evaluate.py
```

---

## XỬ LÝ LỖI THƯỜNG GẶP

| Lỗi | Nguyên nhân | Cách sửa |
|-----|-------------|---------|
| `OSError: [Errno 13] Permission denied: kaggle.json` | Quyền file sai | `chmod 600 ~/.config/kaggle/kaggle.json` |
| `401 - Unauthorized` | API key sai | Tạo lại token ở kaggle.com/settings |
| `FileNotFoundError: Training Images` | Đường dẫn RAW_DIR sai | Chạy Cell 4 để xem cấu trúc thật |
| `ModuleNotFoundError: src` | CODE_DIR sai | Kiểm tra `/kaggle/input/odir5k-code/` có file src/ không |
| `CUDA out of memory` | batch_size quá lớn | Sửa `batch=16` trong Cell 8 hàm `run_experiment` |
| `Kaggle notebook timeout` | Session 12h | Lưu checkpoint và resume |
