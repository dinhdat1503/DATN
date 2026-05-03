#!/bin/bash
# ============================================================
# Script tự động upload source code lên Kaggle Dataset
# Chạy: bash kaggle_upload/upload_code.sh
# ============================================================

set -e  # Dừng nếu có lỗi

PROJECT_ROOT="/media/dinhdat/OD/DOANTOTNGHIEP/DOANTOTNGHIEP"
UPLOAD_DIR="$PROJECT_ROOT/kaggle_upload/odir5k-code"
SPLITS_DIR="$PROJECT_ROOT/archive/splits_clean"

source "$PROJECT_ROOT/.venv/bin/activate"

echo "======================================"
echo " KAGGLE UPLOAD SCRIPT — ODIR-5K Code"
echo "======================================"

# ------ Bước 1: Làm sạch thư mục upload ------
echo "[1/6] Làm sạch thư mục upload..."
rm -rf "$UPLOAD_DIR"
mkdir -p "$UPLOAD_DIR"

# ------ Bước 2: Copy source code ------
echo "[2/6] Copy source code..."
cp -r "$PROJECT_ROOT/src"      "$UPLOAD_DIR/src"
cp -r "$PROJECT_ROOT/configs"  "$UPLOAD_DIR/configs"
cp    "$PROJECT_ROOT/train.py"    "$UPLOAD_DIR/"
cp    "$PROJECT_ROOT/evaluate.py" "$UPLOAD_DIR/"

# Copy splits_clean vào cùng package
echo "[3/6] Copy splits_clean..."
cp -r "$SPLITS_DIR" "$UPLOAD_DIR/splits_clean"

echo "       Số file trong upload:"
find "$UPLOAD_DIR" -type f | wc -l

# ------ Bước 3: Tạo dataset-metadata.json ------
echo "[4/6] Tạo dataset-metadata.json..."
# Lấy username từ dòng '- username: ngodinhdatcpp' → cột thứ 3
KAGGLE_USER=$(kaggle config view | grep 'username' | awk '{print $3}')
if [ -z "$KAGGLE_USER" ]; then
    echo "  ERROR: Chưa cấu hình Kaggle API key!"
    echo "  Xem hướng dẫn: notebooks/kaggle_setup.md"
    exit 1
fi

cat > "$UPLOAD_DIR/dataset-metadata.json" << JSON
{
  "title": "odir5k-code",
  "id": "${KAGGLE_USER}/odir5k-code",
  "licenses": [{"name": "CC0-1.0"}]
}
JSON
echo "       Dataset ID: ${KAGGLE_USER}/odir5k-code"

# ------ Bước 4: Upload / Update ------
echo "[5/6] Upload lên Kaggle..."
cd "$UPLOAD_DIR"

# Thử update trước (nếu dataset đã tồn tại)
if kaggle datasets status "${KAGGLE_USER}/odir5k-code" &>/dev/null; then
    echo "       Dataset đã tồn tại → tạo version mới..."
    kaggle datasets version -p "$UPLOAD_DIR" --dir-mode zip -m "Update code $(date +%Y%m%d-%H%M)"
else
    echo "       Dataset chưa tồn tại → tạo mới..."
    kaggle datasets create -p "$UPLOAD_DIR" --dir-mode zip
fi

echo "[6/6] HOÀN THÀNH!"
echo ""
echo "  Dataset URL: https://www.kaggle.com/datasets/${KAGGLE_USER}/odir5k-code"
echo "  Trong Kaggle notebook, code sẽ ở: /kaggle/input/odir5k-code/"
echo ""
echo "  Thêm vào đầu notebook:"
echo "    import sys"
echo "    sys.path.insert(0, '/kaggle/input/odir5k-code')"
