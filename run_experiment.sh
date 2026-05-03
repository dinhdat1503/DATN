#!/bin/bash
# ============================================================
# Run Ablation Study: Raw vs Enhanced Preprocessing
# ============================================================
# Usage:
#   bash run_experiment.sh              # Chạy cả 2 thực nghiệm
#   bash run_experiment.sh exp_raw      # Chỉ chạy baseline
#   bash run_experiment.sh exp_enhanced # Chỉ chạy enhanced
# ============================================================

set -e  # Dừng nếu có lỗi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Kích hoạt virtual environment ---
if [ -d ".venv" ]; then
    echo "[Setup] Kích hoạt .venv..."
    source .venv/bin/activate
else
    echo "[WARN] Không tìm thấy .venv — dùng Python hệ thống"
fi

echo ""
echo "============================================================"
echo "  ODIR-5K Ablation Study: Preprocessing Comparison"
echo "  Thời gian bắt đầu: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# --- Kiểm tra dependencies ---
python -c "import torch, yaml, albumentations, cv2, sklearn" 2>/dev/null || {
    echo "[ERROR] Thiếu dependencies. Cài đặt:"
    echo "  pip install pyyaml scikit-learn tqdm"
    exit 1
}

# Hàm chạy một thực nghiệm
run_exp() {
    local EXP_NAME=$1
    local CONFIG="configs/${EXP_NAME}.yaml"

    if [ ! -f "$CONFIG" ]; then
        echo "[ERROR] Không tìm thấy config: $CONFIG"
        return 1
    fi

    echo "────────────────────────────────────────────────────────────"
    echo "  Bắt đầu: $EXP_NAME"
    echo "  Config:  $CONFIG"
    echo "  Bắt đầu: $(date '+%H:%M:%S')"
    echo "────────────────────────────────────────────────────────────"

    python train.py --config "$CONFIG" 2>&1 | tee "results/${EXP_NAME}/train.log" || {
        mkdir -p "results/${EXP_NAME}"
        python train.py --config "$CONFIG" 2>&1 | tee "results/${EXP_NAME}/train.log"
    }

    echo ""
    echo "  ✅ Hoàn thành: $EXP_NAME  ($(date '+%H:%M:%S'))"
    echo ""
}

# --- Chạy thực nghiệm ---
TARGET="${1:-all}"  # Nhận argument hoặc mặc định "all"

case "$TARGET" in
    exp_1)
        run_exp exp_1_no_preprocessing
        ;;
    exp_2)
        run_exp exp_2_preprocess_no_aug
        ;;
    exp_3)
        run_exp exp_3_preprocess_with_aug
        ;;
    all)
        run_exp exp_1_no_preprocessing
        run_exp exp_2_preprocess_no_aug
        run_exp exp_3_preprocess_with_aug
        ;;
    *)
        echo "[ERROR] Argument không hợp lệ: $TARGET"
        echo "Usage: bash run_experiment.sh [exp_1|exp_2|exp_3|all]"
        exit 1
        ;;
esac

# --- So sánh kết quả ---
echo "============================================================"
echo "  So sánh kết quả..."
echo "============================================================"
python evaluate.py 2>&1

echo ""
echo "============================================================"
echo "  HOÀN THÀNH"
echo "  Thời gian kết thúc: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Kết quả: results/comparison_table.md"
echo "============================================================"
