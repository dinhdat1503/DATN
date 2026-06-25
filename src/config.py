"""
Cấu hình & tiện ích môi trường cho ODIR-5K Phase 1 (Siamese nhị phân song nhãn).

File này chịu trách nhiệm:
1. Đọc file cấu hình YAML của từng thực nghiệm.
2. Cố định seed ngẫu nhiên (set_seed) để kết quả có thể tái lập.
3. Giải quyết đường dẫn dữ liệu (splits / ảnh / kết quả) tự động cho cả 2 môi trường:
   - Local (máy phát triển Ubuntu)
   - Kaggle Notebook (GPU T4 / P100) — nơi dữ liệu được mount tại /kaggle/input.

Không hardcode siêu tham số trong code — mọi thứ đọc từ YAML.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    """Đọc file cấu hình YAML và trả về dict.

    Args:
        config_path: Đường dẫn tới file .yaml của thực nghiệm.

    Returns:
        Dict chứa toàn bộ cấu hình thực nghiệm.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Cố định seed cho random / numpy / torch để đảm bảo khả năng tái lập kết quả.

    Args:
        seed: Giá trị seed (mặc định 42).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Ưu tiên tốc độ trên GPU (cuDNN benchmark) — chấp nhận sai khác rất nhỏ giữa các lần chạy
        torch.backends.cudnn.benchmark = True
    except ImportError:
        pass
    print(f"[Seed] Đã cố định seed = {seed}")


def is_kaggle() -> bool:
    """Phát hiện môi trường có phải Kaggle Notebook hay không."""
    return os.path.exists("/kaggle/working")


def _first_existing(candidates: list[Path]) -> Path | None:
    """Trả về đường dẫn đầu tiên tồn tại trong danh sách ứng viên (None nếu không có)."""
    for c in candidates:
        if c.exists():
            return c
    return None


def resolve_splits_dir(cfg: dict, project_root: Path) -> Path:
    """Xác định thư mục chứa train/val/test.csv + metadata.json.

    Thử lần lượt: đường dẫn trong config → bỏ tiền tố 'archive/' → các vị trí Kaggle phổ biến.
    """
    raw = cfg["splits_dir"]
    candidates = [
        project_root / raw,
        project_root / raw.replace("archive/", ""),
        Path(raw),
        Path("/kaggle/input/odir5k-code/splits_clean"),
        Path("/kaggle/input/odir5k-code") / Path(raw).name,
        Path("/kaggle/working/splits_clean"),
        Path("/kaggle/working/code/splits_clean"),
    ]
    found = _first_existing(candidates)
    if found is None:
        raise FileNotFoundError(
            f"Không tìm thấy thư mục splits. Đã thử: {[str(c) for c in candidates]}"
        )
    return found


def resolve_img_dir(cfg: dict, project_root: Path) -> Path:
    """Xác định thư mục chứa ảnh đáy mắt (raw hoặc enhanced).

    Hỗ trợ 2 trường hợp:
    - Ảnh enhanced/preprocessed: thư mục phẳng chứa các file <id>_left.jpg, <id>_right.jpg.
    - Ảnh raw ODIR gốc: nằm trong 'ODIR-5K/.../Training Images' → quét tự động dưới /kaggle/input.
    """
    raw = cfg["img_dir"]

    # 1) Thử trực tiếp theo cấu hình + biến thể bỏ tiền tố 'archive/'
    direct = _first_existing([
        project_root / raw,
        project_root / raw.replace("archive/", ""),
        Path(raw),
    ])
    if direct is not None:
        return direct

    # 2) Trường hợp ảnh raw ODIR gốc (thư mục 'Training Images') trên Kaggle
    if "Training Images" in raw:
        for base in ("/kaggle/input", "/kaggle/working"):
            if not os.path.exists(base):
                continue
            for root, dirs, _ in os.walk(base):
                if "Training Images" in dirs:
                    return Path(root) / "Training Images"

    # 3) Ảnh enhanced/preprocessed trên Kaggle — thử theo tên thư mục cuối
    name = Path(raw).name
    kaggle = _first_existing([
        Path("/kaggle/input/odir5k-code") / name,
        Path("/kaggle/working") / name,
        Path("/kaggle/working/code") / name,
    ])
    if kaggle is not None:
        return kaggle

    raise FileNotFoundError(
        f"Không tìm thấy thư mục ảnh '{raw}'. "
        f"Kiểm tra lại dataset đã mount trên Kaggle hoặc đường dẫn local."
    )


def resolve_results_dir(cfg: dict, project_root: Path) -> Path:
    """Xác định và tạo thư mục lưu kết quả (results/<exp_name>/).

    Trên Kaggle ghi vào /kaggle/working/results để có thể tải về sau khi chạy xong.
    """
    rel = cfg["output"]["results_dir"]
    if is_kaggle():
        out = Path("/kaggle/working/results") / rel.replace("results/", "")
    else:
        out = project_root / rel
    out.mkdir(parents=True, exist_ok=True)
    return out
