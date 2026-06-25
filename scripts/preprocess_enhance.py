"""
Tiền xử lý ảnh nâng cao cho ODIR-5K.

Pipeline:
  1. Ben Graham color normalization  – chuẩn hóa ánh sáng, loại bỏ bias thiết bị
  2. CLAHE trên kênh L (LAB)         – tăng cường chi tiết mạch máu & cấu trúc

Input : archive/preprocessed_images/  (512×512, đã crop ROI)
Output: archive/enhanced_images/      (512×512, đã qua CLAHE + Ben Graham)

Sử dụng:
    python scripts/preprocess_enhance.py
    python scripts/preprocess_enhance.py --workers 4
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Hàm tiền xử lý ảnh
# ---------------------------------------------------------------------------

def ben_graham_normalization(
    img: np.ndarray,
    sigma_ratio: float = 1 / 6,
    scale: int = 128,
) -> np.ndarray:
    """Ben Graham color normalization.

    Trừ local average color rồi cộng 128 để chuẩn hóa ánh sáng.
    Phương pháp này giảm ảnh hưởng của khác biệt thiết bị chụp.

    Reference:
        Graham, B. (2015). Kaggle Diabetic Retinopathy competition.
    """
    h, w = img.shape[:2]
    sigma = int(max(h, w) * sigma_ratio)
    if sigma % 2 == 0:
        sigma += 1  # kernel size phải lẻ

    # Tính local average bằng Gaussian blur
    local_avg = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigmaX=sigma)

    # Trừ local average, cộng scale (128)
    result = img.astype(np.float32) - local_avg + scale

    # Clip về [0, 255]
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Áp dụng CLAHE trên kênh L (LAB color space).

    CLAHE tăng cường tương phản cục bộ, giúp làm nổi bật:
    - Mạch máu võng mạc
    - Đĩa thị giác (optic disc)
    - Các tổn thương nhỏ (microaneurysms, hard exudates)
    """
    # Chuyển sang LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Áp dụng CLAHE lên kênh L (luminance)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    # Ghép lại
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return result


def enhance_single_image(
    src_path: str,
    dst_path: str,
    apply_ben_graham: bool = True,
    apply_clahe_flag: bool = True,
    clahe_clip: float = 2.0,
    clahe_grid: tuple[int, int] = (8, 8),
    ben_graham_sigma: float = 1 / 6,
) -> tuple[str, bool, str]:
    """Xử lý 1 ảnh. Trả về (filename, success, error_msg)."""
    filename = os.path.basename(src_path)
    try:
        img = cv2.imread(src_path)
        if img is None:
            return filename, False, f"Khong doc duoc anh: {src_path}"

        # Bước 1: Ben Graham normalization
        if apply_ben_graham:
            img = ben_graham_normalization(img, sigma_ratio=ben_graham_sigma)

        # Bước 2: CLAHE
        if apply_clahe_flag:
            img = apply_clahe(img, clip_limit=clahe_clip, tile_grid_size=clahe_grid)

        # Lưu kết quả
        cv2.imwrite(dst_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return filename, True, ""

    except Exception as e:
        return filename, False, str(e)


# ---------------------------------------------------------------------------
# Hàm chính
# ---------------------------------------------------------------------------

def process_all_images(
    src_dir: Path,
    dst_dir: Path,
    workers: int = 1,
    apply_ben_graham: bool = True,
    apply_clahe_flag: bool = True,
    clahe_clip: float = 2.0,
    clahe_grid: tuple[int, int] = (8, 8),
    ben_graham_sigma: float = 1 / 6,
) -> None:
    """Xử lý tất cả ảnh trong thư mục nguồn."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Lấy danh sách ảnh
    image_files = sorted([
        f for f in src_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    if not image_files:
        print(f"[LOI] Khong tim thay anh nao trong {src_dir}")
        sys.exit(1)

    print(f"Tim thay {len(image_files)} anh trong {src_dir}")
    print(f"Luu ket qua vao {dst_dir}")
    print(f"Ben Graham: {'CO' if apply_ben_graham else 'KHONG'}")
    print(f"CLAHE: {'CO' if apply_clahe_flag else 'KHONG'} "
          f"(clip={clahe_clip}, grid={clahe_grid})")
    print(f"Workers: {workers}")
    print()

    errors = []

    if workers <= 1:
        # Chạy tuần tự
        for img_path in tqdm(image_files, desc="Enhancing"):
            dst_path = str(dst_dir / img_path.name)
            fname, ok, err = enhance_single_image(
                str(img_path), dst_path,
                apply_ben_graham, apply_clahe_flag,
                clahe_clip, clahe_grid, ben_graham_sigma,
            )
            if not ok:
                errors.append((fname, err))
    else:
        # Chạy song song
        futures = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for img_path in image_files:
                dst_path = str(dst_dir / img_path.name)
                future = executor.submit(
                    enhance_single_image,
                    str(img_path), dst_path,
                    apply_ben_graham, apply_clahe_flag,
                    clahe_clip, clahe_grid, ben_graham_sigma,
                )
                futures[future] = img_path.name

            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Enhancing"):
                fname, ok, err = future.result()
                if not ok:
                    errors.append((fname, err))

    # Báo cáo kết quả
    print()
    total = len(image_files)
    success = total - len(errors)
    print(f"HOAN TAT: {success}/{total} anh thanh cong")

    if errors:
        print(f"[CANH BAO] {len(errors)} anh bi loi:")
        for fname, err in errors[:10]:
            print(f"  - {fname}: {err}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tien xu ly anh nang cao ODIR-5K (CLAHE + Ben Graham)"
    )
    parser.add_argument(
        "--src", type=str,
        default="archive/preprocessed_images",
        help="Thu muc anh nguon (da crop ROI, 512x512)",
    )
    parser.add_argument(
        "--dst", type=str,
        default="archive/enhanced_images",
        help="Thu muc luu anh da enhance",
    )
    parser.add_argument("--workers", type=int, default=1, help="So luong workers")
    parser.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clip limit")
    parser.add_argument("--clahe-grid", type=int, default=8, help="CLAHE tile grid size")
    parser.add_argument("--no-ben-graham", action="store_true", help="Bo qua Ben Graham")
    parser.add_argument("--no-clahe", action="store_true", help="Bo qua CLAHE")
    parser.add_argument(
        "--ben-graham-sigma", type=float, default=1/6,
        help="Ty le sigma cho Ben Graham (default: 1/6 kich thuoc anh)",
    )
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not src_dir.exists():
        print(f"[LOI] Thu muc nguon khong ton tai: {src_dir}")
        sys.exit(1)

    process_all_images(
        src_dir=src_dir,
        dst_dir=dst_dir,
        workers=args.workers,
        apply_ben_graham=not args.no_ben_graham,
        apply_clahe_flag=not args.no_clahe,
        clahe_clip=args.clahe_clip,
        clahe_grid=(args.clahe_grid, args.clahe_grid),
        ben_graham_sigma=args.ben_graham_sigma,
    )


if __name__ == "__main__":
    main()
