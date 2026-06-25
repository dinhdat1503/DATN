"""
Entry-point huấn luyện ODIR-5K Phase 1 — Siamese nhị phân song nhãn (Normal vs Pathological).

Quy trình: load config → cố định seed → resolve đường dẫn (Kaggle/local) →
tính focal_alpha tự động → build dataloaders song nhãn → build model Siamese →
build loss đa nhiệm → fit (two-stage + early stopping) → đánh giá Test.

Cách dùng:
    python train.py --config configs/exp_3_cnn_binary_enhanced_aug.yaml
    python train.py --config configs/exp_6_swin_binary_enhanced_aug.yaml --dry-run
    python train.py --config configs/exp_3_cnn_binary_enhanced_aug.yaml \
        --resume results/exp_3_cnn_binary_enhanced_aug/last_model.pth
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

# Đảm bảo import được package src.* khi chạy từ thư mục gốc
sys.path.insert(0, str(Path(__file__).parent))

from src.augment import BinocularAugmentCollator
from src.config import (
    is_kaggle,
    load_config,
    resolve_img_dir,
    resolve_results_dir,
    resolve_splits_dir,
    set_seed,
)
from src.dataset import build_dataloaders
from src.engine import evaluate_test, fit
from src.losses import MultiTaskLoss
from src.models import build_model


def compute_focal_alpha(train_csv: Path, age_min_filter: int = 5) -> tuple[float, int, int]:
    """Tính focal_alpha tự động = N_normal / N_total ở mức BỆNH NHÂN trên tập Train.

    Vì lớp Pathological (≈68%) nhiều hơn Normal (≈32%), alpha nhỏ (≈0.323) sẽ giảm trọng số
    lớp Pathological và tăng trọng số lớp Normal → cân bằng học, chống sụp đổ lớp.

    Returns:
        (alpha, n_normal, n_pathological)
    """
    df = pd.read_csv(train_csv)
    if age_min_filter > 0:
        df = df[df["Patient Age"] >= age_min_filter]
    g = df.groupby("ID").first()
    n_normal = int((g["N"] == 1).sum())
    n_path = int((g["N"] == 0).sum())
    total = max(n_normal + n_path, 1)
    return n_normal / total, n_normal, n_path


def main(config_path: str, dry_run: bool = False, resume: str | None = None) -> None:
    cfg = load_config(config_path)
    exp_name = cfg["experiment_name"]

    print(f"\n{'='*64}\n  THỰC NGHIỆM: {exp_name}\n  {cfg.get('description', '')}\n{'='*64}\n")

    # --- Seed & device ---
    seed = int(cfg["training"].get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device} | Kaggle={is_kaggle()}")

    # --- Đường dẫn ---
    project_root = Path(__file__).parent
    splits_dir = resolve_splits_dir(cfg, project_root)
    img_dir = resolve_img_dir(cfg, project_root)
    results_dir = resolve_results_dir(cfg, project_root)
    print(f"[Data] splits={splits_dir}\n[Data] images={img_dir}\n[Output] {results_dir}\n")

    tr = cfg["training"]
    img_size = int(tr["img_size"])
    age_min_filter = int(tr.get("age_min_filter", 5))

    # --- Focal alpha tự động ---
    l_cfg = cfg["loss"]
    alpha_cfg = str(l_cfg.get("focal_alpha", "auto")).lower().strip()
    if alpha_cfg == "auto":
        focal_alpha, n_normal, n_path = compute_focal_alpha(splits_dir / "train.csv", age_min_filter)
        print(f"[Focal] Train (BN): Normal={n_normal} ({focal_alpha*100:.1f}%) / "
              f"Pathological={n_path} ({(1-focal_alpha)*100:.1f}%) → alpha={focal_alpha:.4f}")
    else:
        focal_alpha = float(alpha_cfg)
        print(f"[Focal] alpha cố định = {focal_alpha}")

    # --- Collator tăng cường song nhãn (chỉ train) ---
    aug = cfg.get("augmentation", {})
    train_collate = None
    if aug.get("use_mixup", False) or aug.get("use_cutmix", False):
        train_collate = BinocularAugmentCollator(
            use_mixup=aug.get("use_mixup", False),
            use_cutmix=aug.get("use_cutmix", False),
            mixup_alpha=aug.get("mixup_alpha", 0.4),
            mixup_prob=aug.get("mixup_prob", 0.5),
            cutmix_alpha=aug.get("cutmix_alpha", 1.0),
            cutmix_prob=aug.get("cutmix_prob", 0.5),
            seed=seed,
        )
        print(f"[Augment] {train_collate}")

    # --- DataLoaders ---
    dataloaders, age_mean, age_std = build_dataloaders(
        splits_dir=splits_dir,
        img_dir=img_dir,
        img_size=img_size,
        batch_size=int(tr["batch_size"]),
        num_workers=int(tr.get("num_workers", 4)),
        pin_memory=(device.type == "cuda"),
        age_min_filter=age_min_filter,
        train_collate=train_collate,
    )
    print(f"[Loader] train={len(dataloaders['train'].dataset)} "
          f"val={len(dataloaders['val'].dataset)} test={len(dataloaders['test'].dataset)}\n")

    # --- Model ---
    m_cfg = cfg["model"]
    model = build_model(
        model_type=cfg["model_type"],
        pretrained=m_cfg.get("pretrained", True),
        img_size=img_size,
        dropout=m_cfg.get("dropout", 0.3),
    ).to(device)
    print(f"[Model] {model}\n")

    # --- Loss ---
    criterion = MultiTaskLoss(
        focal_alpha=focal_alpha,
        focal_gamma=float(l_cfg.get("focal_gamma", 2.0)),
        lam_age=float(l_cfg.get("lam_age", 0.05)),
    )

    # Lưu config dùng để chạy vào thư mục kết quả (truy vết)
    shutil.copy(config_path, results_dir / "config.yaml")

    # --- Train ---
    best_path = fit(
        model=model, dataloaders=dataloaders, criterion=criterion, cfg=cfg,
        device=device, results_dir=results_dir, age_mean=age_mean, age_std=age_std,
        dry_run=dry_run, resume=resume,
    )

    # --- Test ---
    if not dry_run:
        evaluate_test(
            model=model, dataloaders=dataloaders, criterion=criterion, device=device,
            age_mean=age_mean, age_std=age_std, results_dir=results_dir,
            exp_name=exp_name, best_path=best_path,
        )

    print(f"\n{'='*64}\n  HOÀN THÀNH: {exp_name}\n  Results: {results_dir}\n{'='*64}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ODIR-5K Phase 1 (Siamese nhị phân song nhãn)")
    parser.add_argument("--config", "-c", required=True, help="Đường dẫn file YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Chạy 1 batch để kiểm tra pipeline")
    parser.add_argument("--resume", "-r", default=None, help="Checkpoint để tiếp tục huấn luyện")
    args = parser.parse_args()
    main(config_path=args.config, dry_run=args.dry_run, resume=args.resume)
