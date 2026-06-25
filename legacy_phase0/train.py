"""
Training script cho ODIR-5K 3-Way Ablation Study.

Thực nghiệm:
    EXP 1: Ảnh gốc, không xử lý gì
    EXP 2: Ảnh enhanced (ROI+BenGraham+CLAHE), không MixUp/CutMix
    EXP 3: Ảnh enhanced + MixUp + CutMix

Usage:
    python train.py --config configs/exp_1_no_preprocessing.yaml
    python train.py --config configs/exp_2_preprocess_no_aug.yaml
    python train.py --config configs/exp_3_preprocess_with_aug.yaml
    python train.py --config configs/exp_3_preprocess_with_aug.yaml --dry-run
    python train.py --config configs/exp_2_preprocess_no_aug.yaml --resume results/exp_2/last_model.pth
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
import yaml

# Thêm project root vào sys.path để import src.*
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.cutmix import CutMixCollator
from src.dataset import ODIRDataset, get_dataloaders
from src.loss import MultiTaskLoss
from src.mixup import MixUpCollator
from src.models import build_model
from src.transforms import get_transforms
from src.utils import LABELS, compute_multilabel_metrics, get_pos_weight_from_metadata, load_metadata, find_best_thresholds
from src.binocular_dataset import BinocularDataset, get_binocular_dataloaders
from src.binocular_augment import BinocularAugmentCollator

# Cố gắng import sklearn cho AUC-ROC
try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[CANH BAO] sklearn không có — AUC-ROC sẽ không tính được")

# Cố gắng import tqdm cho progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_train_loader(
    dataset: ODIRDataset,
    aug_cfg: dict,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    use_weighted_sampler: bool = False,
) -> torch.utils.data.DataLoader:
    """Tạo DataLoader cho train set, hỗ trợ MixUp và/hoặc CutMix.

    Logic ưu tiên:
        - use_mixup=True + use_cutmix=True → dùng CutMixCollator
          (CutMix bao gồm cả bước tương tự MixUp ở level batch;
           trong thực tế dùng 1 trong 2 hoặc xen kẽ ngẫu nhiên)
        - use_mixup=True only → MixUpCollator
        - use_cutmix=True only → CutMixCollator
        - Cả 2 đều False → DataLoader chuẩn
    """
    import torch
    from torch.utils.data import DataLoader

    use_mixup  = aug_cfg.get("use_mixup", False)
    use_cutmix = aug_cfg.get("use_cutmix", False)

    base_kwargs = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = True,
    )

    if use_weighted_sampler:
        import numpy as np
        from torch.utils.data import WeightedRandomSampler
        
        # 1. Lấy ma trận nhãn nhị phân [N, 8] của tập train
        labels_matrix = dataset.df[LABELS].values.astype(float)
        
        # 2. Tính số lượng ảnh thực tế của từng lớp bệnh
        class_counts = labels_matrix.sum(axis=0)
        
        # 3. Tính trọng số cho từng lớp (tỷ lệ nghịch với số lượng)
        class_weights = 1.0 / np.maximum(class_counts, 1.0)
        
        # 4. Trọng số của mẫu bằng tổng trọng số các lớp nó mang nhãn dương
        sample_weights = np.dot(labels_matrix, class_weights)
        sample_weights = np.maximum(sample_weights, 1e-5)  # Tránh lỗi chia cho 0
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        base_kwargs["sampler"] = sampler
        base_kwargs["shuffle"] = False
        print(f"[Loader] Khởi chạy WeightedRandomSampler. Trọng số lớp: "
              f"{dict(zip(LABELS, np.round(class_weights, 4)))}")
    else:
        base_kwargs["shuffle"] = True

    if use_mixup and use_cutmix:
        # Xen kẽ ngẫu nhiên: 50% batch dùng MixUp, 50% CutMix
        # Dùng CutMixCollator với prob=0.5 trên CutMix side;
        # MixUp được tích hợp như một option thay thế ngẫu nhiên.
        # Cách đơn giản nhất: dùng CutMix (chứa stochastic behaviour)
        # và MixUp riêng theo xác suất từ config
        collator = _MixCutCollator(
            mixup_alpha  = aug_cfg.get("mixup_alpha", 0.4),
            mixup_prob   = aug_cfg.get("mixup_prob", 0.5),
            cutmix_alpha = aug_cfg.get("cutmix_alpha", 1.0),
            cutmix_prob  = aug_cfg.get("cutmix_prob", 0.5),
        )
        print(f"[Augmentation] MixUp(α={aug_cfg.get('mixup_alpha',0.4)}) + "
              f"CutMix(α={aug_cfg.get('cutmix_alpha',1.0)}) — xen kẽ ngẫu nhiên")
        return DataLoader(dataset, **base_kwargs, collate_fn=collator)

    elif use_mixup:
        collator = MixUpCollator(
            alpha = aug_cfg.get("mixup_alpha", 0.4),
            prob  = aug_cfg.get("mixup_prob", 0.5),
        )
        print(f"[Augmentation] MixUp only (α={aug_cfg.get('mixup_alpha',0.4)})")
        return DataLoader(dataset, **base_kwargs, collate_fn=collator)

    elif use_cutmix:
        collator = CutMixCollator(
            alpha = aug_cfg.get("cutmix_alpha", 1.0),
            prob  = aug_cfg.get("cutmix_prob", 0.5),
        )
        print(f"[Augmentation] CutMix only (α={aug_cfg.get('cutmix_alpha',1.0)})")
        return DataLoader(dataset, **base_kwargs, collate_fn=collator)

    else:
        print("[Augmentation] Không dùng MixUp/CutMix")
        return DataLoader(dataset, **base_kwargs)


class _MixCutCollator:
    """Collator kết hợp: mỗi batch ngẫu nhiên dùng MixUp hoặc CutMix."""

    def __init__(self, mixup_alpha, mixup_prob, cutmix_alpha, cutmix_prob):
        self.mixup  = MixUpCollator(alpha=mixup_alpha, prob=1.0)   # xác suất xử lý qua mixup
        self.cutmix = CutMixCollator(alpha=cutmix_alpha, prob=1.0)
        self.mixup_prob = mixup_prob
        import random
        self._rng = random

    def __call__(self, batch):
        # 50/50: chọn MixUp hoặc CutMix cho mỗi batch
        if self._rng.random() < self.mixup_prob:
            return self.mixup(batch)
        return self.cutmix(batch)


def load_config(config_path: str) -> dict:
    """Đọc config YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_results_dir(results_dir: str | Path) -> Path:
    """Tạo thư mục kết quả."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def save_checkpoint(state: dict, path: Path) -> None:
    """Lưu checkpoint."""
    torch.save(state, path)
    print(f"  → Lưu checkpoint: {path}")


def log_epoch(log_path: Path, epoch_data: dict) -> None:
    """Ghi một dòng log vào file CSV."""
    file_exists = log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=epoch_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(epoch_data)


def compute_auc_roc(
    all_probs: list[list[float]],
    all_targets: list[list[float]],
) -> float:
    """Tính macro AUC-ROC cho multi-label classification."""
    if not HAS_SKLEARN:
        return -1.0
    import numpy as np
    probs   = np.array(all_probs)
    targets = np.array(all_targets)
    try:
        return float(roc_auc_score(targets, probs, average="macro"))
    except Exception:
        return -1.0


def compute_mae_pearson(
    preds: list[float],
    targets: list[float],
    age_mean: float,
    age_std: float,
) -> dict[str, float]:
    """Tính MAE và Pearson Correlation cho tuổi (denormalized)."""
    import numpy as np
    preds_arr   = np.array(preds) * age_std + age_mean
    targets_arr = np.array(targets) * age_std + age_mean

    mae = float(np.mean(np.abs(preds_arr - targets_arr)))

    if len(preds_arr) > 1 and preds_arr.std() > 0:
        corr_matrix = np.corrcoef(preds_arr, targets_arr)
        pearson = float(corr_matrix[0, 1])
    else:
        pearson = 0.0

    return {"mae": mae, "pearson": pearson}


def build_optimizer_and_scheduler(model, cfg, is_frozen, total_epochs):
    """Hàm tạo lại Optimizer và Scheduler"""
    if is_frozen:
        lr = float(cfg['training'].get('frozen_lr', 1e-3))
        # Ở Stage 1, chỉ lấy các tham số yêu cầu gradient (tức là MLP heads)
        params = [p for p in model.parameters() if p.requires_grad]
        t_max = int(cfg['training'].get('freeze_epochs', 5))
    else:
        lr = float(cfg['training'].get('unfrozen_lr', 3e-5))
        # Ở Stage 2, mọi tham số đều cần gradient
        params = model.parameters()
        t_max = total_epochs - int(cfg['training'].get('freeze_epochs', 5))

    optimizer = torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=float(cfg['optimizer'].get('weight_decay', 0.05))
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=1e-6
    )
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Train / Validate một epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    mode: str,
    age_mean: float,
    age_std: float,
    threshold: float | list[float] | torch.Tensor = 0.5,
    scaler: torch.cuda.amp.GradScaler | None = None,
    dry_run: bool = False,
    binary_mode: bool = False,
    binocular: bool = False,
    accum_steps: int = 1,
) -> tuple[dict, list, list]:
    """Chạy một epoch huấn luyện hoặc đánh giá (train hoặc val/test).

    Args:
        model: Model đang train/eval
        loader: DataLoader song nhãn hoặc đơn nhãn
        criterion: MultiTaskLoss
        optimizer: Optimizer (None nếu đang ở chế độ đánh giá)
        device: CPU hoặc CUDA
        mode: "train" hoặc "val" hoặc "test"
        age_mean, age_std: Chỉ số tuổi trung bình/lệch chuẩn để giải chuẩn hóa khi tính MAE
        scaler: GradScaler cho mixed precision training
        dry_run: Chỉ chạy 1 batch để kiểm thử lỗi nhanh
        binary_mode: Chế độ phân loại nhị phân (Normal vs Pathological)
        binocular: Chế độ Siamese song mắt y sinh xử lý đồng thời hai mắt
        accum_steps: Số bước tích luỹ gradient trước khi cập nhật trọng số

    Returns:
        Dict chứa tất cả các chỉ số (metrics) của epoch.
    """
    is_train = (mode == "train")
    model.train() if is_train else model.eval()

    # Accumulators
    total_loss = total_cls = total_reg = 0.0
    all_probs    = []   # Xác suất sigmoid đầu ra
    all_targets  = []   # Nhãn thực tế (ground truth)
    all_age_pred = []   # Tuổi dự đoán đã chuẩn hóa
    all_age_true = []   # Tuổi thực tế đã chuẩn hóa

    it = enumerate(loader)
    if HAS_TQDM:
        it = enumerate(tqdm(loader, desc=f"  {mode}", leave=False, ncols=90))

    ctx = torch.no_grad() if not is_train else torch.enable_grad()

    # Khởi tạo optimizer sạch cho tích luỹ gradient
    if is_train and optimizer is not None:
        optimizer.zero_grad()

    with ctx:
        for batch_idx, batch in it:
            if binocular:
                # Nạp dữ liệu cặp mắt ở chế độ song nhãn
                left_image = batch["left_image"].to(device)
                right_image = batch["right_image"].to(device)
                left_missing = batch["left_missing"].to(device)
                right_missing = batch["right_missing"].to(device)
                labels_cls = batch["label"].to(device)
                age_true = batch["age"].to(device)
                bs = left_image.size(0)
            else:
                # Nạp dữ liệu ảnh đơn gốc
                images   = batch["image"].to(device)
                labels   = batch["labels"].to(device)
                age_true = batch["age"].to(device)
                bs = images.size(0)
                if binary_mode:
                    # Chuyển đổi nhãn đơn từ đa nhãn sang nhị phân
                    labels_cls = (labels[:, 0:1] == 0).float()
                else:
                    labels_cls = labels

            # Tự động kích hoạt Mixed Precision (autocast)
            use_amp = (device.type == "cuda")
            with torch.cuda.amp.autocast(enabled=use_amp):
                # Lan truyền tiến
                if binocular:
                    output = model(
                        left_image=left_image,
                        right_image=right_image,
                        left_missing=left_missing,
                        right_missing=right_missing
                    )
                else:
                    output = model(images)
                
                logits   = output["logits"]
                age_pred = output["age_pred"]

                # Tính toán joint loss
                loss, detail = criterion(logits, labels_cls, age_pred, age_true)
                
                # Chia tỉ lệ loss theo số bước tích luỹ gradient
                if is_train and optimizer is not None:
                    loss = loss / accum_steps

            # Lan truyền ngược
            if is_train and optimizer is not None:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    # Cập nhật trọng số khi gom đủ số batch tích luỹ
                    if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()

            # Tích luỹ giá trị loss và predictions
            total_loss += detail["loss_total"] * accum_steps * bs if is_train else detail["loss_total"] * bs
            total_cls  += detail["loss_cls"]   * accum_steps * bs if is_train else detail["loss_cls"]   * bs
            total_reg  += detail["loss_reg"]   * accum_steps * bs if is_train else detail["loss_reg"]   * bs

            probs = torch.sigmoid(logits).detach().cpu().tolist()
            all_probs.extend(probs)
            all_targets.extend(labels_cls.detach().cpu().tolist())
            all_age_pred.extend(age_pred.squeeze(1).detach().cpu().tolist())
            all_age_true.extend(age_true.squeeze(1).detach().cpu().tolist())

            if dry_run:
                break

    n = len(all_probs) if len(all_probs) > 0 else 1
    metrics = {
        f"{mode}_loss":     total_loss / n,
        f"{mode}_loss_cls": total_cls  / n,
        f"{mode}_loss_reg": total_reg  / n,
    }

    # Chuyển đổi dữ liệu sang tensor để tính toán metrics thống kê
    import torch as _t
    probs_t   = _t.FloatTensor(all_probs)
    targets_t = _t.FloatTensor(all_targets)
    
    if binary_mode:
        # Lấy giá trị ngưỡng phân loại nhị phân
        thresh_val = threshold[0] if isinstance(threshold, list) else threshold
        if isinstance(thresh_val, torch.Tensor):
            thresh_val = thresh_val.item()
            
        # Tính toán bộ metrics nhị phân y sinh
        from src.utils import compute_binary_metrics
        bin_metrics = compute_binary_metrics(probs_t, targets_t, threshold=thresh_val)
        
        metrics[f"{mode}_accuracy"] = bin_metrics["accuracy"]
        metrics[f"{mode}_precision_macro"] = bin_metrics["precision"]
        metrics[f"{mode}_recall_macro"] = bin_metrics["sensitivity"]  # Sensitivity (Độ nhạy)
        metrics[f"{mode}_f1_macro"] = bin_metrics["f1"]
        metrics[f"{mode}_specificity"] = bin_metrics["specificity"]  # Specificity (Độ đặc hiệu)
        metrics[f"{mode}_sensitivity"] = bin_metrics["sensitivity"]
        metrics[f"{mode}_auc_roc"] = bin_metrics["auc"]
    else:
        # Đánh giá đa nhãn Phase 2 gốc
        ml_metrics = compute_multilabel_metrics(probs_t, targets_t, threshold=threshold)
        for k, v in ml_metrics.items():
            metrics[f"{mode}_{k}"] = v

    # Đánh giá AUC-ROC cho đa nhãn
    if not binary_mode:
        metrics[f"{mode}_auc_roc"] = compute_auc_roc(all_probs, all_targets)

    # Đánh giá sai số tuổi võng mạc
    age_m = compute_mae_pearson(all_age_pred, all_age_true, age_mean, age_std)
    metrics[f"{mode}_age_mae"]     = age_m["mae"]
    metrics[f"{mode}_age_pearson"] = age_m["pearson"]

    return metrics, all_probs, all_targets




# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config_path: str, dry_run: bool = False, resume: str | None = None) -> None:
    """Main training function.

    Args:
        config_path: Đường dẫn đến file YAML config.
        dry_run: Nếu True, chỉ chạy 1 batch để kiểm tra pipeline.
        resume: Đường dẫn checkpoint để tiếp tục training.
    """
    # --- Load config ---
    cfg = load_config(config_path)
    exp_name = cfg["experiment_name"]
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {exp_name}")
    print(f"  {cfg.get('description', '')}")
    print(f"{'='*60}\n")

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # --- Paths ---
    project_root = Path(config_path).parent.parent
    
    # Auto-detect if we are running in Kaggle environment
    is_kaggle = os.path.exists("/kaggle/working")
    
    # Resolve splits_dir
    splits_dir = project_root / cfg["splits_dir"]
    if is_kaggle or not splits_dir.exists():
        for candidate in [
            project_root / cfg["splits_dir"].replace("archive/", ""),
            Path("/kaggle/working/code") / cfg["splits_dir"].replace("archive/", ""),
            Path("/kaggle/working/code/splits_clean"),
            Path("/kaggle/working/splits_clean"),
        ]:
            if candidate.exists():
                splits_dir = candidate
                break

    # Resolve img_dir
    img_dir = project_root / cfg["img_dir"]
    if is_kaggle or not img_dir.exists():
        if "Training Images" in str(cfg["img_dir"]):
            for root, dirs, files in os.walk('/kaggle/input'):
                if 'Training Images' in dirs:
                    candidate = Path(root) / 'Training Images'
                    if candidate.exists():
                        img_dir = candidate
                        break
        else:
            for candidate in [
                project_root / cfg["img_dir"].replace("archive/", ""),
                Path("/kaggle/working") / Path(cfg["img_dir"]).name,
                Path("/kaggle/working/preprocessed_images"),
                Path("/kaggle/working/enhanced_images"),
            ]:
                if candidate.exists():
                    img_dir = candidate
                    break

    # Resolve results_dir
    if is_kaggle:
        rel_results = cfg["output"]["results_dir"].replace("results/", "")
        results_dir = setup_results_dir(Path("/kaggle/working/results") / rel_results)
    else:
        results_dir = setup_results_dir(project_root / cfg["output"]["results_dir"])

    print(f"[Data] Splits: {splits_dir}")
    print(f"[Data] Images: {img_dir}")
    print(f"[Output] Results: {results_dir}\n")

    # --- Config binary_mode & binocular ---
    binary_mode = cfg.get("binary_mode", False)
    binocular   = cfg.get("binocular", False)

    # --- Metadata & pos_weight ---
    metadata_path = splits_dir / "metadata.json"
    metadata      = load_metadata(metadata_path)
    
    if binary_mode:
        import pandas as pd
        train_csv = splits_dir / "train.csv"
        df_train = pd.read_csv(train_csv)
        n_normal = int(df_train['N'].sum())
        n_pathological = len(df_train) - n_normal
        pos_weight = torch.FloatTensor([n_normal / max(n_pathological, 1)])
        print(f"[Binary Mode] n_normal={n_normal}, n_pathological={n_pathological}, pos_weight={pos_weight.item():.2f}")
    else:
        pos_weight    = get_pos_weight_from_metadata(metadata)
        
    age_mean      = metadata["age_stats"]["mean"]
    age_std       = metadata["age_stats"]["std"]
    print(f"[Metadata] age_mean={age_mean:.2f}, age_std={age_std:.2f}")

    # --- DataLoaders ---
    tr_cfg  = cfg["training"]
    aug_cfg = cfg.get("augmentation", {})
    img_size    = tr_cfg["img_size"]
    batch_size  = tr_cfg["batch_size"]
    num_workers = tr_cfg.get("num_workers", 0)
    pin_memory  = (device.type == "cuda")

    if binocular:
        # Cấu hình nạp dữ liệu ở chế độ song nhãn (cặp 2 mắt)
        dataloaders = get_binocular_dataloaders(
            splits_dir  = splits_dir,
            img_dir     = img_dir,
            img_size    = img_size,
            batch_size  = batch_size,
            num_workers = num_workers,
            pin_memory  = pin_memory,
        )

        collate_fn = None
        if aug_cfg.get("use_mixup", False) or aug_cfg.get("use_cutmix", False):
            collate_fn = BinocularAugmentCollator(
                use_mixup   = aug_cfg.get("use_mixup", False),
                use_cutmix  = aug_cfg.get("use_cutmix", False),
                mixup_alpha = aug_cfg.get("mixup_alpha", 0.4),
                mixup_prob  = aug_cfg.get("mixup_prob", 0.5),
                cutmix_alpha= aug_cfg.get("cutmix_alpha", 1.0),
                cutmix_prob = aug_cfg.get("cutmix_prob", 0.5),
            )
            print(f"[Loader] Gom lô song nhãn tăng cường (Binocular MixUp/CutMix): "
                  f"mixup={aug_cfg.get('use_mixup')}, cutmix={aug_cfg.get('use_cutmix')}")

        # Nạp lại loader tập train với tăng cường dữ liệu đặc thù song nhãn
        train_dataset = BinocularDataset(
            csv_path   = splits_dir / "train.csv",
            img_dir    = img_dir,
            transforms = get_transforms(mode="train", img_size=img_size),
            img_size   = img_size,
            age_mean   = dataloaders["train"].dataset.age_mean,
            age_std    = dataloaders["train"].dataset.age_std,
        )

        dataloaders["train"] = torch.utils.data.DataLoader(
            train_dataset,
            batch_size  = batch_size,
            shuffle     = True,
            num_workers = num_workers,
            pin_memory  = pin_memory,
            drop_last   = True,
            collate_fn  = collate_fn,
        )
    else:
        # Chế độ đơn nhãn gốc
        dataloaders = get_dataloaders(
            splits_dir   = splits_dir,
            img_dir      = img_dir,
            img_size     = img_size,
            batch_size   = batch_size,
            num_workers  = num_workers,
            pin_memory   = pin_memory,
        )

        train_dataset = ODIRDataset(
            csv_path   = splits_dir / "train.csv",
            img_dir    = img_dir,
            transforms = get_transforms(mode="train", img_size=img_size),
            age_mean   = dataloaders["train"].dataset.age_mean,
            age_std    = dataloaders["train"].dataset.age_std,
        )
        dataloaders["train"] = build_train_loader(
            dataset              = train_dataset,
            aug_cfg              = aug_cfg,
            batch_size           = batch_size,
            num_workers          = num_workers,
            pin_memory           = pin_memory,
            use_weighted_sampler = tr_cfg.get("use_weighted_sampler", False),
        )

    print(f"[Loader] Train={len(train_dataset)}, "
          f"Val={len(dataloaders['val'].dataset)}, "
          f"Test={len(dataloaders['test'].dataset)}\n")

    # --- Model ---
    m_cfg      = cfg["model"]
    model_type = cfg.get("model_type", "cnn")
    num_labels = 1 if binary_mode else 8
    model = build_model(
        model_type      = model_type,
        pretrained      = m_cfg.get("pretrained", True),
        freeze_backbone = m_cfg.get("freeze_backbone", False),
        dropout_cls     = m_cfg.get("dropout_cls", 0.3),
        dropout_reg     = m_cfg.get("dropout_reg", 0.2),
        img_size        = tr_cfg.get("img_size", 224),
        variant         = m_cfg.get("variant", "tiny"),
        pretrained_path = m_cfg.get("pretrained_path", None),
        num_labels      = num_labels,
        binocular       = binocular,  # Chuyển cờ song nhãn
    ).to(device)
    print(f"[Model] {model}\n")

    # --- Loss ---
    l_cfg   = cfg["loss"]
    focal_alpha_val = l_cfg.get("focal_alpha", 0.25)
    if str(focal_alpha_val).lower().strip() == "auto":
        # Tự động tính toán focal_alpha dựa trên tỷ lệ mẫu Bình thường (Normal) trong tập Train
        if binocular:
            labels_list = [p["label"] for p in train_dataset.patients]
        else:
            import pandas as pd
            train_csv = splits_dir / "train.csv"
            df_train = pd.read_csv(train_csv)
            labels_list = (df_train['N'] == 0).astype(int).tolist()
        n_pos = sum(labels_list)  # Số lượng mẫu Bệnh lý (Pathological, y=1)
        n_neg = len(labels_list) - n_pos  # Số lượng mẫu Bình thường (Normal, y=0)
        focal_alpha_val = n_neg / len(labels_list)
        print(f"[Focal Loss] Tỷ lệ Train: {n_neg} Normal ({n_neg/len(labels_list)*100:.1f}%) "
              f"/ {n_pos} Pathological ({n_pos/len(labels_list)*100:.1f}%)")
        print(f"[Focal Loss] Tự động cấu hình alpha = {focal_alpha_val:.4f} (1-alpha = {1.0-focal_alpha_val:.4f})")
    else:
        focal_alpha_val = float(focal_alpha_val)
        print(f"[Focal Loss] Sử dụng alpha cấu hình cứng: {focal_alpha_val}")

    criterion = MultiTaskLoss(
        pos_weight  = pos_weight,
        lam         = l_cfg.get("lam", 0.1),
        loss_type   = l_cfg.get("loss_type", "bce"),
        gamma_neg   = l_cfg.get("gamma_neg", 4.0),
        gamma_pos   = l_cfg.get("gamma_pos", 1.0),
        clip        = l_cfg.get("clip", 0.05),
        focal_alpha = focal_alpha_val,
        focal_gamma = l_cfg.get("focal_gamma", 2.0),
        device      = device,
    )
    print(f"[Loss]  {criterion}\n")

    # --- Optimizer & Scheduler ---
    epochs = tr_cfg["epochs"]
    freeze_epochs = int(cfg['training'].get('freeze_epochs', 5))
    accum_steps = tr_cfg.get("gradient_accumulation_steps", 1)
    print(f"[Accumulation] Sử dụng gradient_accumulation_steps = {accum_steps}")

    if cfg['training'].get('two_stage', False):
        model.freeze_backbone()
        optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, is_frozen=True, total_epochs=epochs)
    else:
        o_cfg = cfg["optimizer"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr           = o_cfg.get("lr", 3e-4),
            weight_decay = o_cfg.get("weight_decay", 0.01),
        )
        s_cfg     = cfg["scheduler"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = s_cfg.get("T_max", epochs),
            eta_min = s_cfg.get("eta_min", 1e-6),
        )

    # --- Thiết lập Early Stopping & Resume ---
    patience  = tr_cfg.get("early_stopping_patience", 5)
    es_metric_name = tr_cfg.get("early_stopping_metric", "f1")
    if es_metric_name == "auc":
        es_key = "val_auc_roc"
        es_display_name = "AUC-ROC"
    else:
        es_key = "val_f1_macro"
        es_display_name = "F1-Score"

    start_epoch    = 1
    best_es_val    = 0.0
    best_val_f1    = 0.0
    early_stop_cnt = 0
    print(f"[Early Stopping] Giám sát chỉ số: {es_display_name} ({es_key}) với patience={patience}")

    if resume:
        print(f"[Resume] Loading checkpoint: {resume}")
        ckpt       = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch    = ckpt["epoch"] + 1
        
        if cfg['training'].get('two_stage', False):
            ckpt_epoch = ckpt["epoch"]
            ckpt_is_frozen = ckpt_epoch <= freeze_epochs
            start_is_frozen = start_epoch <= freeze_epochs
            
            if start_is_frozen:
                model.freeze_backbone()
                optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, is_frozen=True, total_epochs=epochs)
            else:
                model.unfreeze_backbone()
                optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, is_frozen=False, total_epochs=epochs)
            
            if ckpt_is_frozen == start_is_frozen:
                optimizer.load_state_dict(ckpt["optimizer_state"])
                scheduler.load_state_dict(ckpt["scheduler_state"])
                print("  → Nạp thành công trạng thái optimizer và scheduler từ checkpoint.")
            else:
                print(f"  → Phát hiện chuyển giao Giai đoạn ({ckpt_epoch} -> {start_epoch}): Reset Optimizer và Scheduler sạch cho Stage 2.")
        else:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            
        best_val_f1    = ckpt.get("best_val_f1", 0.0)
        best_es_val    = ckpt.get("best_es_val", ckpt.get("val_metrics", {}).get(es_key, best_val_f1))
        early_stop_cnt = ckpt.get("early_stop_cnt", 0)
        print(f"  → Tiếp tục từ epoch {start_epoch}, best_{es_key}={best_es_val:.4f}\n")

    # --- Lưu config vào results_dir ---
    import shutil
    shutil.copy(config_path, results_dir / "config.yaml")

    # --- Log file ---
    log_path = results_dir / "training_log.csv"

    # --- Training Loop ---
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    n_epochs  = 1 if dry_run else tr_cfg["epochs"]
    patience  = tr_cfg.get("early_stopping_patience", 5)
    best_path = results_dir / "best_model.pth"
    last_path = results_dir / "last_model.pth"

    print(f"{'─'*60}")
    print(f"  Bắt đầu training: {n_epochs} epoch(s)")
    if dry_run:
        print("  [DRY-RUN MODE] — chỉ 1 epoch để kiểm tra pipeline")
    print(f"{'─'*60}\n")

    for epoch in range(start_epoch, n_epochs + 1):
        epoch_start = time.time()

        if cfg['training'].get('two_stage', False):
            if epoch <= freeze_epochs:
                if epoch == start_epoch and not resume:
                    print(f"[STAGE 1] Đóng băng Backbone. Train MLP Heads với LR={cfg['training'].get('frozen_lr', 1e-3)}")
                    model.freeze_backbone()
                    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, is_frozen=True, total_epochs=epochs)
            elif epoch == freeze_epochs + 1:
                if epoch != start_epoch:
                    print(f"\n[STAGE 2] 🔓 Mở khóa toàn bộ mạng. Rebuild Optimizer với LR={cfg['training'].get('unfrozen_lr', 3e-5)}")
                    model.unfreeze_backbone()
                    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, is_frozen=False, total_epochs=epochs)

        current_lr  = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch:02d}/{n_epochs}]  lr={current_lr:.2e}")

        # Huấn luyện
        train_m, _, _ = run_epoch(
            model, dataloaders["train"], criterion, optimizer,
            device, "train", age_mean, age_std, scaler=scaler,
            dry_run=dry_run, binary_mode=binary_mode, binocular=binocular,
            accum_steps=accum_steps
        )

        # Đánh giá validation (sử dụng ngưỡng mặc định 0.5 để thu về xác suất thô)
        val_m, val_probs, val_targets = run_epoch(
            model, dataloaders["val"], criterion, None,
            device, "val", age_mean, age_std,
            threshold=0.5,
            dry_run=dry_run, binary_mode=binary_mode, binocular=binocular
        )

        # Căn chỉnh Youden Threshold động ngay lập tức đối với phân loại nhị phân y sinh
        if binary_mode:
            import torch as _t
            from src.utils import find_best_binary_threshold, compute_binary_metrics
            val_probs_t = _t.FloatTensor(val_probs)
            val_targets_t = _t.FloatTensor(val_targets)
            best_val_thresh = find_best_binary_threshold(val_probs_t, val_targets_t)
            
            # Tính toán lại bộ chỉ số nhị phân tối ưu và ghi đè vào val_m
            opt_metrics = compute_binary_metrics(val_probs_t, val_targets_t, threshold=best_val_thresh)
            val_m["val_accuracy"] = opt_metrics["accuracy"]
            val_m["val_precision_macro"] = opt_metrics["precision"]
            val_m["val_recall_macro"] = opt_metrics["sensitivity"]
            val_m["val_f1_macro"] = opt_metrics["f1"]
            val_m["val_specificity"] = opt_metrics["specificity"]
            val_m["val_sensitivity"] = opt_metrics["sensitivity"]
            val_m["val_auc_roc"] = opt_metrics["auc"]
            
            print(f"  [Calibration] Ngưỡng tối ưu={best_val_thresh:.4f} | Val F1={opt_metrics['f1']:.4f} | Val AUC={opt_metrics['auc']:.4f}")

        scheduler.step()

        elapsed = time.time() - epoch_start
        if binary_mode:
            print(
                f"  Train loss={train_m['train_loss']:.4f} "
                f"| Val loss={val_m['val_loss']:.4f} "
                f"| Val F1={val_m['val_f1_macro']:.4f} "
                f"| Val AUC={val_m['val_auc_roc']:.4f} "
                f"| Val Sens={val_m['val_sensitivity']:.4f} "
                f"| Val Spec={val_m['val_specificity']:.4f} "
                f"| Val MAE={val_m['val_age_mae']:.2f} yrs"
                f"  [{elapsed:.1f}s]"
            )
        else:
            print(
                f"  Train loss={train_m['train_loss']:.4f} "
                f"| Val loss={val_m['val_loss']:.4f} "
                f"| Val F1={val_m['val_f1_macro']:.4f} "
                f"| Val AUC={val_m['val_auc_roc']:.4f} "
                f"| Val MAE={val_m['val_age_mae']:.2f} yrs"
                f"  [{elapsed:.1f}s]"
            )

        # Log CSV
        epoch_log = {"epoch": epoch, "lr": current_lr, **train_m, **val_m}
        log_epoch(log_path, epoch_log)

        # Giám sát chỉ số tối ưu Early Stopping
        es_val = val_m[es_key]
        if es_val > best_es_val:
            best_es_val = es_val
            best_val_f1 = val_m.get("val_f1_macro", 0.0)
            early_stop_cnt = 0
            if cfg["output"].get("save_best_model", True) and not dry_run:
                save_checkpoint({
                    "epoch":           epoch,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_es_val":     best_es_val,
                    "best_val_f1":     best_val_f1,
                    "val_metrics":     val_m,
                    "config":          cfg,
                }, best_path)
        else:
            early_stop_cnt += 1
            print(f"  [EarlyStopping] {early_stop_cnt}/{patience} — best {es_display_name}={best_es_val:.4f}")
            if early_stop_cnt >= patience and not dry_run:
                print(f"\n  *** EARLY STOPPING tại epoch {epoch} ***\n")
                break

        if not dry_run:
            save_checkpoint({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_es_val":     best_es_val,
                "best_val_f1":     best_val_f1,
                "early_stop_cnt":  early_stop_cnt,
                "config":          cfg,
            }, last_path)

    # --- Đánh giá trên Test set ---
    if not dry_run and best_path.exists():
        print(f"\n{'─'*60}")
        print(f"  Đánh giá TEST SET với best model (val F1={best_val_f1:.4f})")
        print(f"{'─'*60}")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

        print("[Dynamic Thresholding] Đang quét tìm ngưỡng tối ưu trên tập Validation...")
        val_m_tmp, val_probs, val_targets = run_epoch(
            model, dataloaders["val"], criterion, None,
            device, "val", age_mean, age_std, threshold=0.5,
            binary_mode=binary_mode, binocular=binocular
        )
        
        if binary_mode:
            from src.utils import find_best_binary_threshold
            best_thresholds = [find_best_binary_threshold(val_probs, val_targets)]
            print(f"  → Ngưỡng tối ưu tìm được cho phân loại nhị phân (Youden's Index): {best_thresholds[0]:.2f}\n")
        else:
            best_thresholds = find_best_thresholds(val_probs, val_targets)
            print("  → Ngưỡng tối ưu tìm được cho 8 bệnh lý:")
            from src.utils import get_label_names
            lbl_names = get_label_names()
            for idx, lbl in enumerate(LABELS):
                print(f"    - {lbl} ({lbl_names[lbl]}): {best_thresholds[idx]:.2f}")
            print()

        # Đánh giá Test ngưỡng mặc định 0.5 bằng 1 forward pass duy nhất
        test_m_default, test_probs, test_targets = run_epoch(
            model, dataloaders["test"], criterion, None,
            device, "test", age_mean, age_std, threshold=0.5,
            binary_mode=binary_mode, binocular=binocular
        )
        if binary_mode:
            print(
                f"  [NGƯỠNG MẶC ĐỊNH 0.5] "
                f"TEST F1={test_m_default['test_f1_macro']:.4f} "
                f"| Accuracy={test_m_default['test_accuracy']:.4f} "
                f"| AUC-ROC={test_m_default['test_auc_roc']:.4f} "
                f"| Sens={test_m_default['test_sensitivity']:.4f} "
                f"| Spec={test_m_default['test_specificity']:.4f} "
                f"| Age MAE={test_m_default['test_age_mae']:.2f} yrs"
            )
        else:
            print(
                f"  [NGƯỠNG MẶC ĐỊNH 0.5] "
                f"TEST F1-macro={test_m_default['test_f1_macro']:.4f} "
                f"| AUC-ROC={test_m_default['test_auc_roc']:.4f} "
                f"| Age MAE={test_m_default['test_age_mae']:.2f} yrs"
            )

        # Tính toán bộ metrics tối ưu (Youden) trực tiếp từ dự đoán đã thu được ở bước trên để tránh double forward pass
        if binary_mode:
            import torch as _t
            from src.utils import compute_binary_metrics
            test_probs_t = _t.FloatTensor(test_probs)
            test_targets_t = _t.FloatTensor(test_targets)
            opt_metrics = compute_binary_metrics(test_probs_t, test_targets_t, threshold=best_thresholds[0])
            
            test_m_opt = test_m_default.copy()
            test_m_opt["test_accuracy"] = opt_metrics["accuracy"]
            test_m_opt["test_precision_macro"] = opt_metrics["precision"]
            test_m_opt["test_recall_macro"] = opt_metrics["sensitivity"]
            test_m_opt["test_f1_macro"] = opt_metrics["f1"]
            test_m_opt["test_specificity"] = opt_metrics["specificity"]
            test_m_opt["test_sensitivity"] = opt_metrics["sensitivity"]
            test_m_opt["test_auc_roc"] = opt_metrics["auc"]
            
            print(
                f"  [NGƯỠNG TỐI ƯU ĐỘNG]  "
                f"TEST F1={test_m_opt['test_f1_macro']:.4f} "
                f"| Accuracy={test_m_opt['test_accuracy']:.4f} "
                f"| AUC-ROC={test_m_opt['test_auc_roc']:.4f} "
                f"| Sens={test_m_opt['test_sensitivity']:.4f} "
                f"| Spec={test_m_opt['test_specificity']:.4f} "
                f"| Age MAE={test_m_opt['test_age_mae']:.2f} yrs"
            )
        else:
            import torch as _t
            from src.utils import compute_multilabel_metrics
            test_probs_t = _t.FloatTensor(test_probs)
            test_targets_t = _t.FloatTensor(test_targets)
            opt_metrics = compute_multilabel_metrics(test_probs_t, test_targets_t, threshold=best_thresholds)
            
            test_m_opt = test_m_default.copy()
            for k, v in opt_metrics.items():
                test_m_opt[f"test_{k}"] = v
                
            print(
                f"  [NGƯỠNG TỐI ƯU ĐỘNG]  "
                f"TEST F1-macro={test_m_opt['test_f1_macro']:.4f} "
                f"| AUC-ROC={test_m_opt['test_auc_roc']:.4f} "
                f"| Age MAE={test_m_opt['test_age_mae']:.2f} yrs"
            )

        # Lưu kết quả
        test_results_path = results_dir / "test_results.json"
        with open(test_results_path, "w") as f:
            json.dump({
                "experiment": exp_name,
                "best_val_f1_default": best_val_f1,
                "optimal_thresholds": best_thresholds,
                "metrics_default_0.5": test_m_default,
                "metrics_optimal_dynamic": test_m_opt,
            }, f, indent=2)
        print(f"\n  → Kết quả test lưu tại: {test_results_path}")

    print(f"\n{'='*60}")
    print(f"  HOÀN THÀNH: {exp_name}")
    print(f"  Best Val F1-macro: {best_val_f1:.4f}")
    print(f"  Results dir: {results_dir}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ODIR-5K Multi-task Learning model"
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Đường dẫn đến file YAML config (vd: configs/exp_raw.yaml)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ chạy 1 epoch để kiểm tra pipeline, không lưu model"
    )
    parser.add_argument(
        "--resume", "-r",
        default=None,
        help="Đường dẫn checkpoint để tiếp tục training (vd: results/exp_raw/last_model.pth)"
    )
    args = parser.parse_args()

    train(
        config_path = args.config,
        dry_run     = args.dry_run,
        resume      = args.resume,
    )
