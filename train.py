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
) -> dict:
    """Chạy một epoch (train hoặc val/test).

    Args:
        model: Model đang train/eval
        loader: DataLoader
        criterion: MultiTaskLoss
        optimizer: Optimizer (None nếu eval mode)
        device: CPU hoặc CUDA
        mode: "train" hoặc "val" hoặc "test"
        age_mean, age_std: Để denormalize tuổi khi tính MAE
        scaler: GradScaler cho mixed precision training
        dry_run: Chỉ chạy 1 batch để kiểm thử
        binary_mode: Chế độ phân loại nhị phân

    Returns:
        Dict chứa tất cả metrics của epoch.
    """
    is_train = (mode == "train")
    model.train() if is_train else model.eval()

    # Accumulators
    total_loss = total_cls = total_reg = 0.0
    all_probs    = []   # sigmoid probabilities
    all_targets  = []   # ground truth labels
    all_age_pred = []   # predicted age (normalized)
    all_age_true = []   # true age (normalized)

    it = enumerate(loader)
    if HAS_TQDM:
        it = enumerate(tqdm(loader, desc=f"  {mode}", leave=False, ncols=90))

    ctx = torch.no_grad() if not is_train else torch.enable_grad()

    with ctx:
        for batch_idx, batch in it:
            images   = batch["image"].to(device)
            labels   = batch["labels"].to(device)
            age_true = batch["age"].to(device)

            if binary_mode:
                # Cột 0 là Normal (1 nếu normal, 0 nếu có bệnh). 
                # Đổi thành: 1 nếu có bệnh (pathological), 0 nếu bình thường
                labels_cls = (labels[:, 0:1] == 0).float()
            else:
                labels_cls = labels

            # Auto mixed precision context (autocast)
            use_amp = (device.type == "cuda")
            with torch.cuda.amp.autocast(enabled=use_amp):
                # Forward
                output   = model(images)
                logits   = output["logits"]
                age_pred = output["age_pred"]

                # Loss
                loss, detail = criterion(logits, labels_cls, age_pred, age_true)

            if is_train and optimizer is not None:
                optimizer.zero_grad()
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            # Accumulate
            bs = images.size(0)
            total_loss += detail["loss_total"] * bs
            total_cls  += detail["loss_cls"]   * bs
            total_reg  += detail["loss_reg"]   * bs

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

    # F1-macro (threshhold 0.5)
    import torch as _t
    probs_t   = _t.FloatTensor(all_probs)
    targets_t = _t.FloatTensor(all_targets)
    
    if binary_mode:
        thresh_val = threshold[0] if isinstance(threshold, list) else threshold
        if isinstance(thresh_val, torch.Tensor):
            thresh_val = thresh_val.item()
        pred_binary = (probs_t >= thresh_val).float()
        tp = (pred_binary * targets_t).sum().item()
        fp = (pred_binary * (1 - targets_t)).sum().item()
        fn = ((1 - pred_binary) * targets_t).sum().item()
        tn = ((1 - pred_binary) * (1 - targets_t)).sum().item()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (pred_binary == targets_t).float().mean().item()
        
        metrics[f"{mode}_accuracy"] = accuracy
        metrics[f"{mode}_precision_macro"] = precision
        metrics[f"{mode}_recall_macro"] = recall
        metrics[f"{mode}_f1_macro"] = f1
    else:
        ml_metrics = compute_multilabel_metrics(probs_t, targets_t, threshold=threshold)
        for k, v in ml_metrics.items():
            metrics[f"{mode}_{k}"] = v

    # AUC-ROC
    if binary_mode:
        if not HAS_SKLEARN:
            metrics[f"{mode}_auc_roc"] = -1.0
        else:
            import numpy as np
            try:
                metrics[f"{mode}_auc_roc"] = float(roc_auc_score(np.array(all_targets), np.array(all_probs)))
            except:
                metrics[f"{mode}_auc_roc"] = -1.0
    else:
        metrics[f"{mode}_auc_roc"] = compute_auc_roc(all_probs, all_targets)

    # Age metrics
    age_m = compute_mae_pearson(all_age_pred, all_age_true, age_mean, age_std)
    metrics[f"{mode}_age_mae"]     = age_m["mae"]
    metrics[f"{mode}_age_pearson"] = age_m["pearson"]

    return metrics


def get_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    dry_run: bool = False,
    binary_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Thu thập toàn bộ sigmoid probabilities và targets từ DataLoader.

    Dùng riêng cho việc quét tìm ngưỡng tối ưu động trên tập Validation.
    """
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)
            if binary_mode:
                labels_cls = (labels[:, 0:1] == 0).float()
            else:
                labels_cls = labels
            output = model(images)
            logits = output["logits"]
            probs = torch.sigmoid(logits).cpu().tolist()
            all_probs.extend(probs)
            all_targets.extend(labels_cls.cpu().tolist())
            if dry_run:
                break

    return torch.FloatTensor(all_probs), torch.FloatTensor(all_targets)


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
    splits_dir   = project_root / cfg["splits_dir"]
    img_dir      = project_root / cfg["img_dir"]
    results_dir  = setup_results_dir(project_root / cfg["output"]["results_dir"])

    print(f"[Data] Splits: {splits_dir}")
    print(f"[Data] Images: {img_dir}")
    print(f"[Output] Results: {results_dir}\n")

    # --- Config binary_mode ---
    binary_mode = cfg.get("binary_mode", False)

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

    # Val/Test loaders — chuẩn, không augmentation
    dataloaders = get_dataloaders(
        splits_dir   = splits_dir,
        img_dir      = img_dir,
        img_size     = img_size,
        batch_size   = batch_size,
        num_workers  = num_workers,
        pin_memory   = pin_memory,
    )

    # Train loader — thay bằng MixUp/CutMix nếu được bật trong config
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
    model_type = cfg.get("model_type", "cnn")   # 'cnn' hoặc 'swin'
    num_labels = 1 if binary_mode else 8
    model = build_model(
        model_type      = model_type,
        pretrained      = m_cfg.get("pretrained", True),
        freeze_backbone = m_cfg.get("freeze_backbone", False),
        dropout_cls     = m_cfg.get("dropout_cls", 0.3),
        dropout_reg     = m_cfg.get("dropout_reg", 0.2),
        img_size        = tr_cfg.get("img_size", 224),     # Swin/ViT cần biết img_size
        variant         = m_cfg.get("variant", "tiny"),    # 'tiny'|'small'|'base'
        pretrained_path = m_cfg.get("pretrained_path", None), # Truyền đường dẫn trọng số pre-trained nếu có
        num_labels      = num_labels,
    ).to(device)
    print(f"[Model] {model}\n")

    # --- Loss ---
    l_cfg   = cfg["loss"]
    criterion = MultiTaskLoss(
        pos_weight = pos_weight,
        lam        = l_cfg.get("lam", 0.1),
        loss_type  = l_cfg.get("loss_type", "bce"),
        gamma_neg  = l_cfg.get("gamma_neg", 4.0),
        gamma_pos  = l_cfg.get("gamma_pos", 1.0),
        clip       = l_cfg.get("clip", 0.05),
        device     = device,
    )
    print(f"[Loss]  {criterion}\n")

    # --- Optimizer & Scheduler ---
    epochs = tr_cfg["epochs"]
    freeze_epochs = int(cfg['training'].get('freeze_epochs', 5))

    if cfg['training'].get('two_stage', False):
        # Khởi tạo tạm thời cho Stage 1 (Frozen) trước khi vào loop hoặc resume
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

    # --- Resume ---
    start_epoch    = 1
    best_val_f1    = 0.0
    early_stop_cnt = 0

    if resume:
        print(f"[Resume] Loading checkpoint: {resume}")
        ckpt       = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        
        start_epoch    = ckpt["epoch"] + 1
        
        if cfg['training'].get('two_stage', False):
            ckpt_epoch = ckpt["epoch"]
            ckpt_is_frozen = ckpt_epoch <= freeze_epochs
            start_is_frozen = start_epoch <= freeze_epochs
            
            # Khởi tạo đúng Stage cho start_epoch trước khi nạp state
            if start_is_frozen:
                model.freeze_backbone()
                optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, is_frozen=True, total_epochs=epochs)
            else:
                model.unfreeze_backbone()
                optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, is_frozen=False, total_epochs=epochs)
            
            # Chỉ nạp optimizer/scheduler state nếu cùng stage
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
        early_stop_cnt = ckpt.get("early_stop_cnt", 0)
        print(f"  → Tiếp tục từ epoch {start_epoch}, best_val_f1={best_val_f1:.4f}\n")

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

        # Logic kiểm tra chuyển giai đoạn (Transition Boundary)
        if cfg['training'].get('two_stage', False):
            if epoch <= freeze_epochs:
                # Đang ở Stage 1
                if epoch == start_epoch and not resume:  # Chỉ in/build lần đầu khi chạy từ đầu
                    print(f"[STAGE 1] Đóng băng Backbone. Train MLP Heads với LR={cfg['training'].get('frozen_lr', 1e-3)}")
                    model.freeze_backbone()
                    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, is_frozen=True, total_epochs=epochs)
            elif epoch == freeze_epochs + 1:
                # Chuyển giao sang Stage 2
                if epoch != start_epoch:  # Chỉ rebuild nếu không phải khôi phục trực tiếp từ Stage 2
                    print(f"\n[STAGE 2] 🔓 Mở khóa toàn bộ mạng. Rebuild Optimizer với LR={cfg['training'].get('unfrozen_lr', 3e-5)}")
                    model.unfreeze_backbone()
                    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, is_frozen=False, total_epochs=epochs)

        current_lr  = optimizer.param_groups[0]["lr"]

        print(f"[Epoch {epoch:02d}/{n_epochs}]  lr={current_lr:.2e}")

        # Train
        train_m = run_epoch(
            model, dataloaders["train"], criterion, optimizer,
            device, "train", age_mean, age_std, scaler=scaler,
            dry_run=dry_run, binary_mode=binary_mode
        )

        # Validate
        val_m = run_epoch(
            model, dataloaders["val"], criterion, None,
            device, "val", age_mean, age_std,
            dry_run=dry_run, binary_mode=binary_mode
        )

        # Scheduler step
        scheduler.step()

        # Print summary
        elapsed = time.time() - epoch_start
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

        # Best model
        val_f1 = val_m["val_f1_macro"]
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stop_cnt = 0
            if cfg["output"].get("save_best_model", True) and not dry_run:
                save_checkpoint({
                    "epoch":           epoch,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val_f1":     best_val_f1,
                    "val_metrics":     val_m,
                    "config":          cfg,
                }, best_path)
        else:
            early_stop_cnt += 1
            print(f"  [EarlyStopping] {early_stop_cnt}/{patience} — best F1={best_val_f1:.4f}")
            if early_stop_cnt >= patience and not dry_run:
                print(f"\n  *** EARLY STOPPING tại epoch {epoch} ***\n")
                break

        # Last checkpoint
        if not dry_run:
            save_checkpoint({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_f1":     best_val_f1,
                "early_stop_cnt":  early_stop_cnt,
                "config":          cfg,
            }, last_path)

    # --- Đánh giá trên Test set với best model ---
    if not dry_run and best_path.exists():
        print(f"\n{'─'*60}")
        print(f"  Đánh giá TEST SET với best model (val F1={best_val_f1:.4f})")
        print(f"{'─'*60}")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

        # 1. Tìm ngưỡng động tối ưu trên tập Validation
        print("[Dynamic Thresholding] Đang quét tìm ngưỡng tối ưu trên tập Validation...")
        val_probs, val_targets = get_predictions(model, dataloaders["val"], device, binary_mode=binary_mode)
        best_thresholds = find_best_thresholds(val_probs, val_targets)
        
        if binary_mode:
            print(f"  → Ngưỡng tối ưu tìm được cho phân loại nhị phân: {best_thresholds[0]:.2f}\n")
        else:
            print("  → Ngưỡng tối ưu tìm được cho 8 bệnh lý:")
            from src.utils import get_label_names
            lbl_names = get_label_names()
            for idx, lbl in enumerate(LABELS):
                print(f"    - {lbl} ({lbl_names[lbl]}): {best_thresholds[idx]:.2f}")
            print()

        # 2. Đánh giá Test với ngưỡng mặc định 0.5
        test_m_default = run_epoch(
            model, dataloaders["test"], criterion, None,
            device, "test", age_mean, age_std, threshold=0.5,
            binary_mode=binary_mode
        )
        print(
            f"  [NGƯỠNG MẶC ĐỊNH 0.5] "
            f"TEST F1-macro={test_m_default['test_f1_macro']:.4f} "
            f"| AUC-ROC={test_m_default['test_auc_roc']:.4f} "
            f"| Age MAE={test_m_default['test_age_mae']:.2f} yrs"
        )

        # 3. Đánh giá Test với ngưỡng động tối ưu vừa tìm được
        test_m_opt = run_epoch(
            model, dataloaders["test"], criterion, None,
            device, "test", age_mean, age_std, threshold=best_thresholds,
            binary_mode=binary_mode
        )
        print(
            f"  [NGƯỠNG TỐI ƯU ĐỘNG]  "
            f"TEST F1-macro={test_m_opt['test_f1_macro']:.4f} "
            f"| AUC-ROC={test_m_opt['test_auc_roc']:.4f} "
            f"| Age MAE={test_m_opt['test_age_mae']:.2f} yrs"
        )

        # Lưu test results
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
