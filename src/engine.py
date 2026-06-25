"""
Động cơ huấn luyện & đánh giá (engine) cho ODIR-5K Phase 1 — Siamese nhị phân song nhãn.

Cung cấp:
- run_epoch: chạy 1 epoch train hoặc đánh giá (val/test), trả về metrics + dự đoán.
- fit: vòng lặp huấn luyện chính (two-stage freeze→unfreeze, AMP, gradient accumulation,
  cân chỉnh ngưỡng Youden trên val, early stopping theo AUC, lưu checkpoint + log).
- evaluate_test: đánh giá best model trên tập Test ở cả ngưỡng 0.5 và ngưỡng Youden tối ưu.

Toàn bộ chỉ phục vụ bài toán nhị phân song nhãn (không còn nhánh đa nhãn/đơn ảnh của Phase 0).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch

from src.metrics import compute_age_metrics, compute_binary_metrics, find_best_threshold

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# --- Tương thích API AMP (torch.amp mới ↔ torch.cuda.amp cũ trên Kaggle) ---
try:
    from torch.amp import GradScaler as _GradScaler
    from torch.amp import autocast as _autocast

    def make_autocast(enabled: bool):
        return _autocast("cuda", enabled=enabled)

    def make_scaler(enabled: bool):
        return _GradScaler("cuda", enabled=enabled)
except (ImportError, TypeError):  # torch cũ
    from torch.cuda.amp import GradScaler as _GradScaler  # type: ignore
    from torch.cuda.amp import autocast as _autocast      # type: ignore

    def make_autocast(enabled: bool):
        return _autocast(enabled=enabled)

    def make_scaler(enabled: bool):
        return _GradScaler(enabled=enabled)


# ---------------------------------------------------------------------------
# Một epoch train / eval
# ---------------------------------------------------------------------------

def run_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    mode: str,
    age_mean: float,
    age_std: float,
    scaler=None,
    accum_steps: int = 1,
    dry_run: bool = False,
    threshold: float = 0.5,
) -> tuple[dict, list, list]:
    """Chạy một epoch.

    Args:
        model: Mạng Siamese.
        loader: DataLoader song nhãn.
        criterion: MultiTaskLoss.
        optimizer: Optimizer (None nếu đánh giá).
        device: cpu/cuda.
        mode: "train" | "val" | "test".
        age_mean, age_std: thống kê tuổi để giải chuẩn hóa MAE.
        scaler: GradScaler cho AMP (chỉ khi train trên CUDA).
        accum_steps: số bước tích lũy gradient.
        dry_run: chỉ chạy 1 batch để test pipeline.
        threshold: ngưỡng tính metrics.

    Returns:
        (metrics, all_probs, all_targets)
    """
    is_train = (mode == "train")
    model.train() if is_train else model.eval()

    total_loss = total_cls = total_reg = 0.0
    n_samples = 0
    all_probs: list[float] = []
    all_targets: list[float] = []
    all_age_pred: list[float] = []
    all_age_true: list[float] = []

    use_amp = (device.type == "cuda")
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    if is_train and optimizer is not None:
        optimizer.zero_grad()

    iterator = loader
    if HAS_TQDM:
        iterator = tqdm(loader, desc=f"  {mode}", leave=False, ncols=90)

    n_batches = len(loader)

    with ctx:
        for batch_idx, batch in enumerate(iterator):
            left = batch["left_image"].to(device, non_blocking=True)
            right = batch["right_image"].to(device, non_blocking=True)
            left_m = batch["left_missing"].to(device)
            right_m = batch["right_missing"].to(device)
            labels = batch["label"].to(device)
            age_true = batch["age"].to(device)
            bs = left.size(0)

            with make_autocast(use_amp):
                output = model(left, right, left_m, right_m)
                logits = output["logits"]
                age_pred = output["age_pred"]
                loss, detail = criterion(logits, labels, age_pred, age_true)
                loss_back = loss / accum_steps if is_train else loss

            if is_train and optimizer is not None:
                is_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == n_batches)
                if scaler is not None and use_amp:
                    scaler.scale(loss_back).backward()
                    if is_step:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss_back.backward()
                    if is_step:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()

            total_loss += detail["loss_total"] * bs
            total_cls += detail["loss_cls"] * bs
            total_reg += detail["loss_reg"] * bs
            n_samples += bs

            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten().tolist())
            all_targets.extend(labels.detach().cpu().numpy().flatten().tolist())
            all_age_pred.extend(age_pred.detach().cpu().numpy().flatten().tolist())
            all_age_true.extend(age_true.detach().cpu().numpy().flatten().tolist())

            if dry_run:
                break

    n = max(n_samples, 1)
    metrics = {
        f"{mode}_loss": total_loss / n,
        f"{mode}_loss_cls": total_cls / n,
        f"{mode}_loss_reg": total_reg / n,
    }
    bm = compute_binary_metrics(all_probs, all_targets, threshold=threshold)
    for k in ("accuracy", "precision", "sensitivity", "specificity", "f1", "auc"):
        metrics[f"{mode}_{k}"] = bm[k]
    am = compute_age_metrics(all_age_pred, all_age_true, age_mean, age_std)
    metrics[f"{mode}_age_mae"] = am["mae"]
    metrics[f"{mode}_age_pearson"] = am["pearson"]

    return metrics, all_probs, all_targets


# ---------------------------------------------------------------------------
# Optimizer & Scheduler (hỗ trợ two-stage)
# ---------------------------------------------------------------------------

def build_optimizer_scheduler(model, cfg: dict, stage: str, total_epochs: int):
    """Tạo AdamW + CosineAnnealingLR theo giai đoạn.

    Args:
        stage: "frozen" (Stage 1 — chỉ train head) hoặc "unfrozen" (Stage 2 — toàn mạng).
    """
    tr = cfg["training"]
    weight_decay = float(tr.get("weight_decay", 0.05))
    eta_min = float(tr.get("eta_min", 1e-6))
    freeze_epochs = int(tr.get("freeze_epochs", 5))

    if stage == "frozen":
        lr = float(tr.get("frozen_lr", 1e-3))
        params = [p for p in model.parameters() if p.requires_grad]
        t_max = max(1, freeze_epochs)
    else:
        lr = float(tr.get("unfrozen_lr", 1e-4))
        params = model.parameters()
        t_max = max(1, total_epochs - freeze_epochs) if tr.get("two_stage", True) else max(1, total_epochs)

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    return optimizer, scheduler


def _log_csv(log_path: Path, row: dict) -> None:
    """Ghi một dòng metrics vào file CSV (tự tạo header lần đầu)."""
    exists = log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Vòng lặp huấn luyện chính
# ---------------------------------------------------------------------------

def fit(
    model: torch.nn.Module,
    dataloaders: dict,
    criterion,
    cfg: dict,
    device: torch.device,
    results_dir: Path,
    age_mean: float,
    age_std: float,
    dry_run: bool = False,
    resume: str | None = None,
) -> Path:
    """Huấn luyện mô hình với two-stage + early stopping theo AUC + calibration Youden.

    Returns:
        Đường dẫn tới best_model.pth.
    """
    tr = cfg["training"]
    epochs = 1 if dry_run else int(tr["epochs"])
    two_stage = bool(tr.get("two_stage", True))
    freeze_epochs = int(tr.get("freeze_epochs", 5))
    accum_steps = int(tr.get("gradient_accumulation_steps", 1))
    patience = int(tr.get("early_stopping_patience", 10))

    scaler = make_scaler(device.type == "cuda")
    log_path = results_dir / "training_log.csv"
    best_path = results_dir / "best_model.pth"
    last_path = results_dir / "last_model.pth"

    # --- Khởi tạo giai đoạn ---
    if two_stage:
        model.freeze_backbone()
        stage = "frozen"
    else:
        stage = "unfrozen"
    optimizer, scheduler = build_optimizer_scheduler(model, cfg, stage, epochs)

    start_epoch = 1
    best_auc = 0.0
    best_thresh = 0.5
    no_improve = 0

    # --- Resume (best-effort) ---
    if resume and Path(resume).exists():
        print(f"[Resume] Nạp checkpoint: {resume}")
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_auc = ckpt.get("best_auc", 0.0)
        best_thresh = ckpt.get("best_thresh", 0.5)
        no_improve = ckpt.get("no_improve", 0)
        # Xác định lại giai đoạn theo epoch tiếp tục
        new_stage = "frozen" if (two_stage and start_epoch <= freeze_epochs) else "unfrozen"
        if new_stage == "unfrozen":
            model.unfreeze_backbone()
        optimizer, scheduler = build_optimizer_scheduler(model, cfg, new_stage, epochs)
        if ckpt.get("stage") == new_stage:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
                scheduler.load_state_dict(ckpt["scheduler_state"])
            except Exception:
                print("  [Resume] Không nạp được optimizer/scheduler — khởi tạo mới.")
        stage = new_stage
        print(f"  → Tiếp tục từ epoch {start_epoch}, best_auc={best_auc:.4f}, stage={stage}")

    print(f"\n{'─'*64}")
    print(f"  Bắt đầu huấn luyện: {epochs} epoch | two_stage={two_stage} | "
          f"freeze_epochs={freeze_epochs} | accum={accum_steps} | patience={patience}")
    if dry_run:
        print("  [DRY-RUN] chỉ 1 epoch / 1 batch để kiểm tra pipeline")
    print(f"{'─'*64}\n")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        # Chuyển Stage 1 → Stage 2
        if two_stage and not dry_run and epoch == freeze_epochs + 1 and stage == "frozen":
            print(f"\n[STAGE 2] 🔓 Mở khóa toàn mạng — LR={tr.get('unfrozen_lr', 1e-4)}")
            model.unfreeze_backbone()
            optimizer, scheduler = build_optimizer_scheduler(model, cfg, "unfrozen", epochs)
            stage = "unfrozen"
        elif epoch == start_epoch and two_stage and stage == "frozen":
            print(f"[STAGE 1] ❄️  Đóng băng backbone — train head, LR={tr.get('frozen_lr', 1e-3)}")

        lr = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch:02d}/{epochs}] lr={lr:.2e} stage={stage}")

        train_m, _, _ = run_epoch(
            model, dataloaders["train"], criterion, optimizer, device, "train",
            age_mean, age_std, scaler=scaler, accum_steps=accum_steps, dry_run=dry_run,
        )

        val_m, val_probs, val_targets = run_epoch(
            model, dataloaders["val"], criterion, None, device, "val",
            age_mean, age_std, dry_run=dry_run, threshold=0.5,
        )

        # Cân chỉnh ngưỡng Youden trên val (dùng lại dự đoán đã có — không forward lại)
        cur_thresh = find_best_threshold(val_probs, val_targets)
        opt = compute_binary_metrics(val_probs, val_targets, threshold=cur_thresh)
        for k in ("accuracy", "precision", "sensitivity", "specificity", "f1", "auc"):
            val_m[f"val_{k}"] = opt[k]
        val_m["val_threshold"] = cur_thresh

        scheduler.step()
        elapsed = time.time() - t0

        print(f"  train_loss={train_m['train_loss']:.4f} | val_loss={val_m['val_loss']:.4f} "
              f"| val_AUC={val_m['val_auc']:.4f} | val_F1={val_m['val_f1']:.4f} "
              f"| Sens={val_m['val_sensitivity']:.4f} | Spec={val_m['val_specificity']:.4f} "
              f"| thr={cur_thresh:.2f} | MAE={val_m['val_age_mae']:.2f}y [{elapsed:.1f}s]")

        _log_csv(log_path, {"epoch": epoch, "lr": lr, "stage": stage, **train_m, **val_m})

        # Early stopping theo val_auc
        val_auc = val_m["val_auc"]
        improved = val_auc > best_auc
        if improved:
            best_auc = val_auc
            best_thresh = cur_thresh
            no_improve = 0
            if cfg["output"].get("save_best_model", True) and not dry_run:
                torch.save({
                    "epoch": epoch, "stage": stage,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_auc": best_auc, "best_thresh": best_thresh,
                    "no_improve": no_improve, "val_metrics": val_m, "config": cfg,
                }, best_path)
                print(f"  ✓ Lưu best_model (val_AUC={best_auc:.4f}, thr={best_thresh:.2f})")
        else:
            no_improve += 1
            print(f"  [EarlyStopping] {no_improve}/{patience} — best val_AUC={best_auc:.4f}")

        if not dry_run:
            torch.save({
                "epoch": epoch, "stage": stage,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_auc": best_auc, "best_thresh": best_thresh,
                "no_improve": no_improve, "config": cfg,
            }, last_path)

        if no_improve >= patience and not dry_run:
            print(f"\n  *** EARLY STOPPING tại epoch {epoch} (best val_AUC={best_auc:.4f}) ***\n")
            break

    print(f"\n[Train xong] best val_AUC={best_auc:.4f}, best_thresh={best_thresh:.2f}")
    return best_path


# ---------------------------------------------------------------------------
# Đánh giá trên tập Test
# ---------------------------------------------------------------------------

def evaluate_test(
    model: torch.nn.Module,
    dataloaders: dict,
    criterion,
    device: torch.device,
    age_mean: float,
    age_std: float,
    results_dir: Path,
    exp_name: str,
    best_path: Path,
) -> dict:
    """Đánh giá best model trên Test ở cả ngưỡng 0.5 và ngưỡng Youden (tìm trên val)."""
    if not best_path.exists():
        print(f"[CẢNH BÁO] Không có best_model tại {best_path} — bỏ qua đánh giá test.")
        return {}

    print(f"\n{'─'*64}\n  ĐÁNH GIÁ TEST SET — {exp_name}\n{'─'*64}")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # Tìm ngưỡng tối ưu trên val
    _, val_probs, val_targets = run_epoch(
        model, dataloaders["val"], criterion, None, device, "val",
        age_mean, age_std, threshold=0.5,
    )
    best_thresh = find_best_threshold(val_probs, val_targets)
    print(f"  Ngưỡng Youden tối ưu (trên val): {best_thresh:.2f}")

    # Test — 1 forward pass duy nhất, tính metrics ở cả 2 ngưỡng
    test_m_05, test_probs, test_targets = run_epoch(
        model, dataloaders["test"], criterion, None, device, "test",
        age_mean, age_std, threshold=0.5,
    )
    opt = compute_binary_metrics(test_probs, test_targets, threshold=best_thresh)

    def _fmt(m, acc_key="test_accuracy", prefix="test_"):
        return (f"Acc={m[prefix+'accuracy']:.4f} AUC={m[prefix+'auc']:.4f} "
                f"F1={m[prefix+'f1']:.4f} Sens={m[prefix+'sensitivity']:.4f} "
                f"Spec={m[prefix+'specificity']:.4f}")

    print(f"  [Ngưỡng 0.5] {_fmt(test_m_05)} | Age MAE={test_m_05['test_age_mae']:.2f}y")
    print(f"  [Ngưỡng {best_thresh:.2f}] Acc={opt['accuracy']:.4f} AUC={opt['auc']:.4f} "
          f"F1={opt['f1']:.4f} Sens={opt['sensitivity']:.4f} Spec={opt['specificity']:.4f}")

    results = {
        "experiment": exp_name,
        "best_val_auc": float(ckpt.get("best_auc", 0.0)),
        "optimal_threshold": float(best_thresh),
        "metrics_threshold_0.5": test_m_05,
        "metrics_threshold_optimal": {
            "test_accuracy": opt["accuracy"],
            "test_precision": opt["precision"],
            "test_sensitivity": opt["sensitivity"],
            "test_specificity": opt["specificity"],
            "test_f1": opt["f1"],
            "test_auc": opt["auc"],
            "test_age_mae": test_m_05["test_age_mae"],
            "test_age_pearson": test_m_05["test_age_pearson"],
        },
    }
    out_path = results_dir / "test_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  → Lưu kết quả test: {out_path}")
    return results
