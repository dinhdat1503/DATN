"""
Inference module cho ODIR-5K Siamese Binocular (Phase 1).
Hỗ trợ:
- Tiền xử lý ảnh đáy mắt (ROI Crop -> Ben Graham -> CLAHE).
- Tải mô hình Siamese (EfficientNet-B0 hoặc Swin-Tiny) và trọng số.
- Suy diễn đồng thời hai mắt (mắt trái, mắt phải) và dự đoán tuổi võng mạc.
- Tính toán Grad-CAM độc lập cho từng mắt sử dụng cơ chế tích lũy hooks.
"""

from __future__ import annotations

import os
import cv2
import functools
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

# ── Hằng số cấu hình ──
LABELS = ['Normal', 'Pathological']
IMG_SIZE = 384  # Cấu hình chuẩn của Phase 1

# ── Tiền xử lý ảnh (giữ nguyên quy trình y khoa chuẩn) ──
def crop_roi(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """ROI Crop — loại bỏ viền đen xung quanh ảnh đáy mắt."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > tol
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return cv2.resize(img, (512, 512))
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    return cv2.resize(img[r0:r1+1, c0:c1+1], (512, 512))


def ben_graham(img: np.ndarray, sigma_ratio: float = 1/6, scale: int = 128) -> np.ndarray:
    """Ben Graham Normalization — chuẩn hóa ánh sáng không đều."""
    h, w = img.shape[:2]
    sigma = int(max(h, w) * sigma_ratio)
    if sigma % 2 == 0:
        sigma += 1
    local = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigmaX=sigma)
    return np.clip(img.astype(np.float32) - local + scale, 0, 255).astype(np.uint8)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """CLAHE — tăng cường tương phản cục bộ."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


def preprocess_image(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pipeline tiền xử lý đầy đủ: ROI Crop -> Ben Graham -> CLAHE."""
    step1_roi = crop_roi(img_bgr)
    step2_bg  = ben_graham(step1_roi)
    step3_enh = apply_clahe(step2_bg)
    return step1_roi, step2_bg, step3_enh


def prepare_tensor(img_bgr: np.ndarray, img_size: int = IMG_SIZE) -> torch.Tensor:
    """Chuyển ảnh BGR -> tensor normalized (1, 3, H, W) theo ImageNet."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    
    # ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)


# ── Đọc thống kê tuổi (Z-score) ──
@functools.lru_cache(maxsize=1)
def load_age_stats() -> tuple[float, float]:
    """Tải thống kê tuổi võng mạc từ train.csv hoặc trả về mặc định nếu không tìm thấy."""
    project_root = Path(__file__).parent.parent
    train_csv = project_root / "archive" / "splits_clean" / "train.csv"
    if train_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(train_csv)
            df = df[df["Patient Age"] >= 5]
            ages = df["Patient Age"].values.astype(float)
            return float(ages.mean()), float(ages.std())
        except Exception:
            pass
    return 58.14, 11.26


# ── Nạp mô hình Siamese ──
def load_model(weights_path: str, model_type: str = 'cnn', device: str = 'cpu') -> torch.nn.Module:
    """Load mô hình Siamese từ file .pth."""
    import sys
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.models import build_model
    
    # Khởi tạo mô hình Siamese không tải pretrained ImageNet (vì sẽ load checkpoint)
    model = build_model(model_type=model_type, pretrained=False, img_size=IMG_SIZE)
    
    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    except Exception:
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)
        
    if "model_state" in state_dict:
        model.load_state_dict(state_dict["model_state"])
    else:
        model.load_state_dict(state_dict)
        
    model.to(device)
    model.eval()
    return model


# ── Suy diễn song nhãn ──
@torch.no_grad()
def predict(
    model: torch.nn.Module,
    left_img_bgr: Optional[np.ndarray],
    right_img_bgr: Optional[np.ndarray],
    left_missing: bool = False,
    right_missing: bool = False,
    device: str = 'cpu',
) -> dict:
    """Chạy suy diễn song nhãn -> trả về xác suất chẩn đoán và tuổi võng mạc."""
    # 1. Chuẩn bị tensor mắt trái
    if left_missing or left_img_bgr is None:
        left_tensor = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
        left_miss_t = torch.tensor([True], dtype=torch.bool)
    else:
        left_tensor = prepare_tensor(left_img_bgr)
        left_miss_t = torch.tensor([False], dtype=torch.bool)

    # 2. Chuẩn bị tensor mắt phải
    if right_missing or right_img_bgr is None:
        right_tensor = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
        right_miss_t = torch.tensor([True], dtype=torch.bool)
    else:
        right_tensor = prepare_tensor(right_img_bgr)
        right_miss_t = torch.tensor([False], dtype=torch.bool)

    left_tensor = left_tensor.to(device)
    right_tensor = right_tensor.to(device)
    left_miss_t = left_miss_t.to(device)
    right_miss_t = right_miss_t.to(device)

    # 3. Forward pass
    output = model(left_tensor, right_tensor, left_miss_t, right_miss_t)
    
    logits = output['logits'].squeeze(0)  # Logits của lớp bệnh lý
    age_norm = output['age_pred'].squeeze().item()

    # Phân loại nhị phân
    prob_pathological = torch.sigmoid(logits).item()
    prob_normal = 1.0 - prob_pathological

    # Giải chuẩn hóa tuổi
    age_mean, age_std = load_age_stats()
    predicted_age = age_norm * age_std + age_mean

    return {
        'prob_pathological': prob_pathological,
        'prob_normal': prob_normal,
        'predicted_age': round(predicted_age, 1),
    }


# ── Grad-CAM song nhãn ──
def compute_siamese_gradcam(
    model: torch.nn.Module,
    left_img_bgr: Optional[np.ndarray],
    right_img_bgr: Optional[np.ndarray],
    left_missing: bool = False,
    right_missing: bool = False,
    device: str = 'cpu',
    model_type: str = 'cnn',
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Tính Grad-CAM cho cả 2 mắt từ mạng Siamese backbone chia sẻ trọng số.
    
    Vì backbone được gọi 2 lần riêng biệt trong forward pass, ta sử dụng cơ chế hooks
    tích lũy dạng danh sách để tách riêng kích hoạt và gradient của mắt trái và mắt phải.
    """
    model.eval()
    
    # 1. Chuẩn bị đầu vào
    if left_missing or left_img_bgr is None:
        left_tensor = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
        left_miss_t = torch.tensor([True], dtype=torch.bool)
    else:
        left_tensor = prepare_tensor(left_img_bgr)
        left_miss_t = torch.tensor([False], dtype=torch.bool)

    if right_missing or right_img_bgr is None:
        right_tensor = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
        right_miss_t = torch.tensor([True], dtype=torch.bool)
    else:
        right_tensor = prepare_tensor(right_img_bgr)
        right_miss_t = torch.tensor([False], dtype=torch.bool)

    left_tensor = left_tensor.to(device).requires_grad_(True)
    right_tensor = right_tensor.to(device).requires_grad_(True)
    left_miss_t = left_miss_t.to(device)
    right_miss_t = right_miss_t.to(device)

    # 2. Tìm lớp trung gian để tính toán CAM
    target_layer = None
    if model_type == 'cnn':
        # Tìm lớp Conv2D cuối cùng trong backbone
        for name, module in model.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
    else:
        # Tìm LayerNorm cuối cùng trong backbone Swin
        for name, module in model.backbone.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                target_layer = module

    if target_layer is None:
        return None, None

    # Danh sách để tích lũy kích hoạt và gradient
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out.clone().detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].clone().detach())

    hook_fwd = target_layer.register_forward_hook(forward_hook)
    hook_bwd = target_layer.register_full_backward_hook(backward_hook)

    try:
        # 3. Forward pass & Backward pass trên score
        output = model(left_tensor, right_tensor, left_miss_t, right_miss_t)
        logits = output['logits']  # [1, 1]

        model.zero_grad()
        score = logits[0, 0]
        score.backward()
    finally:
        # Tháo hooks ngay lập tức
        hook_fwd.remove()
        hook_bwd.remove()

    # Nếu không tích lũy đủ cho cả hai lượt gọi backbone thì hủy bỏ
    if len(activations) < 2 or len(gradients) < 2:
        return None, None

    # Tách biệt:
    # Forward: 0 = Mắt trái, 1 = Mắt phải
    # Backward: 0 = Mắt phải, 1 = Mắt trái (do lan truyền ngược chạy từ cuối về)
    left_act = activations[0]
    left_grad = gradients[1]
    
    right_act = activations[1]
    right_grad = gradients[0]

    # Hàm tạo bản đồ nhiệt
    def generate_heatmap(act, grad, img_bgr):
        if model_type == 'cnn':
            act = act[0]
            grad = grad[0]
            weights = grad.mean(dim=(1, 2))  # [C]
            cam = torch.zeros(act.shape[1:], device=device)
            for i, w in enumerate(weights):
                cam += w * act[i]
            cam = F.relu(cam)
        else:
            # Swin Transformer
            if act.ndim == 4:
                # Trải phẳng không gian nếu là [B, H, W, C]
                act = act.flatten(1, 2)
                grad = grad.flatten(1, 2)

            if act.ndim == 3:
                act = act[0]   # [N, C]
                grad = grad[0]  # [N, C]
                weights = grad.mean(dim=0)  # [C]
                cam_tokens = (act * weights).sum(dim=-1)  # [N]
                n_tokens = cam_tokens.shape[0]
                hw = int(n_tokens ** 0.5)
                if hw * hw == n_tokens:
                    cam = cam_tokens.reshape(hw, hw)
                else:
                    hw = 7
                    cam = cam_tokens[:hw*hw].reshape(hw, hw)
            else:
                # Dự phòng cuối cùng cho Swin
                act = act[0]
                grad = grad[0]
                cam_vec = act * grad
                if cam_vec.ndim > 2:
                    cam = cam_vec.sum(dim=-1)
                else:
                    cam = cam_vec
            cam = F.relu(cam)

        cam = cam.float()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = torch.zeros_like(cam)

        cam_np = cam.detach().cpu().numpy()
        h, w = img_bgr.shape[:2]
        cam_resized = cv2.resize(cam_np, (w, h))

        # Áp bản đồ nhiệt màu JET
        cam_uint8 = (cam_resized * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_bgr, 0.55, heatmap, 0.45, 0)
        return overlay

    left_overlay = None if (left_missing or left_img_bgr is None) else generate_heatmap(left_act, left_grad, left_img_bgr)
    right_overlay = None if (right_missing or right_img_bgr is None) else generate_heatmap(right_act, right_grad, right_img_bgr)

    return left_overlay, right_overlay
