"""
Inference module: load model + predict từ ảnh đáy mắt.
Hoạt động ở 2 chế độ:
  - Có best.pth → inference thật (bao gồm Grad-CAM heatmap)
  - Không có best.pth → demo mode (chỉ hiện preprocessing)

Grad-CAM:
  - CNN (EfficientNet-B0): hook vào layer cuối của backbone → gradient-weighted activation map
  - Swin Transformer: hook vào layer norm cuối của backbone → attention-based heatmap
"""
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

# ── Constants ──
LABELS = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract',
          'AMD', 'Hypertension', 'Myopia', 'Other']
LABEL_ICONS = ['👁️', '🩸', '🟢', '🔵', '🟡', '❤️‍🩹', '👓', '📋']
AGE_MEAN = 58.14
AGE_STD  = 11.26
IMG_SIZE = 224


# ── Preprocessing ──
def crop_roi(img, tol=7):
    """ROI Crop — loại bỏ viền đen xung quanh ảnh đáy mắt."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > tol
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return cv2.resize(img, (512, 512))
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    return cv2.resize(img[r0:r1+1, c0:c1+1], (512, 512))


def ben_graham(img, sigma_ratio=1/6, scale=128):
    """Ben Graham Normalization — chuẩn hóa ánh sáng không đều."""
    h, w = img.shape[:2]
    sigma = int(max(h, w) * sigma_ratio)
    if sigma % 2 == 0:
        sigma += 1
    local = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigmaX=sigma)
    return np.clip(img.astype(np.float32) - local + scale, 0, 255).astype(np.uint8)


def apply_clahe(img):
    """CLAHE — tăng cường tương phản cục bộ."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


def preprocess_image(img_bgr):
    """Pipeline tiền xử lý đầy đủ: ROI → Ben Graham → CLAHE."""
    step1_roi = crop_roi(img_bgr)
    step2_bg  = ben_graham(step1_roi)
    step3_enh = apply_clahe(step2_bg)
    return step1_roi, step2_bg, step3_enh


def prepare_tensor(img_bgr, img_size=IMG_SIZE):
    """Chuyển ảnh BGR → tensor normalized (1, 3, H, W)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)


# ── Model Loading ──
def load_model(weights_path, model_type='cnn', device='cpu'):
    """Load model từ file best.pth."""
    import sys
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.models import build_model
    model = build_model(model_type=model_type, pretrained=False, img_size=IMG_SIZE)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ── Inference ──
@torch.no_grad()
def predict(model, img_bgr, device='cpu'):
    """Chạy inference → trả về probabilities + predicted age."""
    tensor = prepare_tensor(img_bgr).to(device)
    output = model(tensor)

    logits   = output['logits'].squeeze(0)
    age_norm = output['age_pred'].squeeze().item()

    probs = torch.sigmoid(logits).cpu().numpy()
    age   = age_norm * AGE_STD + AGE_MEAN

    return {
        'probabilities': {LABELS[i]: float(probs[i]) for i in range(len(LABELS))},
        'predicted_age': round(age, 1),
        'raw_probs': probs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM Implementation
# ─────────────────────────────────────────────────────────────────────────────

def _find_last_conv_layer(model):
    """
    Tìm lớp Conv2d cuối cùng trong backbone EfficientNet.
    Dùng để hook Grad-CAM.
    """
    last_conv = None
    for name, module in model.backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = (name, module)
    return last_conv


def _find_swin_norm_layer(model):
    """
    Tìm LayerNorm cuối trong backbone Swin Transformer.
    Dùng để hook Grad-CAM cho Swin.
    """
    last_norm = None
    for name, module in model.backbone.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            last_norm = (name, module)
    return last_norm


def compute_gradcam(
    model,
    img_bgr: np.ndarray,
    target_label_idx: Optional[int] = None,
    device: str = 'cpu',
    model_type: str = 'cnn',
) -> np.ndarray:
    """
    Tính Grad-CAM heatmap cho ảnh đầu vào.

    Args:
        model: Model đã load (EfficientNetMTL hoặc SwinMTL).
        img_bgr: Ảnh BGR (đã qua preprocess_image).
        target_label_idx: Index nhãn muốn visualize (0-7).
                          Nếu None → tự chọn nhãn có xác suất cao nhất.
        device: 'cpu' hoặc 'cuda'.
        model_type: 'cnn' hoặc 'swin'.

    Returns:
        heatmap_bgr: Ảnh BGR kích thước (H, W, 3) — heatmap đã overlay lên ảnh gốc.
        cam_raw: numpy array (H, W) giá trị [0, 1] — raw CAM trước khi overlay.
    """
    model.eval()
    tensor = prepare_tensor(img_bgr, img_size=IMG_SIZE).to(device)
    tensor.requires_grad_(True)

    # Storage cho activations và gradients
    activations = {}
    gradients   = {}

    # ── Tìm target layer và đăng ký hooks ──
    if model_type == 'cnn':
        # EfficientNet: lấy activation map từ head_conv (lớp Conv cuối backbone)
        # Tìm layer có tên chứa "head_conv" hoặc Conv cuối cùng
        target_layer = None
        for name, module in model.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                target_name  = name

        if target_layer is None:
            # Fallback: dùng gradient của input
            return _gradcam_fallback(model, img_bgr, target_label_idx, device)

        def save_activation(module, inp, out):
            activations['feat'] = out.detach()

        def save_gradient(module, grad_in, grad_out):
            gradients['feat'] = grad_out[0].detach()

        hook_fwd = target_layer.register_forward_hook(save_activation)
        hook_bwd = target_layer.register_backward_hook(save_gradient)

        # Forward pass
        output = model(tensor)
        logits = output['logits']

        # Chọn target label
        probs = torch.sigmoid(logits)
        if target_label_idx is None:
            target_label_idx = int(probs.argmax().item())

        # Backward
        model.zero_grad()
        score = logits[0, target_label_idx]
        score.backward()

        hook_fwd.remove()
        hook_bwd.remove()

        # ── Tính CAM ──
        act  = activations['feat'][0]   # [C, H, W]
        grad = gradients['feat'][0]     # [C, H, W]

        # Global average pooling của gradients → weights
        weights = grad.mean(dim=(1, 2))  # [C]

        # Weighted combination of activation maps
        cam = torch.zeros(act.shape[1:], device=device)
        for i, w in enumerate(weights):
            cam += w * act[i]

        cam = F.relu(cam)

    else:
        # ── Swin Transformer: Gradient-weighted feature map ──
        # Swin không có spatial feature map sau GAP như CNN.
        # Ta hook vào layer TRƯỚC global average pooling (norm layer cuối).
        target_layer = None
        for name, module in model.backbone.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                target_layer = module
                target_name  = name

        if target_layer is None:
            return _gradcam_fallback(model, img_bgr, target_label_idx, device)

        def save_activation_swin(module, inp, out):
            activations['feat'] = out  # [B, N, C] — cần giữ grad

        def save_gradient_swin(module, grad_in, grad_out):
            gradients['feat'] = grad_out[0]  # [B, N, C]

        hook_fwd = target_layer.register_forward_hook(save_activation_swin)
        hook_bwd = target_layer.register_full_backward_hook(
            lambda m, gi, go: gradients.update({'feat': go[0]})
        )

        # Forward
        output = model(tensor)
        logits = output['logits']
        probs  = torch.sigmoid(logits)

        if target_label_idx is None:
            target_label_idx = int(probs.argmax().item())

        model.zero_grad()
        score = logits[0, target_label_idx]
        score.backward()

        hook_fwd.remove()
        hook_bwd.remove()

        act  = activations['feat']   # [B, N, C] hoặc [B, C]
        grad = gradients.get('feat', None)

        if act is None:
            return _gradcam_fallback(model, img_bgr, target_label_idx, device)

        # Nếu act là 3D [B, N, C] → reshape về spatial map
        if act.ndim == 3:
            act  = act[0]   # [N, C]
            if grad is not None and grad.ndim == 3:
                grad = grad[0]  # [N, C]

            # Tính weights từ gradient (GAP trên token dimension)
            if grad is not None:
                weights = grad.mean(dim=0)   # [C]
                cam_tokens = (act * weights).sum(dim=-1)  # [N]
            else:
                cam_tokens = act.mean(dim=-1)  # [N]

            # Reshape về 2D spatial map (N = (H/patch_size)^2)
            n_tokens = cam_tokens.shape[0]
            hw = int(n_tokens ** 0.5)
            if hw * hw == n_tokens:
                cam = cam_tokens.reshape(hw, hw)
            else:
                # Không reshape được → dùng mean activation
                hw  = 7
                cam = cam_tokens[:hw*hw].reshape(hw, hw)
        else:
            # Nếu act là 2D [B, C] (sau GAP) → gradient * activation
            act  = act[0]   # [C]
            if grad is not None:
                grad = grad[0] if grad.ndim > 1 else grad
                cam_vec = (act * grad)
                cam = cam_vec.abs().unsqueeze(0).unsqueeze(0)  # [1,1,1,C]
            else:
                cam = act.abs().unsqueeze(0).unsqueeze(0)
            cam = cam.squeeze()
            cam = cam.unsqueeze(0).unsqueeze(0)
            cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0).float(),
                                size=(7, 7), mode='bilinear').squeeze()

        cam = F.relu(cam)

    # ── Normalize và resize CAM → kích thước ảnh gốc ──
    cam = cam.float()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    else:
        cam = torch.zeros_like(cam)

    cam_np = cam.detach().cpu().numpy()

    # Resize về kích thước ảnh gốc
    h, w = img_bgr.shape[:2]
    cam_resized = cv2.resize(cam_np, (w, h))

    # Tạo heatmap màu
    cam_uint8  = (cam_resized * 255).astype(np.uint8)
    heatmap    = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

    # Overlay lên ảnh gốc
    overlay    = cv2.addWeighted(img_bgr, 0.55, heatmap, 0.45, 0)

    return overlay, cam_resized


def _gradcam_fallback(model, img_bgr, target_label_idx, device):
    """
    Fallback đơn giản: Saliency map dựa trên gradient của input.
    Dùng khi không tìm được target layer phù hợp.
    """
    tensor = prepare_tensor(img_bgr, img_size=IMG_SIZE).to(device)
    tensor.requires_grad_(True)
    model.eval()

    output = model(tensor)
    logits = output['logits']
    probs  = torch.sigmoid(logits)

    if target_label_idx is None:
        target_label_idx = int(probs.argmax().item())

    model.zero_grad()
    score = logits[0, target_label_idx]
    score.backward()

    # Saliency map từ gradient của input tensor
    saliency = tensor.grad[0].abs().max(dim=0)[0]  # [H, W]
    saliency = saliency.detach().cpu().numpy()

    if saliency.max() > saliency.min():
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    h, w = img_bgr.shape[:2]
    saliency_resized = cv2.resize(saliency, (w, h))
    cam_uint8  = (saliency_resized * 255).astype(np.uint8)
    heatmap    = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    overlay    = cv2.addWeighted(img_bgr, 0.55, heatmap, 0.45, 0)
    return overlay, saliency_resized


def gradcam_for_label(model, img_bgr, label_name: str, device='cpu', model_type='cnn'):
    """
    Hàm tiện ích: tính Grad-CAM cho một nhãn bệnh cụ thể theo tên.

    Args:
        model: Model đã load.
        img_bgr: Ảnh BGR đã qua preprocessing.
        label_name: Tên nhãn ('Normal', 'Diabetes', ...).
        device, model_type: như compute_gradcam.

    Returns:
        (overlay_bgr, cam_raw) — như compute_gradcam.
    """
    if label_name in LABELS:
        idx = LABELS.index(label_name)
    else:
        idx = None
    return compute_gradcam(model, img_bgr, target_label_idx=idx,
                           device=device, model_type=model_type)
