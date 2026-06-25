"""
ODIR-5K — Web App Chẩn Đoán Bệnh Lý Nhãn Khoa Song Nhãn (Phase 1)
Streamlit app: Tải ảnh mắt trái & phải -> Tiền xử lý -> Chẩn đoán Siamese + Grad-CAM
Giao diện màu trắng tối giản, không sử dụng sticker hay icon.
"""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st
from pathlib import Path

from inference import (
    preprocess_image,
    predict,
    load_model,
    LABELS,
    IMG_SIZE,
    compute_siamese_gradcam,
)

# ── Load model cached ở mức module (Tránh định nghĩa trong khối if) ──
@st.cache_resource
def get_model(path, mtype):
    return load_model(str(path), model_type=mtype, device='cpu')

def detect_model_type(weight_path: Path) -> str:
    """Tự động nhận diện model_type từ config.yaml hoặc tên thư mục chứa trọng số."""
    config_path = weight_path.parent / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                mtype = cfg.get("model_type")
                if mtype in ("cnn", "swin"):
                    return mtype
        except Exception:
            pass
            
    folder_name = weight_path.parent.name.lower()
    if "swin" in folder_name:
        return "swin"
    return "cnn"

# ── Cấu hình trang (Không dùng emoji) ──
st.set_page_config(
    page_title="ODIR-5K - Chẩn Đoán Nhãn Khoa Song Nhãn",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS cho giao diện màu trắng tối giản y khoa (No icons, no stickers) ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    /* Theme màu sáng y khoa cao cấp */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #FAFCFF !important;
        color: #1E293B !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #F1F5F9 !important;
        border-right: 1px solid #E2E8F0;
    }

    /* Container chính */
    .main .block-container {
        padding-top: 2.5rem;
        max-width: 1200px;
    }

    /* Header khởi đầu trang với gradient y khoa nhẹ nhàng */
    .header-container {
        border-bottom: 3px solid #0EA5E9;
        padding-bottom: 1.5rem;
        margin-bottom: 2.5rem;
        background: linear-gradient(135deg, #FFFFFF, #F0F9FF);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.05);
        border-left: 4px solid #0EA5E9;
    }
    .header-title {
        color: #0F172A;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 2.2rem;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .header-subtitle {
        color: #475569;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.05rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }

    /* Các mục tiêu đề phân đoạn */
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 2.5rem;
        margin-bottom: 1.2rem;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 0.5rem;
        letter-spacing: -0.01em;
        position: relative;
    }
    .section-title::after {
        content: "";
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 60px;
        height: 2px;
        background-color: #0EA5E9;
    }

    /* Thẻ hiển thị trạng thái hệ thống */
    .status-badge {
        padding: 0.4rem 1rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
    }
    .status-ready {
        background-color: #F0FDF4;
        color: #15803D;
        border: 1px solid #BBF7D0;
    }
    .status-demo {
        background-color: #FFF7ED;
        color: #C2410C;
        border: 1px solid #FFEDD5;
    }

    /* Card kết quả chẩn đoán cao cấp */
    .diagnostic-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .diagnostic-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.08), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        border-color: #CBD5E1;
    }
    .diagnostic-result-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .diagnostic-result-value {
        font-size: 2.3rem;
        font-weight: 800;
        margin-top: 0.6rem;
        margin-bottom: 1.2rem;
        font-family: 'Outfit', sans-serif;
        letter-spacing: -0.02em;
    }
    .result-normal {
        color: #16A34A;
        text-shadow: 0 1px 2px rgba(22, 163, 74, 0.05);
    }
    .result-pathological {
        color: #DC2626;
        text-shadow: 0 1px 2px rgba(220, 38, 38, 0.05);
    }

    /* Khối số liệu thống kê chi tiết */
    .metric-grid {
        display: flex;
        flex-wrap: wrap;
        width: 100%;
        margin-top: 1.5rem;
        gap: 1rem;
    }
    .metric-item {
        flex: 1;
        min-width: 160px;
        background-color: #F8FAFC;
        border: 1px solid #F1F5F9;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        transition: background-color 0.2s ease;
    }
    .metric-item:hover {
        background-color: #F1F5F9;
    }
    .metric-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
        font-family: 'Outfit', sans-serif;
    }
    .metric-lbl {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 0.3rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    /* Nhãn hiển thị tên mắt */
    .eye-label {
        font-weight: 700;
        font-size: 1.15rem;
        margin-bottom: 0.8rem;
        color: #0F172A;
        border-left: 4px solid #0EA5E9;
        padding-left: 0.6rem;
        letter-spacing: -0.01em;
    }

    /* Nhãn bước tiền xử lý */
    .step-badge {
        background-color: #F1F5F9;
        color: #475569;
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 0.6rem;
        border: 1px solid #E2E8F0;
        letter-spacing: 0.02em;
    }

    /* Kiểu input file upload Streamlit cao cấp */
    .stFileUploader > div {
        border-radius: 8px;
        border: 1px dashed #CBD5E1;
        background-color: #FFFFFF;
        padding: 0.5rem;
        transition: all 0.2s ease;
    }
    .stFileUploader > div:hover {
        border-color: #0EA5E9;
        background-color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar Cấu hình (Không dùng icon) ──
with st.sidebar:
    st.markdown("### CẤU HÌNH HỆ THỐNG")

    # Quét tự động các file trọng số
    weights_dir = Path(__file__).parent.parent / "results"
    
    show_last_model = st.checkbox("Hiển thị trọng số epoch cuối (last_model.pth)", value=False)
    
    experiments = []
    if weights_dir.exists():
        for p in weights_dir.glob("**/best_model.pth"):
            experiments.append(p.parent.name)
        experiments = sorted(list(set(experiments)))

    # Tự động đọc optimal_threshold từ test_results.json nếu có
    default_threshold = 0.5
    selected_weight = None
    model_loaded = False
    model_key = "cnn"

    if experiments:
        selected_exp = st.selectbox(
            "Thực nghiệm (Experiment)",
            experiments,
            help="Chọn cấu hình thực nghiệm đã huấn luyện."
        )

        exp_dir = weights_dir / selected_exp
        available_files = []
        if (exp_dir / "best_model.pth").exists():
            available_files.append("best_model.pth")
        if show_last_model and (exp_dir / "last_model.pth").exists():
            available_files.append("last_model.pth")

        if available_files:
            selected_filename = st.selectbox(
                "Kiểu trọng số (Weights File)",
                available_files,
                format_func=lambda f: "Tối ưu nhất (best_model.pth)" if f == "best_model.pth" else "Epoch cuối (last_model.pth)"
            )
            selected_weight = exp_dir / selected_filename
            model_loaded = True

            # Tự động nhận diện cấu hình mô hình từ thư mục kết quả
            model_key = detect_model_type(selected_weight)
            model_type_display = "Swin Transformer" if model_key == "swin" else "EfficientNet-B0 (CNN)"
            st.markdown(f"Kiến trúc tự động nhận diện: **{model_type_display}**")

            # Đọc optimal_threshold
            test_results_path = selected_weight.parent / "test_results.json"
            if test_results_path.exists():
                try:
                    import json
                    with open(test_results_path, "r", encoding="utf-8") as f:
                        res = json.load(f)
                        opt_thresh = res.get("optimal_threshold", 0.5)
                        default_threshold = round(float(opt_thresh), 2)
                except Exception:
                    pass
        else:
            st.error("Không tìm thấy tệp trọng số nào cho thực nghiệm này.")
    else:
        st.warning("Cảnh báo: Chưa có tệp trọng số trong thư mục results. Đang ở chế độ Demo.")

    st.markdown("---")

    # Ngưỡng phân loại
    threshold = st.slider(
        "Ngưỡng chẩn đoán",
        min_value=0.1,
        max_value=0.9,
        value=default_threshold,
        step=0.05,
        help="Mặc định sử dụng ngưỡng tối ưu Youden của mô hình được load."
    )

    st.markdown("---")
    st.markdown("### THÔNG TIN BỆNH NHÂN")
    real_age = st.number_input(
        "Tuổi thật của bệnh nhân",
        min_value=0, max_value=120, value=0, step=1,
        help="Nhập tuổi để tính toán Retinal Age Gap (Tuổi sinh học - Tuổi thật)."
    )

    st.markdown("---")
    st.markdown("### CẤU HÌNH GRAD-CAM")
    show_gradcam = st.toggle("Hiển thị bản đồ nhiệt Grad-CAM", value=True)

    st.markdown("---")
    st.markdown("### MÔ TẢ MÔ HÌNH")
    st.text(f"Kiến trúc: {model_key.upper()}")
    st.text(f"Input size: {IMG_SIZE}x{IMG_SIZE}")
    st.text("Nhiệm vụ: Phân loại nhị phân song nhãn")

    st.markdown("---")
    st.markdown(
        "<p style='color: #6C757D; font-size: 0.75rem;'>"
        "ODIR-5K Siamese Binocular Classification<br>"
        "Ngô Đình Đạt - Lớp 64HTTT2<br>"
        "GVHD: TS. Lê Thị Tú Kiên</p>",
        unsafe_allow_html=True,
    )


# ── Tiêu đề trang (Không dùng icon) ──
st.markdown("""
<div class="header-container">
    <h1 class="header-title">Hệ Thống Chẩn Đoán Bệnh Lý Nhãn Khoa Song Nhãn</h1>
    <p class="header-subtitle">Phân tích ảnh đáy mắt song nhãn bằng AI - Mạng Siamese đa nhiệm (EfficientNet-B0 / Swin Transformer)</p>
</div>
""", unsafe_allow_html=True)

# Trạng thái nguồn tải model
if model_loaded:
    st.markdown('<span class="status-badge status-ready">Mô hình sẵn sàng</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="status-badge status-demo">Chế độ Demo - Chỉ hiển thị tiền xử lý ảnh</span>', unsafe_allow_html=True)


# ── Chế độ Debug: Đọc ảnh từ URL query params left_file / right_file ──
_params = st.query_params
_debug_left_path = _params.get("left_file", None)
_debug_right_path = _params.get("right_file", None)
_debug_left_img: "np.ndarray | None" = None
_debug_right_img: "np.ndarray | None" = None

if _debug_left_path:
    _p = Path(_debug_left_path)
    if _p.exists():
        _debug_left_img = cv2.imread(str(_p))

if _debug_right_path:
    _p = Path(_debug_right_path)
    if _p.exists():
        _debug_right_img = cv2.imread(str(_p))

_is_debug_mode = (_debug_left_img is not None) or (_debug_right_img is not None)

if _is_debug_mode:
    st.info(
        f"**Chế độ Debug** — Đọc ảnh từ URL query params.\n\n"
        f"- Mắt trái: `{_debug_left_path}`\n"
        f"- Mắt phải: `{_debug_right_path}`",
        icon="🔍"
    )

# ── Vùng tải ảnh song nhãn (Trái / Phải song song) ──
col_uploader_l, col_uploader_r = st.columns(2)

with col_uploader_l:
    st.markdown('<div class="eye-label">Mắt trái (Left Eye)</div>', unsafe_allow_html=True)
    left_missing = st.checkbox("Thiếu ảnh mắt trái", value=False)
    left_file = None
    if not left_missing and not _is_debug_mode:
        left_file = st.file_uploader(
            "Tải ảnh đáy mắt trái",
            type=["jpg", "jpeg", "png", "bmp"],
            key="left_eye"
        )
    elif _is_debug_mode and _debug_left_img is not None:
        st.caption(f"Debug: {Path(_debug_left_path).name}")
        st.image(cv2.cvtColor(_debug_left_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    elif not left_missing:
        left_file = st.file_uploader(
            "Tải ảnh đáy mắt trái",
            type=["jpg", "jpeg", "png", "bmp"],
            key="left_eye"
        )

with col_uploader_r:
    st.markdown('<div class="eye-label">Mắt phải (Right Eye)</div>', unsafe_allow_html=True)
    right_missing = st.checkbox("Thiếu ảnh mắt phải", value=False)
    right_file = None
    if not right_missing and not _is_debug_mode:
        right_file = st.file_uploader(
            "Tải ảnh đáy mắt phải",
            type=["jpg", "jpeg", "png", "bmp"],
            key="right_eye"
        )
    elif _is_debug_mode and _debug_right_img is not None:
        st.caption(f"Debug: {Path(_debug_right_path).name}")
        st.image(cv2.cvtColor(_debug_right_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    elif not right_missing:
        right_file = st.file_uploader(
            "Tải ảnh đáy mắt phải",
            type=["jpg", "jpeg", "png", "bmp"],
            key="right_eye"
        )


# Kiểm tra xem có ít nhất một mắt có ảnh để phân tích
if _is_debug_mode:
    has_left = (not left_missing) and (_debug_left_img is not None)
    has_right = (not right_missing) and (_debug_right_img is not None)
else:
    has_left = (not left_missing) and (left_file is not None)
    has_right = (not right_missing) and (right_file is not None)

if has_left or has_right:
    left_img = None
    right_img = None
    
    # ── PHẦN 1: Tiến trình tiền xử lý ảnh ──
    st.markdown('<div class="section-title">Quy Trình Tiền Xử Lý Ảnh</div>', unsafe_allow_html=True)
    
    prep_col_l, prep_col_r = st.columns(2)
    
    left_steps = None
    right_steps = None
    
    with prep_col_l:
        st.markdown("#### Mắt trái")
        if has_left:
            if _is_debug_mode:
                left_img = _debug_left_img
            else:
                left_bytes = np.frombuffer(left_file.read(), np.uint8)
                left_img = cv2.imdecode(left_bytes, cv2.IMREAD_COLOR)
            if left_img is not None:
                step1_roi_l, step2_bg_l, step3_enh_l = preprocess_image(left_img)
                left_steps = (step1_roi_l, step2_bg_l, step3_enh_l)
                
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.markdown('<span class="step-badge">ROI Crop</span>', unsafe_allow_html=True)
                    st.image(cv2.cvtColor(step1_roi_l, cv2.COLOR_BGR2RGB), use_container_width=True)
                with sc2:
                    st.markdown('<span class="step-badge">Ben Graham</span>', unsafe_allow_html=True)
                    st.image(cv2.cvtColor(step2_bg_l, cv2.COLOR_BGR2RGB), use_container_width=True)
                with sc3:
                    st.markdown('<span class="step-badge">CLAHE</span>', unsafe_allow_html=True)
                    st.image(cv2.cvtColor(step3_enh_l, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.error("Lỗi đọc ảnh mắt trái.")
        else:
            st.text("Mắt trái không có dữ liệu (Thiếu mắt).")
            
    with prep_col_r:
        st.markdown("#### Mắt phải")
        if has_right:
            if _is_debug_mode:
                right_img = _debug_right_img
            else:
                right_bytes = np.frombuffer(right_file.read(), np.uint8)
                right_img = cv2.imdecode(right_bytes, cv2.IMREAD_COLOR)
            if right_img is not None:
                step1_roi_r, step2_bg_r, step3_enh_r = preprocess_image(right_img)
                right_steps = (step1_roi_r, step2_bg_r, step3_enh_r)
                
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.markdown('<span class="step-badge">ROI Crop</span>', unsafe_allow_html=True)
                    st.image(cv2.cvtColor(step1_roi_r, cv2.COLOR_BGR2RGB), use_container_width=True)
                with sc2:
                    st.markdown('<span class="step-badge">Ben Graham</span>', unsafe_allow_html=True)
                    st.image(cv2.cvtColor(step2_bg_r, cv2.COLOR_BGR2RGB), use_container_width=True)
                with sc3:
                    st.markdown('<span class="step-badge">CLAHE</span>', unsafe_allow_html=True)
                    st.image(cv2.cvtColor(step3_enh_r, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.error("Lỗi đọc ảnh mắt phải.")
        else:
            st.text("Mắt phải không có dữ liệu (Thiếu mắt).")
            
    st.markdown("---")

    # ── PHẦN 2: Kết quả chẩn đoán mô hình Siamese ──
    if model_loaded and selected_weight:
        st.markdown('<div class="section-title">Kết Quả Chẩn Đoán Từ Mô Hình</div>', unsafe_allow_html=True)
        
        with st.spinner("Mô hình đang thực hiện chẩn đoán song nhãn..."):
            model = get_model(selected_weight, model_key)
            
            # Lấy ảnh đầu vào đã qua CLAHE (bước 3)
            l_input = left_steps[2] if left_steps is not None else None
            r_input = right_steps[2] if right_steps is not None else None
            
            result = predict(
                model=model,
                left_img_bgr=l_input,
                right_img_bgr=r_input,
                left_missing=not has_left,
                right_missing=not has_right,
                device='cpu'
            )
            
        prob_pathological = result['prob_pathological']
        predicted_age = result['predicted_age']
        
        # Xác định kết quả
        is_pathological = (prob_pathological >= threshold)
        diag_class_str = "result-pathological" if is_pathological else "result-normal"
        diag_text = "Bệnh lý (Pathological)" if is_pathological else "Bình thường (Normal)"
        diag_confidence = prob_pathological if is_pathological else (1.0 - prob_pathological)
        
        # Tính toán Retinal Age Gap
        gap_html = ""
        if real_age > 0:
            gap = round(predicted_age - real_age, 1)
            gap_sign = f"+{gap}" if gap > 0 else str(gap)
            gap_interp = "Lão hóa sớm" if gap > 5 else ("Nhẹ" if gap > 2 else ("Trẻ hơn" if gap < -2 else "Bình thường"))
            gap_html = f"""<div class="metric-item">
<div class="metric-val">{gap_sign} năm</div>
<div class="metric-lbl">Retinal Age Gap ({gap_interp})</div>
</div>"""
            
        # Sử dụng chuỗi phẳng không lùi đầu dòng quá 4 dấu cách để tránh bị markdown nhận diện nhầm thành code block
        html_content = f"""<div class="diagnostic-card">
<div class="diagnostic-result-title">Kết quả phân tích chung</div>
<div class="diagnostic-result-value {diag_class_str}">{diag_text} ({diag_confidence:.1%})</div>
<div class="metric-grid">
<div class="metric-item" style="padding-left: 0;">
<div class="metric-val">{predicted_age:.1f} tuổi</div>
<div class="metric-lbl">Tuổi võng mạc (Retinal Age)</div>
</div>
{gap_html}
<div class="metric-item">
<div class="metric-val">{threshold:.2f}</div>
<div class="metric-lbl">Ngưỡng thiết lập</div>
</div>
<div class="metric-item">
<div class="metric-val">{"Mắt trái và phải" if (has_left and has_right) else ("Mắt trái" if has_left else "Mắt phải")}</div>
<div class="metric-lbl">Dữ liệu vào</div>
</div>
</div>
</div>"""

        st.markdown(html_content, unsafe_allow_html=True)
        
        # ── PHẦN 3: Grad-CAM song nhãn ──
        if show_gradcam:
            st.markdown('<div class="section-title">Grad-CAM - Bản Đồ Vùng Chú Ý Của Mô Hình</div>', unsafe_allow_html=True)
            st.markdown("Bản đồ nhiệt Grad-CAM chỉ ra vùng mô hình tập trung cao nhất để đưa ra quyết định chẩn đoán.")
            
            with st.spinner("Đang tính toán Grad-CAM song nhãn..."):
                l_input = left_steps[2] if left_steps is not None else None
                r_input = right_steps[2] if right_steps is not None else None
                
                left_overlay, right_overlay = compute_siamese_gradcam(
                    model=model,
                    left_img_bgr=l_input,
                    right_img_bgr=r_input,
                    left_missing=not has_left,
                    right_missing=not has_right,
                    device='cpu',
                    model_type=model_key,
                )
                
            gc_col_l, gc_col_r = st.columns(2)
            
            with gc_col_l:
                st.markdown("#### Grad-CAM Mắt trái")
                if has_left:
                    if left_overlay is not None:
                        st.image(cv2.cvtColor(left_overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
                    else:
                        st.text("Không tính được Grad-CAM cho mắt trái.")
                else:
                    st.text("Không có dữ liệu ảnh mắt trái.")
                    
            with gc_col_r:
                st.markdown("#### Grad-CAM Mắt phải")
                if has_right:
                    if right_overlay is not None:
                        st.image(cv2.cvtColor(right_overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
                    else:
                        st.text("Không tính được Grad-CAM cho mắt phải.")
                else:
                    st.text("Không có dữ liệu ảnh mắt phải.")
                    
    else:
        # Chế độ Demo
        st.markdown('<div class="section-title">Kết Quả Chẩn Đoán (Demo)</div>', unsafe_allow_html=True)
        st.info(
            "Thông báo: Ứng dụng đang chạy ở chế độ Demo. Hãy tải tệp best_model.pth vào thư mục "
            "results/exp_X/ để kích hoạt tính năng chẩn đoán y khoa bằng mô hình Siamese."
        )

else:
    # Giao diện mặc định khi chưa tải ảnh
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; background-color: #F8F9FA;
                border: 1px dashed #DEE2E6; border-radius: 4px; margin: 2rem 0;">
        <h3 style="color: #212529; margin-top: 0; font-weight: 500;">Tải ảnh đáy mắt để bắt đầu chẩn đoán</h3>
        <p style="color: #6C757D; margin-bottom: 0;">
            Vui lòng tải lên ảnh đáy mắt trái, mắt phải và thiết lập trạng thái thiếu mắt nếu có ở phía trên.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Thông Tin Chi Tiết Về Dự Án</div>', unsafe_allow_html=True)
    st.markdown("""
    Hệ thống chẩn đoán nhãn khoa sử dụng bộ dữ liệu ODIR-5K:
    - Bình thường (Normal): Bệnh nhân có cả hai mắt bình thường, không có tổn thương y khoa.
    - Bệnh lý (Pathological): Bệnh nhân mắc ít nhất 1 trong 7 bệnh lý đáy mắt chủ yếu (Tiểu đường, Tăng nhãn áp, Đục thủy tinh thể, Thoái hóa hoàng điểm, Cận thị bệnh lý, Tăng huyết áp, hoặc các bệnh đáy mắt khác).
    
    Ứng dụng áp dụng mạng Siamese song nhãn giúp trích xuất đặc trưng song song của cả hai mắt thông qua một mạng backbone chia sẻ trọng số, sau đó phân tích đặc trưng tích hợp để đưa ra kết luận chẩn đoán ở mức bệnh nhân và ước lượng tuổi võng mạc (Retinal Age).
    """)
