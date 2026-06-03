"""
ODIR-5K — Web App Chẩn Đoán Bệnh Lý Nhãn Khoa
Streamlit app: upload ảnh đáy mắt → tiền xử lý → chẩn đoán + Grad-CAM
"""
import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from inference import (
    preprocess_image, predict, load_model,
    LABELS, LABEL_ICONS, IMG_SIZE,
    compute_gradcam,
)

# ── Page Config ──
st.set_page_config(
    page_title="ODIR-5K — Chẩn Đoán Nhãn Khoa",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Header gradient */
    .header-gradient {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .header-gradient h1 {
        color: #fff;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2rem;
        margin: 0;
    }
    .header-gradient p {
        color: rgba(255,255,255,0.7);
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        margin: 0.5rem 0 0 0;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4fc3f7;
        font-family: 'Inter', sans-serif;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
        margin-top: 0.3rem;
    }

    /* Disease bar */
    .disease-bar {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        border-left: 4px solid;
        transition: transform 0.2s;
    }
    .disease-bar:hover {
        transform: translateX(4px);
    }
    .disease-bar .name {
        font-weight: 600;
        min-width: 120px;
        color: #e0e0e0;
    }
    .disease-bar .prob {
        font-weight: 700;
        min-width: 60px;
        text-align: right;
    }

    /* Step labels */
    .step-label {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }

    /* Upload area */
    .stFileUploader > div {
        border-radius: 12px;
    }

    /* Sidebar */
    .sidebar .block-container {
        padding-top: 1rem;
    }

    /* Status badge */
    .status-badge {
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-ready {
        background: rgba(46, 204, 113, 0.15);
        color: #2ecc71;
        border: 1px solid rgba(46, 204, 113, 0.3);
    }
    .status-demo {
        background: rgba(241, 196, 15, 0.15);
        color: #f1c40f;
        border: 1px solid rgba(241, 196, 15, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──
with st.sidebar:
    st.markdown("## ⚙️ Cài đặt")

    # Model selection
    model_type = st.selectbox(
        "Kiến trúc mô hình",
        ["EfficientNet-B0 (CNN)", "Swin Transformer"],
        index=0,
    )
    model_key = 'cnn' if 'CNN' in model_type else 'swin'

    # Model weights
    weights_dir = Path(__file__).parent.parent / "results"
    available_weights = list(weights_dir.glob("*/best.pth")) if weights_dir.exists() else []

    if available_weights:
        selected_weight = st.selectbox(
            "Model weights",
            available_weights,
            format_func=lambda p: p.parent.name,
        )
        model_loaded = True
    else:
        st.warning("⚠️ Chưa có file `best.pth`. Đang ở chế độ Demo (chỉ hiện tiền xử lý).")
        st.info("Sau khi train xong trên Kaggle, copy file `best.pth` vào `results/exp_X/`")
        selected_weight = None
        model_loaded = False

    st.markdown("---")

    # Threshold
    threshold = st.slider("Ngưỡng chẩn đoán", 0.1, 0.9, 0.5, 0.05)

    st.markdown("---")
    st.markdown("### 🧬 Retinal Age Gap")
    real_age = st.number_input(
        "Tuổi thật của bệnh nhân",
        min_value=0, max_value=120, value=0, step=1,
        help="Nhập 0 nếu không biết tuổi thật. Retinal Age Gap = Tuổi sinh học − Tuổi thật."
    )

    st.markdown("---")
    st.markdown("### 🔥 Grad-CAM")
    show_gradcam = st.toggle("Hiển thị Grad-CAM heatmap", value=True)
    if show_gradcam:
        gradcam_label = st.selectbox(
            "Nhãn Grad-CAM",
            ["(Tự động — nhãn cao nhất)"] + LABELS,
        )

    st.markdown("---")
    st.markdown("### 📊 Thông tin mô hình")
    st.markdown(f"- **Kiến trúc**: {model_type}")
    st.markdown(f"- **Input size**: {IMG_SIZE}×{IMG_SIZE}")
    st.markdown(f"- **Tasks**: 8 bệnh + dự đoán tuổi")
    st.markdown(f"- **Ngưỡng**: {threshold}")

    st.markdown("---")
    st.markdown(
        "<p style='color: rgba(255,255,255,0.4); font-size: 0.75rem;'>"
        "ODIR-5K Multi-task Learning<br>"
        "Ngô Đình Đạt — 2251161965<br>"
        "GVHD: TS. Lê Thị Tú Kiên</p>",
        unsafe_allow_html=True,
    )


# ── Header ──
st.markdown("""
<div class="header-gradient">
    <h1>👁️ Hệ Thống Chẩn Đoán Bệnh Lý Nhãn Khoa</h1>
    <p>Phân tích ảnh đáy mắt bằng AI — Multi-task Deep Learning (EfficientNet-B0 / Swin Transformer)</p>
</div>
""", unsafe_allow_html=True)

# Status
if model_loaded:
    st.markdown('<span class="status-badge status-ready">🟢 Model sẵn sàng</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="status-badge status-demo">🟡 Chế độ Demo — Chỉ hiện tiền xử lý</span>', unsafe_allow_html=True)


# ── Upload ──
uploaded = st.file_uploader(
    "📤 Upload ảnh đáy mắt (fundus image)",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Chấp nhận ảnh fundus dạng JPG/PNG. Khuyến nghị kích thước gốc từ thiết bị chụp."
)

if uploaded:
    # Đọc ảnh
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("❌ Không đọc được ảnh. Vui lòng thử file khác.")
        st.stop()

    # ── PHẦN 1: Tiền xử lý ──
    st.markdown("## 🔬 Quá Trình Tiền Xử Lý")

    step1_roi, step2_bg, step3_enh = preprocess_image(img_bgr)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<span class="step-label">BƯỚC 0</span>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", width="stretch")
        st.caption(f"Kích thước: {img_bgr.shape[1]}×{img_bgr.shape[0]}")

    with col2:
        st.markdown('<span class="step-label">BƯỚC 1 — ROI Crop</span>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(step1_roi, cv2.COLOR_BGR2RGB), caption="ROI Crop 512×512", width="stretch")
        st.caption("Loại bỏ viền đen, crop vùng võng mạc")

    with col3:
        st.markdown('<span class="step-label">BƯỚC 2 — Ben Graham</span>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(step2_bg, cv2.COLOR_BGR2RGB), caption="Ben Graham Normalization", width="stretch")
        st.caption("Chuẩn hóa ánh sáng không đều")

    with col4:
        st.markdown('<span class="step-label">BƯỚC 3 — CLAHE</span>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(step3_enh, cv2.COLOR_BGR2RGB), caption="CLAHE Enhanced", width="stretch")
        st.caption("Tăng cường tương phản cục bộ")

    st.markdown("---")

    # ── PHẦN 2: Chẩn đoán ──
    if model_loaded and selected_weight:
        st.markdown("## 🩺 Kết Quả Chẩn Đoán")

        # Load model (cached)
        @st.cache_resource
        def get_model(path, mtype):
            return load_model(str(path), model_type=mtype, device='cpu')

        with st.spinner("🔄 Đang phân tích..."):
            model = get_model(selected_weight, model_key)
            result = predict(model, step3_enh, device='cpu')

        probs = result['probabilities']
        age   = result['predicted_age']

        # Retinal Age Gap
        gap = round(age - real_age, 1) if real_age > 0 else None

        # Metrics cards
        detected = [l for l, p in probs.items() if p >= threshold]
        top_disease = max(probs, key=probs.get)

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{len(detected)}</div>
                <div class="label">Bệnh phát hiện (≥{threshold:.0%})</div>
            </div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{age} tuổi</div>
                <div class="label">Tuổi sinh học (Retinal Age)</div>
            </div>""", unsafe_allow_html=True)
        with mc3:
            if gap is not None:
                gap_color = "#e74c3c" if gap > 5 else ("#f39c12" if gap > 2 else "#2ecc71")
                gap_sign  = f"+{gap}" if gap > 0 else str(gap)
                gap_interp = "Lão hóa sớm ⚠️" if gap > 5 else ("Nhẹ ↑" if gap > 2 else ("Âm — trẻ hơn ✅" if gap < -2 else "Bình thường ✅"))
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value" style="color:{gap_color}">{gap_sign} năm</div>
                    <div class="label">Retinal Age Gap — {gap_interp}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <div class="value" style="color:rgba(255,255,255,0.3)">—</div>
                    <div class="label">Retinal Age Gap (nhập tuổi thật ở sidebar)</div>
                </div>""", unsafe_allow_html=True)
        with mc4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{probs[top_disease]:.1%}</div>
                <div class="label">Xác suất cao nhất ({top_disease})</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Disease bars
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown("### 📋 Chi tiết từng bệnh lý")
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

            for i, (label, prob) in enumerate(sorted_probs):
                icon = LABEL_ICONS[LABELS.index(label)]
                if prob >= threshold:
                    color = "#e74c3c" if prob >= 0.7 else "#f39c12"
                    status = "⚠️ PHÁT HIỆN"
                else:
                    color = "#2ecc71" if prob < 0.2 else "#95a5a6"
                    status = "✅ Bình thường"

                bar_width = max(prob * 100, 2)
                st.markdown(f"""
                <div class="disease-bar" style="border-left-color: {color};">
                    <span style="font-size: 1.2rem;">{icon}</span>
                    <span class="name">{label}</span>
                    <div style="flex:1; background: rgba(255,255,255,0.08); border-radius: 4px; height: 8px;">
                        <div style="width: {bar_width}%; background: {color}; height: 100%; border-radius: 4px;"></div>
                    </div>
                    <span class="prob" style="color: {color};">{prob:.1%}</span>
                    <span style="font-size: 0.75rem; color: rgba(255,255,255,0.5);">{status}</span>
                </div>
                """, unsafe_allow_html=True)

        with col_right:
            st.markdown("### 🎯 Tóm tắt")
            if detected:
                st.error(f"⚠️ Phát hiện {len(detected)} bệnh lý:")
                for d in detected:
                    icon = LABEL_ICONS[LABELS.index(d)]
                    st.markdown(f"- {icon} **{d}** ({probs[d]:.1%})")
            else:
                st.success("✅ Không phát hiện bệnh lý nào vượt ngưỡng.")

            st.markdown("### 🧬 Tuổi sinh học")
            if gap is not None:
                st.metric("Retinal Age", f"{age} tuổi",
                          delta=f"{'+' if gap>0 else ''}{gap} so với tuổi thật",
                          delta_color="inverse" if gap > 0 else "normal")
            else:
                st.metric("Retinal Age", f"{age} tuổi")

        # ── PHẦN 2b: Grad-CAM ──
        if show_gradcam:
            st.markdown("---")
            st.markdown("## 🔥 Grad-CAM — Vùng Quan Trọng Mô Hình Chú Ý")
            st.caption(
                "Heatmap màu đỏ = vùng ảnh ảnh hưởng nhiều nhất đến quyết định của mô hình. "
                "Xanh lam = ít ảnh hưởng."
            )

            # Chọn target label
            target_idx = None
            if gradcam_label != "(Tự động — nhãn cao nhất)":
                target_idx = LABELS.index(gradcam_label)
                cam_title  = f"Grad-CAM cho nhãn: {gradcam_label}"
            else:
                target_idx = None
                top_auto   = max(probs, key=probs.get)
                cam_title  = f"Grad-CAM tự động — nhãn: {top_auto} ({probs[top_auto]:.1%})"

            with st.spinner("🔄 Đang tính Grad-CAM..."):
                try:
                    heatmap_bgr, cam_raw = compute_gradcam(
                        model, step3_enh,
                        target_label_idx=target_idx,
                        device='cpu',
                        model_type=model_key,
                    )
                    gc1, gc2, gc3 = st.columns(3)
                    with gc1:
                        st.markdown('<span class="step-label">ẢNH ĐÃ XỬ LÝ</span>',
                                    unsafe_allow_html=True)
                        st.image(cv2.cvtColor(step3_enh, cv2.COLOR_BGR2RGB),
                                 caption="Đầu vào mô hình (CLAHE)",
                                 width="stretch")
                    with gc2:
                        st.markdown(f'<span class="step-label">GRAD-CAM</span>',
                                    unsafe_allow_html=True)
                        st.image(cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB),
                                 caption=cam_title,
                                 width="stretch")
                    with gc3:
                        st.markdown('<span class="step-label">RAW CAM (grayscale)</span>',
                                    unsafe_allow_html=True)
                        import numpy as np
                        cam_display = (cam_raw * 255).astype(np.uint8)
                        st.image(cam_display, caption="Activation map (0=thấp, 255=cao)",
                                 width="stretch")
                    st.caption(
                        "💡 **Cách đọc:** Vùng đỏ/cam là nơi mô hình tập trung khi dự đoán nhãn đã chọn. "
                        "Trong ảnh đáy mắt, thường tương ứng với gai thị, mạch máu, hoặc vùng tổn thương."
                    )
                except Exception as e:
                    st.warning(f"⚠️ Không thể tính Grad-CAM: {e}")

    else:
        # Demo mode
        st.markdown("## 🩺 Kết Quả Chẩn Đoán")
        st.info(
            "🔒 **Chế độ Demo** — Chưa có model weights.\n\n"
            "Sau khi train xong trên Kaggle:\n"
            "1. Download `best.pth` từ tab Output\n"
            "2. Copy vào `results/exp_3_cnn_preprocess_with_aug/best.pth`\n"
            "3. Restart app: `streamlit run webapp/app.py`"
        )

    # ── PHẦN 3: Thông tin kỹ thuật ──
    with st.expander("📐 Chi tiết kỹ thuật"):
        st.markdown(f"""
        | Thông số | Giá trị |
        |----------|---------|
        | Ảnh đầu vào | {img_bgr.shape[1]}×{img_bgr.shape[0]} px |
        | Sau ROI Crop | 512×512 px |
        | Input model | {IMG_SIZE}×{IMG_SIZE} px |
        | Normalization | ImageNet (mean=[0.485, 0.456, 0.406]) |
        | Tiền xử lý | ROI Crop → Ben Graham → CLAHE |
        | Multi-task | 8 bệnh (BCE) + Tuổi (SmoothL1) |
        """)

else:
    # Placeholder khi chưa upload
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; background: rgba(255,255,255,0.03);
                border: 2px dashed rgba(255,255,255,0.1); border-radius: 16px; margin: 2rem 0;">
        <p style="font-size: 3rem; margin: 0;">👁️</p>
        <h3 style="color: rgba(255,255,255,0.7); margin: 1rem 0 0.5rem;">Upload ảnh đáy mắt để bắt đầu</h3>
        <p style="color: rgba(255,255,255,0.4);">
            Hỗ trợ JPG, PNG, BMP — Ảnh fundus từ thiết bị chụp đáy mắt
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sample info
    st.markdown("### 🏥 8 Bệnh Lý Được Phát Hiện")
    cols = st.columns(4)
    diseases = [
        ("👁️ Normal", "Mắt bình thường"),
        ("🩸 Diabetes", "Bệnh võng mạc tiểu đường"),
        ("🟢 Glaucoma", "Bệnh tăng nhãn áp"),
        ("🔵 Cataract", "Đục thể thủy tinh"),
        ("🟡 AMD", "Thoái hóa hoàng điểm"),
        ("❤️‍🩹 Hypertension", "Bệnh võng mạc tăng huyết áp"),
        ("👓 Myopia", "Cận thị bệnh lý"),
        ("📋 Other", "Bệnh lý khác"),
    ]
    for i, (name, desc) in enumerate(diseases):
        with cols[i % 4]:
            st.markdown(f"**{name}**")
            st.caption(desc)
