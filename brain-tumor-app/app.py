import streamlit as st
import numpy as np
import cv2, json
from pathlib import Path
from PIL import Image
import onnxruntime as ort
import matplotlib.pyplot as plt

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="TinyBrainNet-MT | Clinical Diagnostic Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CUSTOM CSS STYLING -----------------
st.markdown("""
<style>
    /* Main Background & Font Styling */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Header Banner */
    .header-box {
        background: linear-gradient(135deg, #161b22 0%, #1f2937 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .header-title {
        color: #58a6ff;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-subtitle {
        color: #8b949e;
        font-size: 14px;
        margin: 0;
    }
    
    /* Metric Card Badges */
    .badge-container {
        display: flex;
        gap: 10px;
        margin-top: 12px;
        flex-wrap: wrap;
    }
    .badge {
        background-color: #21262d;
        border: 1px solid #388bfd33;
        color: #79c0ff;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    /* Result Diagnostic Cards */
    .diag-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 16px;
    }
    .diag-header {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #8b949e;
        margin-bottom: 8px;
        font-weight: 600;
    }
    .diag-value {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    
    /* Custom Sidebar adjustments */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- PATHS & MODEL LOADING -----------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "tinybrainnet.onnx"
if not MODEL_PATH.exists():
    MODEL_PATH = Path("model/tinybrainnet.onnx")

META_PATH = BASE_DIR / "model" / "meta.json"
if not META_PATH.exists():
    META_PATH = Path("model/meta.json")

SAMPLES_DIR = BASE_DIR / "samples"
if not SAMPLES_DIR.exists():
    SAMPLES_DIR = Path("samples")

@st.cache_resource
def load_session():
    if not MODEL_PATH.exists():
        st.error(f"Model binary not found at: {MODEL_PATH}")
        return None, {}
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(MODEL_PATH), so, providers=["CPUExecutionProvider"])
    return sess, meta

sess, meta = load_session()

CLASS_NAMES = meta.get("classes", ["Glioma", "Meningioma", "Pituitary", "No_tumor"])
MEAN = np.array(meta.get("input", {}).get("mean", [0.1838, 0.1838, 0.1838]), dtype=np.float32)
STD = np.array(meta.get("input", {}).get("std", [0.1984, 0.1984, 0.1984]), dtype=np.float32)
IMG_SIZE = int(meta.get("input", {}).get("size", 224))
T_OPT = float(meta.get("calibration", {}).get("temperature", 0.693))
TRIAGE_AUTO = float(meta.get("triage", {}).get("auto", 0.62))
TRIAGE_REV = float(meta.get("triage", {}).get("review", 0.50))

# ----------------- TOP BANNER -----------------
st.markdown("""
<div class="header-box">
    <div class="header-title">🧠 TinyBrainNet-MT: Clinical Decision Support Suite</div>
    <p class="header-subtitle">Edge-Optimized Multi-Task Framework for Unified Brain MRI Classification, Segmentation & Saliency Attribution</p>
    <div class="badge-container">
        <span class="badge">Model: UNI-NET (<0.5M Params)</span>
        <span class="badge">Runtime: ONNX CPU</span>
        <span class="badge">In-Domain Acc: ~98.4%</span>
        <span class="badge">Mean Dice: 0.823</span>
        <span class="badge">Calibrated Temp: T=0.693</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.markdown("### 🎛️ Input MRI Control")
    samples = sorted(list(SAMPLES_DIR.glob("*.jpg")) + list(SAMPLES_DIR.glob("*.png")))
    sample_options = ["-- Upload Scan File --"] + [p.name for p in samples]
    selected_sample = st.selectbox("Select Benchmark Case:", sample_options)
    
    img_pil = None
    if selected_sample != "-- Upload Scan File --":
        img_pil = Image.open(SAMPLES_DIR / selected_sample).convert("RGB")
        st.info(f"Loaded: `{selected_sample}`")
    else:
        uploaded_file = st.file_uploader("Upload MRI Scan (JPG / PNG):", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            img_pil = Image.open(uploaded_file).convert("RGB")

    st.markdown("---")
    st.markdown("### ⚙️ Clinical Triage Policy")
    st.caption("Validation-calibrated risk-coverage thresholds:")
    st.write(f"• **Auto-Qualified:** $\ge {TRIAGE_AUTO*100:.0f}\\%$")
    st.write(f"• **Secondary Review:** ${TRIAGE_REV*100:.0f}\\% - {TRIAGE_AUTO*100:.0f}\\%$")
    st.write(f"• **Mandatory Review:** $< {TRIAGE_REV*100:.0f}\\%$")
    st.markdown("---")
    st.caption("TinyBrainNet-MT Research Prototype | Khulna & Sirajganj Clinical Validation")

# ----------------- MAIN WORKSPACE -----------------
if img_pil and sess:
    # Preprocessing
    im = np.array(img_pil)
    im_resized = cv2.resize(im, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = ((im_resized - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)

    # ONNX Multi-Task Inference
    out = sess.run(None, {"input": x})
    cls_logits, seg_logits, cam = out[0][0], out[1][0, 0], out[2][0]

    # Temperature Scaled Classification Probabilities
    logits_scaled = cls_logits / T_OPT
    exp_l = np.exp(logits_scaled - np.max(logits_scaled))
    probs = exp_l / exp_l.sum()
    pred_idx = int(np.argmax(probs))
    pred_cls = CLASS_NAMES[pred_idx]
    conf = float(probs[pred_idx])

    # Classifier Gated Segmentation & Attention Heatmap
    pred_mask = (1.0 / (1.0 + np.exp(-seg_logits))) > 0.5
    cam_act = np.maximum(cam[pred_idx], 0)
    cam_res = cv2.resize(cam_act, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    cam_norm = np.clip(cam_res / (np.percentile(cam_res, 99.5) + 1e-8), 0, 1)

    # Layout: Diagnostic Summary & Probability Metrics
    col_pred, col_metrics = st.columns([1.2, 1])

    with col_pred:
        # Determine Status Alert Theme
        if conf >= TRIAGE_AUTO:
            card_border = "#238636"
            status_text = "✅ Automated Decision Qualified (Normal Tier)"
            status_color = "#3fb950"
        elif conf >= TRIAGE_REV:
            card_border = "#d29922"
            status_text = "⚠️ Secondary Radiologist Review Recommended"
            status_color = "#e3b341"
        else:
            card_border = "#da3633"
            status_text = "🛑 Inconclusive / Mandatory Clinical Audit"
            status_color = "#f85149"

        st.markdown(f"""
        <div class="diag-card" style="border-left: 5px solid {card_border};">
            <div class="diag-header">Primary Differential Diagnosis</div>
            <div class="diag-value" style="color: {status_color};">{pred_cls}</div>
            <div style="font-size: 14px; font-weight: 600; margin-bottom: 8px;">Confidence: {conf*100:.2f}%</div>
            <div style="font-size: 13px; color: #8b949e;">{status_text}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_metrics:
        st.markdown('<div class="diag-card"><div class="diag-header">Differential Distribution</div>', unsafe_allow_html=True)
        for cn, p in zip(CLASS_NAMES, probs):
            st.progress(float(p), text=f"{cn}: {p*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    # Multi-Panel Visual Localization
    st.markdown("### 🔬 Multi-Modal Diagnostic Localization")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), facecolor="#0d1117")

    # Panel 1: Original MRI
    axes[0].imshow(im_resized)
    axes[0].set_title("Input Slice", color="#c9d1d9", fontsize=11, fontweight="bold", pad=8)
    axes[0].axis("off")

    # Panel 2: Classifier-Gated Segmentation
    axes[1].imshow(im_resized)
    if pred_cls.lower() != "no_tumor" and pred_mask.any():
        axes[1].contour(pred_mask.astype(float), levels=[0.5], colors="#f85149", linewidths=2.0)
        axes[1].imshow(np.ma.masked_where(~pred_mask, pred_mask), cmap="autumn", alpha=0.45)
        axes[1].set_title(f"Tumor Mask (Area: {pred_mask.mean()*100:.1f}%)", color="#f85149", fontsize=11, fontweight="bold", pad=8)
    else:
        axes[1].set_title("Gated: No Tumor Boundary", color="#3fb950", fontsize=11, fontweight="bold", pad=8)
    axes[1].axis("off")

    # Panel 3: Grad-CAM++ Visual Attention
    axes[2].imshow(im_resized)
    axes[2].imshow(plt.cm.jet(cam_norm), alpha=0.45)
    if pred_cls.lower() != "no_tumor" and pred_mask.any():
        axes[2].contour(pred_mask.astype(float), levels=[0.5], colors="white", linewidths=1.2)
    axes[2].set_title("Grad-CAM++ Attribution", color="#79c0ff", fontsize=11, fontweight="bold", pad=8)
    axes[2].axis("off")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

else:
    st.info("👈 Please select a benchmark test scan from the sidebar or upload a new Brain MRI image.")
