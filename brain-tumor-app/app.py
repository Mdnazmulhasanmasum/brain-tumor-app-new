import streamlit as st
import numpy as np
import cv2, json
from pathlib import Path
from PIL import Image
import onnxruntime as ort
import matplotlib.pyplot as plt

st.set_page_config(page_title="TinyBrainNet-MT Diagnostic Suite", layout="wide")

# ডাইনামিক পাথ কনফিগারেশন (রুট এবং সাব-ফোল্ডার উভয় পরিবেশেই কাজ করবে)
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
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model binary not found at: {MODEL_PATH}")
        return None, {}
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(MODEL_PATH), so, providers=["CPUExecutionProvider"])
    return sess, meta

sess, meta = load_model()

CLASS_NAMES = meta.get("classes", ["Glioma", "Meningioma", "Pituitary", "No_tumor"])
MEAN = np.array(meta.get("input", {}).get("mean", [0.1838, 0.1838, 0.1838]), dtype=np.float32)
STD = np.array(meta.get("input", {}).get("std", [0.1984, 0.1984, 0.1984]), dtype=np.float32)
IMG_SIZE = int(meta.get("input", {}).get("size", 224))
T_OPT = float(meta.get("calibration", {}).get("temperature", 0.693))
TRIAGE_AUTO = float(meta.get("triage", {}).get("auto", 0.62))
TRIAGE_REV = float(meta.get("triage", {}).get("review", 0.50))

st.title("🧠 TinyBrainNet-MT: Unified Brain MRI Analysis")
st.markdown("**Edge-Ready Multi-Task Model (<0.5M Parameters)** | Simultaneous Classification, Segmentation & Attention")

c1, c2 = st.columns([1, 1.3])

with c1:
    st.subheader("1. Input MRI Slice")
    samples = list(SAMPLES_DIR.glob("*.jpg")) + list(SAMPLES_DIR.glob("*.png"))
    names = ["-- Upload Scan --"] + [p.name for p in samples]
    sel = st.selectbox("Select a test sample or upload an image:", names)
    
    img_pil = None
    if sel != "-- Upload Scan --":
        img_pil = Image.open(SAMPLES_DIR / sel).convert("RGB")
    else:
        up = st.file_uploader("Upload Brain MRI (JPG / PNG)", type=["jpg", "png", "jpeg"])
        if up:
            img_pil = Image.open(up).convert("RGB")
            
    if img_pil:
        st.image(img_pil, caption="Input MRI Scan", use_container_width=True)

if img_pil and sess:
    im = np.array(img_pil)
    im_resized = cv2.resize(im, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = ((im_resized - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)

    out = sess.run(None, {"input": x})
    cls_logits, seg_logits, cam = out[0][0], out[1][0, 0], out[2][0]

    logits_scaled = cls_logits / T_OPT
    exp_l = np.exp(logits_scaled - np.max(logits_scaled))
    probs = exp_l / exp_l.sum()
    pred_idx = int(np.argmax(probs))
    pred_cls = CLASS_NAMES[pred_idx]
    conf = float(probs[pred_idx])

    pred_mask = (1.0 / (1.0 + np.exp(-seg_logits))) > 0.5
    cam_act = np.maximum(cam[pred_idx], 0)
    cam_res = cv2.resize(cam_act, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    cam_norm = np.clip(cam_res / (np.percentile(cam_res, 99.5) + 1e-8), 0, 1)

    with c2:
        st.subheader("2. Prediction & Clinical Triage")
        if conf >= TRIAGE_AUTO:
            st.success(f"**Diagnosis: {pred_cls}** ({conf*100:.1f}% confidence) — ✅ Automated Decision Qualified")
        elif conf >= TRIAGE_REV:
            st.warning(f"**Diagnosis: {pred_cls}** ({conf*100:.1f}% confidence) — ⚠️ Radiologist Review Recommended")
        else:
            st.error(f"**Inconclusive ({pred_cls}: {conf*100:.1f}%)** — 🛑 Mandatory Specialist Review")

        st.write("Class Probabilities:")
        for cn, p in zip(CLASS_NAMES, probs):
            st.progress(float(p), text=f"{cn}: {p*100:.1f}%")

        st.subheader("3. Tumor Boundary & Heatmap")
        fig, ax = plt.subplots(1, 3, figsize=(10, 3.5))
        ax[0].imshow(im_resized); ax[0].set_title("Input"); ax[0].axis("off")

        ax[1].imshow(im_resized)
        if pred_cls.lower() != "no_tumor" and pred_mask.any():
            ax[1].contour(pred_mask.astype(float), levels=[0.5], colors="red", linewidths=2)
            ax[1].imshow(np.ma.masked_where(~pred_mask, pred_mask), cmap="autumn", alpha=0.35)
            ax[1].set_title("Tumor Mask (Red)")
        else:
            ax[1].set_title("No Tumor Detected")
        ax[1].axis("off")

        ax[2].imshow(im_resized); ax[2].imshow(plt.cm.jet(cam_norm), alpha=0.45)
        if pred_cls.lower() != "no_tumor" and pred_mask.any():
            ax[2].contour(pred_mask.astype(float), levels=[0.5], colors="white", linewidths=1.2)
        ax[2].set_title("Grad-CAM++")
        ax[2].axis("off")

        st.pyplot(fig)
        plt.close(fig)
