import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import streamlit as st

sys.path.insert(0, os.path.abspath("."))
from schemas.payload import TABULAR_FEATURE_NAMES, SEVERITY_CLASSES
from models.multi_task import MultiTaskModel
from xai.shap_explainer import FastSHAPExplainer
from xai.narrative_engine import ClinicalNarrativeEngine
from schemas.payload import ClassificationOutput, RegressionOutput

# Page Configuration - Minimalist Dark/Light Research Dashboard
st.set_page_config(
    page_title="DCMF-Net Model Research Interface",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Minimal CSS
st.markdown("""
<style>
    .main { background-color: #09090b; color: #fafafa; }
    .stMetric { background-color: #18181b; border: 1px solid #27272a; padding: 12px; border-radius: 8px; }
    .css-1r6g786 { font-family: monospace; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_preprocessor():
    project_root = os.path.abspath(".")
    artifact_dir = os.path.join(project_root, "artifacts")
    prep_path = os.path.join(artifact_dir, "preprocessor.joblib")
    fp32_path = os.path.join(artifact_dir, "model_state.pt")

    preprocessor = joblib.load(prep_path) if os.path.exists(prep_path) else None

    base_model = MultiTaskModel()
    if os.path.exists(fp32_path):
        checkpoint = torch.load(fp32_path, map_location="cpu", weights_only=False)
        base_model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "N/A")
        val_loss = checkpoint.get("val_loss", "N/A")
    else:
        epoch, val_loss = "N/A", "N/A"

    base_model.eval()
    explainer = FastSHAPExplainer()
    narrative_engine = ClinicalNarrativeEngine()

    return base_model, preprocessor, explainer, narrative_engine, epoch, val_loss

# Load PyTorch Model & Preprocessor
model, preprocessor, explainer, narrative_engine, epoch, val_loss = load_model_and_preprocessor()

# Header Title
st.title("🧠 DCMF-Net Model Research & Evaluation Workspace")
st.caption("Direct PyTorch Model Inference, FastSHAP Attributions, and Multi-Task Clinical Evaluation")

# System Status Info
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    st.metric("Model Architecture", "DCMF-Net Transformer")
with col_s2:
    st.metric("Loaded Checkpoint", f"Epoch {epoch}")
with col_s3:
    st.metric("Validation Loss", f"{val_loss:.4f}" if isinstance(val_loss, float) else str(val_loss))
with col_s4:
    st.metric("Preprocessor", "RobustScaler (Active)" if preprocessor else "Raw Scale")

st.divider()

# Sidebar: Preset Selectors & Controls
st.sidebar.header("🔬 Scenario Presets")
preset = st.sidebar.selectbox(
    "Load Clinical Benchmark Preset:",
    [
        "Custom Inputs",
        "Optimal Healthy Baseline",
        "Mild Work Stress & Fatigue",
        "Moderate Depressive Affect",
        "Severe Crisis & Agitation",
    ]
)

# Preset Values Dictionary
PRESETS = {
    "Optimal Healthy Baseline": [4.5, 4.5, 110.0, 65.0, 4.0, 10.0, 0.65, 15.0, 0.75, 0.35, 0.2, 1.8, 195.0, 3.5, 68.0, 75.0, 36.6, 0.8],
    "Mild Work Stress & Fatigue": [3.0, 3.2, 250.0, 50.0, 8.0, 25.0, 0.40, 22.0, 0.45, 0.25, -0.1, 1.1, 175.0, 2.8, 78.0, 52.0, 36.5, 2.4],
    "Moderate Depressive Affect": [2.0, 2.0, 420.0, 35.0, 12.0, 55.0, 0.20, 28.0, 0.20, 0.15, -0.8, 0.5, 150.0, 2.1, 88.0, 32.0, 36.2, 4.8],
    "Severe Crisis & Agitation": [1.0, 1.0, 580.0, 20.0, 18.0, 90.0, 0.05, 42.0, 0.05, 0.05, -1.6, 0.2, 120.0, 1.4, 115.0, 16.0, 35.8, 8.5],
}

default_vals = PRESETS.get(preset, [3.5, 4.0, 150.0, 55.0, 6.0, 15.0, 0.45, 18.0, 0.65, 0.25, 0.15, 1.2, 180.0, 3.2, 72.0, 65.0, 36.6, 1.4])

# Main Layout: 2 Columns (Left: 18 Feature Inputs, Right: PyTorch Model Output)
left_col, right_col = st.columns([7, 5])

with left_col:
    st.subheader("📋 18 Biomarker Input Sliders")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Behavioral & Lifestyle",
        "Visual Kinematics",
        "Acoustic & Speech",
        "Physiological Biomarkers",
    ])

    feature_inputs = {}

    with tab1:
        feature_inputs["Sleep_Quality"] = st.slider("Sleep Quality (1-5 score)", 1.0, 5.0, float(default_vals[0]), 0.1)
        feature_inputs["Social_Engagement"] = st.slider("Social Engagement (1-5 score)", 1.0, 5.0, float(default_vals[1]), 0.1)
        feature_inputs["Daily_App_Usage_Min"] = st.slider("Daily Screen Time (minutes)", 0.0, 720.0, float(default_vals[2]), 5.0)
        feature_inputs["Typing_Speed_WPM"] = st.slider("Typing Speed (WPM)", 0.0, 150.0, float(default_vals[3]), 1.0)
        feature_inputs["Session_Frequency"] = st.slider("Daily Session Frequency", 0.0, 30.0, float(default_vals[4]), 1.0)
        feature_inputs["Idle_Time_Min"] = st.slider("Screen Idle Duration (minutes)", 0.0, 180.0, float(default_vals[5]), 1.0)

    with tab2:
        feature_inputs["Facial_Emotion_Variance"] = st.slider("Facial Emotion Variance (0-1)", 0.0, 1.0, float(default_vals[6]), 0.01)
        feature_inputs["Eye_Blink_Rate"] = st.slider("Eye Blink Frequency (blinks/min)", 0.0, 80.0, float(default_vals[7]), 1.0)
        feature_inputs["Smile_Intensity"] = st.slider("Smile Intensity AU12 (0-1)", 0.0, 1.0, float(default_vals[8]), 0.01)
        feature_inputs["Head_Motion_Index"] = st.slider("Head Motion Index (0-1)", 0.0, 1.0, float(default_vals[9]), 0.01)

    with tab3:
        feature_inputs["MFCC_Mean"] = st.slider("MFCC Mean Coefficient", -5.0, 5.0, float(default_vals[10]), 0.05)
        feature_inputs["MFCC_Variance"] = st.slider("MFCC Variance", 0.0, 10.0, float(default_vals[11]), 0.1)
        feature_inputs["Pitch_Mean"] = st.slider("Fundamental Frequency F0 (Hz)", 40.0, 350.0, float(default_vals[12]), 1.0)
        feature_inputs["Speech_Rate"] = st.slider("Speech Rate (words/sec)", 0.0, 10.0, float(default_vals[13]), 0.1)

    with tab4:
        feature_inputs["Heart_Rate_BPM"] = st.slider("Resting Heart Rate (BPM)", 40.0, 180.0, float(default_vals[14]), 1.0)
        feature_inputs["HRV_Index"] = st.slider("Heart Rate Variability HRV (ms)", 5.0, 150.0, float(default_vals[15]), 1.0)
        feature_inputs["Skin_Temperature"] = st.slider("Peripheral Skin Temp (°C)", 30.0, 42.0, float(default_vals[16]), 0.1)
        feature_inputs["GSR_Level"] = st.slider("Galvanic Skin Response GSR (μS)", 0.0, 20.0, float(default_vals[17]), 0.1)

# Extract raw list of 18 features in order
raw_tabular = [feature_inputs[name] for name in TABULAR_FEATURE_NAMES]

# Process through Preprocessor & PyTorch Model
if preprocessor:
    scaled_tabular = preprocessor.transform([raw_tabular])[0]
else:
    scaled_tabular = np.array(raw_tabular, dtype=np.float32)

v_tensor = torch.zeros(1, 128)
a_tensor = torch.zeros(1, 256)
t_tensor = torch.from_numpy(np.array([scaled_tabular], dtype=np.float32))

t0 = time.perf_counter()
with torch.no_grad():
    logits, reg_scores, attn_dict = model(v_tensor, a_tensor, t_tensor)
    probs = F.softmax(logits, dim=-1)[0].numpy().tolist()
    pred_cls_id = int(torch.argmax(logits, dim=-1)[0].item())
    pred_cls_name = SEVERITY_CLASSES[pred_cls_id]
    reg_vals = reg_scores[0].numpy().tolist()
latency_ms = (time.perf_counter() - t0) * 1000.0

prob_dict = {cls_name: float(round(p, 4)) for cls_name, p in zip(SEVERITY_CLASSES, probs)}

# FastSHAP Attributions
attributions_list, attributions_dict, shap_latency = explainer.explain(model, v_tensor, a_tensor, scaled_tabular)

# Narrative Generation
cls_out = ClassificationOutput(predicted_class=pred_cls_name, predicted_class_id=pred_cls_id, probabilities=prob_dict)
reg_out = RegressionOutput(depression_score=round(reg_vals[0], 2), anxiety_score=round(reg_vals[1], 2), stress_score=round(reg_vals[2], 2))
narrative_payload = narrative_engine.generate_narrative(cls_out, reg_out, attributions_list, attn_dict)

# Right Column: PyTorch Model Output Results
with right_col:
    st.subheader("⚡ PyTorch Model Inference Results")
    st.caption(f"Forward Pass Latency: {latency_ms:.2f} ms | FastSHAP: {shap_latency:.2f} ms")

    # Predicted Class Banner
    class_colors = {
        "Healthy": "🟢 Healthy",
        "Mild_Stress": "🟡 Mild Stress",
        "Moderate_Stress": "🟧 Moderate Stress",
        "Severe_Stress": "🔴 Severe Stress",
    }
    st.info(f"**Predicted Severity Classification**: `{class_colors.get(pred_cls_name, pred_cls_name)}`")

    # Softmax Probabilities Bar Chart
    st.markdown("#### Softmax Class Probabilities")
    df_probs = pd.DataFrame({
        "Class": SEVERITY_CLASSES,
        "Probability": [prob_dict[c] for c in SEVERITY_CLASSES]
    })
    st.bar_chart(df_probs.set_index("Class"), height=180)

    # Continuous Symptom Scores Metrics
    st.markdown("#### Continuous Symptom Scores")
    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        st.metric("Depression (PHQ-9)", f"{reg_vals[0]:.2f} / 34")
    with c_m2:
        st.metric("Anxiety (GAD-7)", f"{reg_vals[1]:.2f} / 24")
    with c_m3:
        st.metric("Stress (PSS Scale)", f"{reg_vals[2]:.2f} / 39")

    # FastSHAP Feature Impact
    st.markdown("#### FastSHAP Feature Attributions (Top 6)")
    df_shap = pd.DataFrame([
        {"Feature": item.feature_name, "Shapley Value": item.importance_score}
        for item in attributions_list[:6]
    ])
    st.dataframe(df_shap, use_container_width=True, hide_index=True)

    # Clinical Narrative Summary
    st.markdown("#### Clinical Narrative Synthesis")
    st.success(narrative_payload.summary)
