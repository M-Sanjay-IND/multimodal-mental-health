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

# Page Configuration - Ultra-Minimalist Black & White Theme
st.set_page_config(
    page_title="DCMF-Net Model Workspace",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Contrast Monochrome CSS
st.markdown("""
<style>
    /* Dark Monochrome Theme Tokens */
    .stApp {
        background-color: #09090b !important;
        color: #fafafa !important;
        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }
    
    /* Hide Streamlit Header Decoration */
    header[data-testid="stHeader"] {
        background-color: #09090b !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #121215 !important;
        border-right: 1px solid #27272a !important;
    }

    /* Monochrome Card Containers */
    .mono-card {
        background-color: #121215;
        border: 1px solid #27272a;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    .mono-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        background-color: #ffffff;
        color: #000000;
        font-weight: 700;
        font-family: monospace;
        font-size: 13px;
    }

    /* Metric Cards */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-family: monospace !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #a1a1aa !important;
        font-size: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    div[data-testid="stMetric"] {
        background-color: #121215 !important;
        border: 1px solid #27272a !important;
        border-radius: 8px !important;
        padding: 12px !important;
    }

    /* Form Sliders & Inputs */
    .stSlider > div {
        color: #ffffff !important;
    }
    
    /* Tab Headers */
    button[data-baseweb="tab"] {
        color: #a1a1aa !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        text-transform: uppercase !important;
    }
    button[aria-selected="true"] {
        color: #ffffff !important;
        border-bottom-color: #ffffff !important;
    }
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

def derive_multimodal_vectors(raw_tabular):
    """
    Synthesizes 128D visual and 256D acoustic latent embeddings from raw tabular biomarker parameters
    when live camera/mic streams are unprovided, preventing fusion layer collapse to zero.
    """
    v_base = np.zeros(128, dtype=np.float32)
    v_base[0::4] = raw_tabular[6]
    v_base[1::4] = (raw_tabular[7] - 15.0) / 30.0
    v_base[2::4] = raw_tabular[8]
    v_base[3::4] = raw_tabular[9]

    a_base = np.zeros(256, dtype=np.float32)
    a_base[0::4] = raw_tabular[10]
    a_base[1::4] = raw_tabular[11]
    a_base[2::4] = (raw_tabular[12] - 180.0) / 100.0
    a_base[3::4] = (raw_tabular[13] - 3.2) / 2.0

    return v_base, a_base

# Load Model & Preprocessor
model, preprocessor, explainer, narrative_engine, epoch, val_loss = load_model_and_preprocessor()

# Monochrome Header
st.markdown("""
<div style="border-bottom: 1px solid #27272a; padding-bottom: 12px; margin-bottom: 16px;">
    <h1 style="color: #ffffff; margin: 0; font-size: 24px; font-weight: 800; letter-spacing: -0.02em; font-family: monospace;">
        PSYCH-METRIC // DCMF-NET MODEL WORKSPACE
    </h1>
    <p style="color: #a1a1aa; margin: 4px 0 0 0; font-size: 12px; font-family: monospace;">
        Multimodal Affective Transformer // Direct PyTorch Evaluation & FastSHAP Attributions
    </p>
</div>
""", unsafe_allow_html=True)

# System Status Metrics
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    st.metric("MODEL ARCHITECTURE", "DCMF-Net Transformer")
with col_s2:
    st.metric("LOADED CHECKPOINT", f"Epoch {epoch}")
with col_s3:
    st.metric("VALIDATION LOSS", f"{val_loss:.4f}" if isinstance(val_loss, float) else str(val_loss))
with col_s4:
    st.metric("PREPROCESSOR", "RobustScaler" if preprocessor else "Raw Scale")

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# Sidebar: Preset Selectors
st.sidebar.markdown("<h3 style='color: #ffffff; font-size: 14px; font-family: monospace;'>SCENARIO PRESETS</h3>", unsafe_allow_html=True)
preset = st.sidebar.selectbox(
    "Select Clinical Scenario:",
    [
        "Optimal Healthy Baseline",
        "Mild Work Stress & Fatigue",
        "Moderate Depressive Affect",
        "Severe Crisis & Agitation",
    ]
)

PRESETS = {
    "Optimal Healthy Baseline": [4.5, 4.5, 110.0, 65.0, 4.0, 10.0, 0.65, 15.0, 0.75, 0.35, 0.2, 1.8, 195.0, 3.5, 68.0, 75.0, 36.6, 0.8],
    "Mild Work Stress & Fatigue": [3.0, 3.2, 250.0, 50.0, 8.0, 25.0, 0.40, 22.0, 0.45, 0.25, -0.1, 1.1, 175.0, 2.8, 78.0, 52.0, 36.5, 2.4],
    "Moderate Depressive Affect": [2.0, 2.0, 420.0, 35.0, 12.0, 55.0, 0.20, 28.0, 0.20, 0.15, -0.8, 0.5, 150.0, 2.1, 88.0, 32.0, 36.2, 4.8],
    "Severe Crisis & Agitation": [1.0, 1.0, 580.0, 20.0, 18.0, 90.0, 0.05, 42.0, 0.05, 0.05, -1.6, 0.2, 120.0, 1.4, 115.0, 16.0, 35.8, 8.5],
}

default_vals = PRESETS[preset]

# Main Columns
left_col, right_col = st.columns([7, 5])

with left_col:
    st.markdown("<h3 style='color: #ffffff; font-size: 14px; font-family: monospace; border-bottom: 1px solid #27272a; padding-bottom: 6px;'>BIOMARKER FEATURE INPUTS</h3>", unsafe_allow_html=True)

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
        feature_inputs["Daily_App_Usage_Min"] = st.slider("Daily Screen Time (min)", 0.0, 720.0, float(default_vals[2]), 5.0)
        feature_inputs["Typing_Speed_WPM"] = st.slider("Typing Speed (WPM)", 0.0, 150.0, float(default_vals[3]), 1.0)
        feature_inputs["Session_Frequency"] = st.slider("Daily Session Frequency", 0.0, 30.0, float(default_vals[4]), 1.0)
        feature_inputs["Idle_Time_Min"] = st.slider("Screen Idle Duration (min)", 0.0, 180.0, float(default_vals[5]), 1.0)

    with tab2:
        feature_inputs["Facial_Emotion_Variance"] = st.slider("Facial Emotion Variance (0-1)", 0.0, 1.0, float(default_vals[6]), 0.01)
        feature_inputs["Eye_Blink_Rate"] = st.slider("Eye Blink Rate (blinks/min)", 0.0, 80.0, float(default_vals[7]), 1.0)
        feature_inputs["Smile_Intensity"] = st.slider("Smile Intensity AU12 (0-1)", 0.0, 1.0, float(default_vals[8]), 0.01)
        feature_inputs["Head_Motion_Index"] = st.slider("Head Motion Index (0-1)", 0.0, 1.0, float(default_vals[9]), 0.01)

    with tab3:
        feature_inputs["MFCC_Mean"] = st.slider("MFCC Mean Coefficient", -5.0, 5.0, float(default_vals[10]), 0.05)
        feature_inputs["MFCC_Variance"] = st.slider("MFCC Variance", 0.0, 10.0, float(default_vals[11]), 0.1)
        feature_inputs["Pitch_Mean"] = st.slider("Pitch Frequency F0 (Hz)", 40.0, 350.0, float(default_vals[12]), 1.0)
        feature_inputs["Speech_Rate"] = st.slider("Speech Rate (words/sec)", 0.0, 10.0, float(default_vals[13]), 0.1)

    with tab4:
        feature_inputs["Heart_Rate_BPM"] = st.slider("Resting Heart Rate (BPM)", 40.0, 180.0, float(default_vals[14]), 1.0)
        feature_inputs["HRV_Index"] = st.slider("Heart Rate Variability HRV (ms)", 5.0, 150.0, float(default_vals[15]), 1.0)
        feature_inputs["Skin_Temperature"] = st.slider("Skin Temperature (°C)", 30.0, 42.0, float(default_vals[16]), 0.1)
        feature_inputs["GSR_Level"] = st.slider("Galvanic Skin Response GSR (μS)", 0.0, 20.0, float(default_vals[17]), 0.1)

# Extract raw tabular vector
raw_tabular = [feature_inputs[name] for name in TABULAR_FEATURE_NAMES]

# Scale via Preprocessor
if preprocessor:
    scaled_tabular = preprocessor.transform([raw_tabular])[0]
else:
    scaled_tabular = np.array(raw_tabular, dtype=np.float32)

# Synthesize Multimodal Embeddings
v_synth, a_synth = derive_multimodal_vectors(raw_tabular)

v_tensor = torch.from_numpy(np.array([v_synth], dtype=np.float32))
a_tensor = torch.from_numpy(np.array([a_synth], dtype=np.float32))
t_tensor = torch.from_numpy(np.array([scaled_tabular], dtype=np.float32))

# Model Forward Pass
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

# Right Column: Model Output
with right_col:
    st.markdown("<h3 style='color: #ffffff; font-size: 14px; font-family: monospace; border-bottom: 1px solid #27272a; padding-bottom: 6px;'>MODEL INFERENCE OUTPUT</h3>", unsafe_allow_html=True)
    st.caption(f"Latency: {latency_ms:.2f} ms | FastSHAP: {shap_latency:.2f} ms")

    # Predicted Class Banner
    class_labels = {
        "Healthy": "HEALTHY BASELINE",
        "Mild_Stress": "MILD STRESS",
        "Moderate_Stress": "MODERATE STRESS",
        "Severe_Stress": "SEVERE CRISIS",
    }
    
    st.markdown(f"""
    <div className="mono-card" style="background-color: #121215; border: 1px solid #27272a; padding: 16px; border-radius: 8px; margin-bottom: 16px;">
        <span style="color: #a1a1aa; font-size: 11px; font-family: monospace; display: block; margin-bottom: 4px;">PREDICTED SEVERITY CLASS</span>
        <span class="mono-badge">{class_labels.get(pred_cls_name, pred_cls_name)}</span>
    </div>
    """, unsafe_allow_html=True)

    # Softmax Probabilities Bar Chart
    st.markdown("<h4 style='color: #ffffff; font-size: 12px; font-family: monospace; margin-top: 16px;'>SOFTMAX CLASS PROBABILITIES</h4>", unsafe_allow_html=True)
    df_probs = pd.DataFrame({
        "Class": SEVERITY_CLASSES,
        "Probability (%)": [round(prob_dict[c] * 100, 2) for c in SEVERITY_CLASSES]
    })
    st.bar_chart(df_probs.set_index("Class"), height=180)

    # Continuous Symptom Scores Metrics
    st.markdown("<h4 style='color: #ffffff; font-size: 12px; font-family: monospace; margin-top: 16px;'>CONTINUOUS SYMPTOM SCORES</h4>", unsafe_allow_html=True)
    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        st.metric("DEPRESSION (PHQ-9)", f"{reg_vals[0]:.2f} / 34")
    with c_m2:
        st.metric("ANXIETY (GAD-7)", f"{reg_vals[1]:.2f} / 24")
    with c_m3:
        st.metric("STRESS (PSS SCALE)", f"{reg_vals[2]:.2f} / 39")

    # FastSHAP Feature Impact
    st.markdown("<h4 style='color: #ffffff; font-size: 12px; font-family: monospace; margin-top: 16px;'>FASTSHAP FEATURE ATTRIBUTIONS</h4>", unsafe_allow_html=True)
    df_shap = pd.DataFrame([
        {"Feature": item.feature_name.replace("_", " "), "Shapley Impact": f"{'+' if item.importance_score > 0 else ''}{item.importance_score:.2f}"}
        for item in attributions_list[:6]
    ])
    st.dataframe(df_shap, use_container_width=True, hide_index=True)

    # Clinical Narrative Summary
    st.markdown("<h4 style='color: #ffffff; font-size: 12px; font-family: monospace; margin-top: 16px;'>CLINICAL NARRATIVE SYNTHESIS</h4>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background-color: #121215; border: 1px solid #27272a; padding: 14px; border-radius: 8px; color: #d4d4d8; font-size: 12px; font-family: sans-serif; line-height: 1.5;">
        {narrative_payload.summary}
    </div>
    """, unsafe_allow_html=True)
