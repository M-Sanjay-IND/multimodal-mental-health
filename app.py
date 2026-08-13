import os
import sys
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import joblib

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from schemas.payload import (
    EvaluationPayload,
    VisualVector,
    AcousticVector,
    TabularVector,
    ClassificationOutput,
    RegressionOutput,
    SEVERITY_CLASSES,
)
from models.multi_task import MultiTaskModel
from xai.shap_explainer import FastSHAPExplainer
from xai.narrative_engine import ClinicalNarrativeEngine

# ── Artifact paths ────────────────────────────────────────────────────────────
_ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
_MODEL_INT8   = os.path.join(_ARTIFACT_DIR, "model_int8.pt")
_MODEL_FP32   = os.path.join(_ARTIFACT_DIR, "model_state.pt")
_PREP_PATH    = os.path.join(_ARTIFACT_DIR, "preprocessor.joblib")

# ── Lazy-loaded globals ───────────────────────────────────────────────────────
_MODEL            = None
_PREPROCESSOR     = None
_EXPLAINER        = None
_NARRATIVE_ENGINE = None


def _load_once():
    global _MODEL, _PREPROCESSOR, _EXPLAINER, _NARRATIVE_ENGINE
    if _MODEL is not None:
        return

    _EXPLAINER        = FastSHAPExplainer()
    _NARRATIVE_ENGINE = ClinicalNarrativeEngine()

    if os.path.exists(_PREP_PATH):
        _PREPROCESSOR = joblib.load(_PREP_PATH)

    base = MultiTaskModel()
    if os.path.exists(_MODEL_INT8):
        ck = torch.load(_MODEL_INT8, map_location="cpu", weights_only=False)
        _MODEL = torch.ao.quantization.quantize_dynamic(base, {torch.nn.Linear}, dtype=torch.qint8)
        _MODEL.load_state_dict(ck["model_state_dict"])
    elif os.path.exists(_MODEL_FP32):
        ck = torch.load(_MODEL_FP32, map_location="cpu", weights_only=False)
        _MODEL = base
        _MODEL.load_state_dict(ck["model_state_dict"])
    else:
        _MODEL = base

    _MODEL.eval()


# ── Severity colour map ───────────────────────────────────────────────────────
_SEVERITY_COLOUR = {
    "Minimal":  "#22c55e",
    "Mild":     "#eab308",
    "Moderate": "#f97316",
    "Severe":   "#ef4444",
}


def _severity_badge(cls_name: str) -> str:
    colour = _SEVERITY_COLOUR.get(cls_name, "#6b7280")
    return (
        f'<span style="background:{colour};color:#fff;padding:4px 14px;'
        f'border-radius:999px;font-weight:700;font-size:1rem;">'
        f'{cls_name}</span>'
    )


def _score_bar(label: str, value: float, max_val: float, colour: str) -> str:
    pct = min(100, round(value / max_val * 100, 1))
    return (
        f"<div style='margin-bottom:10px'>"
        f"<div style='display:flex;justify-content:space-between;font-size:.85rem;color:#94a3b8'>"
        f"<span>{label}</span><span>{value:.2f} / {max_val:.0f}</span></div>"
        f"<div style='background:#1e293b;border-radius:6px;height:10px;overflow:hidden'>"
        f"<div style='background:{colour};width:{pct}%;height:100%;border-radius:6px;"
        f"transition:width .5s ease'></div></div></div>"
    )


def predict(
    sleep_q, social_e, app_usage, typing_wpm, session_freq, idle_min,
    hrv, gsr, mood_var, phq_q1, phq_q2, gad_q1, heart_rate, resp_rate,
    skin_temp, step_count, bmi, screen_on,
):
    _load_once()

    tabular_vals = [
        float(sleep_q), float(social_e), float(app_usage), float(typing_wpm),
        float(session_freq), float(idle_min), float(hrv), float(gsr),
        float(mood_var), float(phq_q1), float(phq_q2), float(gad_q1),
        float(heart_rate), float(resp_rate), float(skin_temp),
        float(step_count), float(bmi), float(screen_on),
    ]

    v_vec = VisualVector(values=[0.1] * 128)
    a_vec = AcousticVector(values=[0.1] * 256)
    payload = EvaluationPayload(
        visual_vector=v_vec,
        acoustic_vector=a_vec,
        tabular=TabularVector(values=tabular_vals),
    )

    raw_tab = payload.tabular.values
    if _PREPROCESSOR is not None:
        scaled = _PREPROCESSOR.transform([raw_tab])[0]
    else:
        scaled = np.array(raw_tab, dtype=np.float32)

    v_t = torch.from_numpy(np.array([payload.visual_vector.values], dtype=np.float32))
    a_t = torch.from_numpy(np.array([payload.acoustic_vector.values], dtype=np.float32))
    t_t = torch.from_numpy(np.array([scaled], dtype=np.float32))

    with torch.no_grad():
        logits, reg_scores, attn_dict = _MODEL(v_t, a_t, t_t)
        probs      = F.softmax(logits, dim=-1)[0].numpy().tolist()
        pred_id    = int(torch.argmax(logits, dim=-1)[0].item())
        pred_class = SEVERITY_CLASSES[pred_id]
        reg_vals   = reg_scores[0].numpy().tolist()

    cls_out = ClassificationOutput(
        predicted_class=pred_class,
        predicted_class_id=pred_id,
        probabilities={c: round(p, 4) for c, p in zip(SEVERITY_CLASSES, probs)},
    )
    reg_out = RegressionOutput(
        depression_score=round(reg_vals[0], 2),
        anxiety_score=round(reg_vals[1], 2),
        stress_score=round(reg_vals[2], 2),
    )

    attrs, _, _ = _EXPLAINER.explain(_MODEL, v_t, a_t, scaled)
    narrative   = _NARRATIVE_ENGINE.generate_narrative(cls_out, reg_out, attrs, attn_dict)

    # ── Build rich HTML output ────────────────────────────────────────────────
    html = f"""
<div style="font-family:'Inter',sans-serif;background:#0f172a;color:#e2e8f0;
            border-radius:16px;padding:28px 32px;line-height:1.7">

  <h2 style="margin:0 0 6px;font-size:1.4rem;color:#f8fafc">
    🧠 Diagnostic Assessment Result
  </h2>
  <p style="margin:0 0 20px;color:#64748b;font-size:.9rem">
    DCMF-Net · FastSHAP · Clinical Narrative Engine
  </p>

  <div style="margin-bottom:20px">
    <span style="color:#94a3b8;font-size:.85rem;text-transform:uppercase;
                 letter-spacing:.08em">Severity Classification</span><br/>
    {_severity_badge(pred_class)}
  </div>

  <div style="background:#1e293b;border-radius:12px;padding:18px 20px;margin-bottom:20px">
    <div style="color:#94a3b8;font-size:.8rem;text-transform:uppercase;
                letter-spacing:.08em;margin-bottom:12px">Severity Class Probabilities</div>
    {"".join(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">'
        f'<span style="width:70px;font-size:.82rem;color:#cbd5e1">{c}</span>'
        f'<div style="flex:1;background:#0f172a;border-radius:4px;height:8px">'
        f'<div style="background:{"#22c55e" if c=="Minimal" else "#eab308" if c=="Mild" else "#f97316" if c=="Moderate" else "#ef4444"};'
        f'width:{round(p*100,1)}%;height:100%;border-radius:4px"></div></div>'
        f'<span style="font-size:.8rem;color:#64748b;width:42px;text-align:right">{round(p*100,1)}%</span>'
        f'</div>'
        for c, p in zip(SEVERITY_CLASSES, probs)
    )}
  </div>

  <div style="background:#1e293b;border-radius:12px;padding:18px 20px;margin-bottom:20px">
    <div style="color:#94a3b8;font-size:.8rem;text-transform:uppercase;
                letter-spacing:.08em;margin-bottom:12px">Regression Severity Scores</div>
    {_score_bar("Depression (PHQ-9 scale)", reg_out.depression_score, 34, "#818cf8")}
    {_score_bar("Anxiety (GAD-7 scale)",    reg_out.anxiety_score,    24, "#38bdf8")}
    {_score_bar("Stress (PSS scale)",       reg_out.stress_score,     39, "#fb923c")}
  </div>

  <div style="background:#1e293b;border-radius:12px;padding:18px 20px;margin-bottom:20px">
    <div style="color:#94a3b8;font-size:.8rem;text-transform:uppercase;
                letter-spacing:.08em;margin-bottom:10px">📋 Clinical Narrative</div>
    <p style="margin:0;color:#e2e8f0;font-size:.9rem">{narrative.summary}</p>
  </div>

  {'<div style="background:#1e293b;border-radius:12px;padding:18px 20px;margin-bottom:20px"><div style="color:#94a3b8;font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px">⚠️ Key Risk Factors</div><ul style="margin:0;padding-left:18px;color:#fca5a5;font-size:.88rem">' + "".join(f"<li>{r}</li>" for r in narrative.key_risk_factors) + "</ul></div>" if narrative.key_risk_factors else ""}

  {'<div style="background:#1e293b;border-radius:12px;padding:18px 20px"><div style="color:#94a3b8;font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px">💡 Clinical Recommendations</div><ul style="margin:0;padding-left:18px;color:#86efac;font-size:.88rem">' + "".join(f"<li>{r}</li>" for r in narrative.clinical_recommendations) + "</ul></div>" if narrative.clinical_recommendations else ""}

  <p style="margin:20px 0 0;font-size:.75rem;color:#334155;text-align:center">
    ⚠️ This tool is for research and demonstration purposes only — not a clinical diagnosis.
  </p>
</div>
"""
    return html


# ── Gradio UI ─────────────────────────────────────────────────────────────────
_CSS = """
body { background: #0f172a !important; }
.gradio-container { background: #0f172a !important; font-family: 'Inter', sans-serif !important; }
footer { display: none !important; }
.gr-button-primary { background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
                     border: none !important; font-weight: 700 !important; }
.gr-button-primary:hover { opacity: .9 !important; transform: translateY(-1px) !important; }
label { color: #94a3b8 !important; font-size: .82rem !important; }
"""

with gr.Blocks(title="MindScan — Multimodal Mental Health Assessment") as demo:

    gr.HTML("""
    <div style="text-align:center;padding:32px 0 8px;font-family:'Inter',sans-serif">
      <div style="font-size:2.8rem;margin-bottom:6px">🧠</div>
      <h1 style="margin:0;font-size:2rem;font-weight:800;
                 background:linear-gradient(135deg,#6366f1,#38bdf8);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent">
        MindScan — Multimodal Mental Health Assessment
      </h1>
      <p style="color:#64748b;margin:8px 0 0;font-size:.95rem">
        DCMF-Net · FastSHAP XAI · Zero-Hallucination Clinical Narrative Engine
      </p>
    </div>
    """)

    with gr.Row():
        # ── Left panel — inputs ───────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Biometric & Behavioural Inputs")

            with gr.Group():
                gr.Markdown("**Sleep & Wellbeing**")
                sleep_q    = gr.Slider(1.0, 5.0, value=3.0, step=0.5, label="Sleep Quality  (1=Poor → 5=Excellent)")
                social_e   = gr.Slider(1.0, 5.0, value=3.0, step=0.5, label="Social Engagement  (1=Isolated → 5=Active)")
                mood_var   = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="Mood Variability Index")

            with gr.Group():
                gr.Markdown("**Digital Behaviour**")
                app_usage  = gr.Number(value=180.0, label="Daily App Usage (minutes)")
                typing_wpm = gr.Number(value=45.0,  label="Typing Speed (WPM)")
                session_freq = gr.Number(value=12.0, label="Daily Session Frequency")
                idle_min   = gr.Number(value=30.0,  label="Idle Duration (minutes)")
                screen_on  = gr.Number(value=6.5,   label="Screen-On Time (hours)")

            with gr.Group():
                gr.Markdown("**Physiological Sensors**")
                hrv        = gr.Number(value=55.0,  label="Heart Rate Variability — HRV (ms)")
                gsr        = gr.Number(value=0.3,   label="Galvanic Skin Response — GSR (µS)")
                heart_rate = gr.Number(value=72.0,  label="Resting Heart Rate (BPM)")
                resp_rate  = gr.Number(value=16.0,  label="Respiratory Rate (breaths/min)")
                skin_temp  = gr.Number(value=36.5,  label="Skin Temperature (°C)")
                step_count = gr.Number(value=5500.0, label="Daily Step Count")
                bmi        = gr.Number(value=22.5,  label="BMI")

            with gr.Group():
                gr.Markdown("**Self-Report Screening (PHQ / GAD)**")
                phq_q1 = gr.Slider(0, 3, value=1, step=1, label="PHQ-9 Q1 — Anhedonia (0–3)")
                phq_q2 = gr.Slider(0, 3, value=1, step=1, label="PHQ-9 Q2 — Depressed Mood (0–3)")
                gad_q1 = gr.Slider(0, 3, value=1, step=1, label="GAD-7 Q1 — Anxiety Feeling (0–3)")

            run_btn = gr.Button("🔍 Run Assessment", variant="primary", size="lg")

        # ── Right panel — output ──────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 🩺 Assessment Output")
            result_html = gr.HTML(
                value='<div style="background:#1e293b;border-radius:16px;padding:48px;'
                      'text-align:center;color:#475569;font-family:Inter,sans-serif">'
                      '⬅️ Fill in your biometric parameters and click <strong>Run Assessment</strong></div>'
            )

    run_btn.click(
        fn=predict,
        inputs=[
            sleep_q, social_e, app_usage, typing_wpm, session_freq, idle_min,
            hrv, gsr, mood_var, phq_q1, phq_q2, gad_q1,
            heart_rate, resp_rate, skin_temp, step_count, bmi, screen_on,
        ],
        outputs=[result_html],
    )

    gr.HTML("""
    <div style="text-align:center;padding:20px 0 8px;color:#334155;
                font-size:.78rem;font-family:'Inter',sans-serif">
      Built with DCMF-Net · INT8 ONNX · FastSHAP · Gradio · Hugging Face Spaces
    </div>
    """)


if __name__ == "__main__":
    demo.launch(css=_CSS)
