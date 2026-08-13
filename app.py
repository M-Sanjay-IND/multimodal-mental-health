import os
import sys
import gradio as gr
import uvicorn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from server.main import app, predict_payload
from schemas.payload import EvaluationPayload, VisualVector, AcousticVector, TabularVector


def gradio_predict_wrapper(sleep_q, social_e, app_usage, typing_wpm, session_freq, idle_min):
    """Interactive Gradio wrapper for online evaluation demo."""
    # Build dummy visual (128) & acoustic (256) vectors + tabular input
    v_vec = VisualVector(values=[0.1] * 128)
    a_vec = AcousticVector(values=[0.1] * 256)

    tabular_vals = [
        float(sleep_q),
        float(social_e),
        float(app_usage),
        float(typing_wpm),
        float(session_freq),
        float(idle_min),
        0.5, 0.2, 0.7, 0.1, 0.0, 0.5, 120.0, 3.5, 72.0, 45.0, 36.5, 2.5
    ]

    payload = EvaluationPayload(
        visual_vector=v_vec,
        acoustic_vector=a_vec,
        tabular=TabularVector(values=tabular_vals),
    )

    response = predict_payload(payload)

    summary_text = f"### 🧠 Diagnostic Assessment Result\n\n"
    summary_text += f"- **Severity Class**: `{response.classification.predicted_class}`\n"
    summary_text += f"- **Depression Score**: `{response.regression.depression_score} / 34.0`\n"
    summary_text += f"- **Anxiety Score**: `{response.regression.anxiety_score} / 24.0`\n"
    summary_text += f"- **Stress Score**: `{response.regression.stress_score} / 39.0`\n\n"

    if response.xai:
        summary_text += f"### 📋 Clinical Narrative\n{response.xai.narrative.summary}\n\n"
        summary_text += f"#### ⚠️ Key Risk Factors\n" + "\n".join([f"- {r}" for r in response.xai.narrative.key_risk_factors]) + "\n\n"
        summary_text += f"#### 💡 Clinical Recommendations\n" + "\n".join([f"- {rec}" for rec in response.xai.narrative.clinical_recommendations])

    return summary_text


# Build Gradio UI block
with gr.Blocks(title="Multimodal Mental Health Assessment API & Live Demo") as demo:
    gr.Markdown("# 🧠 Multimodal Mental Health Assessment Engine (DCMF-Net)")
    gr.Markdown("Real-time FastAPI REST (`/evaluate/rest`), WebSocket (`/evaluate/ws`), and Explainable AI (FastSHAP) Server.")

    with gr.Row():
        with gr.Column():
            sleep_q = gr.Slider(minimum=1.0, maximum=5.0, value=3.0, step=0.5, label="Sleep Quality (1=Poor, 5=Excellent)")
            social_e = gr.Slider(minimum=1.0, maximum=5.0, value=3.0, step=0.5, label="Social Engagement (1=Isolated, 5=Active)")
            app_usage = gr.Number(value=180.0, label="Daily App Usage (Minutes)")
            typing_wpm = gr.Number(value=45.0, label="Typing Speed (WPM)")
            session_freq = gr.Number(value=12.0, label="Daily Session Frequency")
            idle_min = gr.Number(value=30.0, label="Idle Duration (Minutes)")
            submit_btn = gr.Button("Run Multimodal Assessment", variant="primary")

        with gr.Column():
            output_markdown = gr.Markdown("### Submit parameters to see assessment...")

    submit_btn.click(
        fn=gradio_predict_wrapper,
        inputs=[sleep_q, social_e, app_usage, typing_wpm, session_freq, idle_min],
        outputs=[output_markdown],
    )

# Mount Gradio interface onto FastAPI app so both REST/WebSocket API and UI run together
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860)
