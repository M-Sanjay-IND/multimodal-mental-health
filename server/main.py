import os
import sys
import json
import logging
import joblib
import torch
import torch.nn.functional as F
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas.payload import (
    EvaluationPayload,
    EvaluationResponse,
    ClassificationOutput,
    RegressionOutput,
    SEVERITY_CLASSES,
    TABULAR_FEATURE_NAMES,
)
from models.multi_task import MultiTaskModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server.main")

# Global model and preprocessor references
MODEL = None
PREPROCESSOR = None


def load_server_artifacts():
    global MODEL, PREPROCESSOR
    artifact_dir = "artifacts"
    prep_path = os.path.join(artifact_dir, "preprocessor.joblib")
    model_path = os.path.join(artifact_dir, "model_state.pt")

    logger.info("Loading preprocessor and model state artifacts...")

    if os.path.exists(prep_path):
        PREPROCESSOR = joblib.load(prep_path)
        logger.info(f"Loaded preprocessor from {prep_path}")
    else:
        logger.warning(f"Preprocessor artifact not found at {prep_path}!")

    MODEL = MultiTaskModel()
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        MODEL.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model weights from {model_path}")
    else:
        logger.warning(f"Model checkpoint not found at {model_path}! Using uninitialized weights.")

    MODEL.eval()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_server_artifacts()
    yield


app = FastAPI(
    title="Multimodal Psychiatric Evaluation & Severity Estimation API",
    description="Real-Time Async Gateway for DCMF-Net Multimodal Mental Health Assessment",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {
        "service": "Multimodal Psychiatric Evaluation Gateway",
        "status": "online",
        "health_check": "/health",
        "websocket_endpoint": "/evaluate/ws",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "preprocessor_loaded": PREPROCESSOR is not None,
        "version": "1.0.0",
    }


def predict_payload(payload: EvaluationPayload) -> EvaluationResponse:
    global MODEL, PREPROCESSOR

    if MODEL is None:
        raise RuntimeError("Model is not loaded!")

    # 1. Extract raw tabular values and scale
    raw_tabular = payload.tabular.values
    if PREPROCESSOR is not None:
        scaled_tabular = PREPROCESSOR.transform([raw_tabular])[0]
    else:
        scaled_tabular = np.array(raw_tabular, dtype=np.float32)

    # 2. Build Tensors
    v_tensor = torch.from_numpy(np.array([payload.visual_vector.values], dtype=np.float32))
    a_tensor = torch.from_numpy(np.array([payload.acoustic_vector.values], dtype=np.float32))
    t_tensor = torch.from_numpy(np.array([scaled_tabular], dtype=np.float32))

    # 3. Model Forward Pass
    with torch.no_grad():
        logits, reg_scores, _ = MODEL(v_tensor, a_tensor, t_tensor)

        probs = F.softmax(logits, dim=-1)[0].numpy().tolist()
        pred_cls_id = int(torch.argmax(logits, dim=-1)[0].item())
        pred_cls_name = SEVERITY_CLASSES[pred_cls_id]

        reg_vals = reg_scores[0].numpy().tolist()

    prob_dict = {cls_name: float(round(p, 4)) for cls_name, p in zip(SEVERITY_CLASSES, probs)}

    cls_output = ClassificationOutput(
        predicted_class=pred_cls_name,
        predicted_class_id=pred_cls_id,
        probabilities=prob_dict,
    )

    reg_output = RegressionOutput(
        depression_score=float(round(reg_vals[0], 2)),
        anxiety_score=float(round(reg_vals[1], 2)),
        stress_score=float(round(reg_vals[2], 2)),
    )

    response = EvaluationResponse(
        status="success",
        classification=cls_output,
        regression=reg_output,
    )

    return response


@app.post("/evaluate/rest", response_model=EvaluationResponse)
def evaluate_rest(payload: EvaluationPayload):
    try:
        return predict_payload(payload)
    except Exception as e:
        logger.error(f"Error in REST evaluation: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.websocket("/evaluate/ws")
async def evaluate_websocket(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted.")

    try:
        while True:
            raw_text = await websocket.receive_text()
            try:
                # Parse JSON string into Pydantic model
                payload_data = json.loads(raw_text)
                payload = EvaluationPayload.model_validate(payload_data)

                # Run inference
                response = predict_payload(payload)

                # Send back response JSON string
                await websocket.send_text(response.model_dump_json())

            except (ValidationError, json.JSONDecodeError) as ve:
                err_response = {"status": "error", "error_type": "validation_error", "details": str(ve)}
                await websocket.send_text(json.dumps(err_response))
            except Exception as ex:
                err_response = {"status": "error", "error_type": "inference_error", "details": str(ex)}
                await websocket.send_text(json.dumps(err_response))

    except WebSocketDisconnect:
        logger.info("WebSocket connection disconnected.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
