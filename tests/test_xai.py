import os
import sys
import json
import pytest
import torch
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.multi_task import MultiTaskModel
from xai.shap_explainer import FastSHAPExplainer
from xai.narrative_engine import ClinicalNarrativeEngine
from schemas.payload import ClassificationOutput, RegressionOutput
from server.main import app, load_server_artifacts


@pytest.fixture(scope="module", autouse=True)
def init_server():
    load_server_artifacts()


client = TestClient(app)


def test_fastshap_explainer_speed_and_structure():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    model = MultiTaskModel()
    model.eval()

    explainer = FastSHAPExplainer()

    v_in = torch.randn(1, 128)
    a_in = torch.randn(1, 256)
    t_scaled = np.zeros(18, dtype=np.float32)

    attributions_list, attributions_dict, latency_ms = explainer.explain(model, v_in, a_in, t_scaled)

    assert len(attributions_list) == 18
    assert len(attributions_dict) == 18
    assert latency_ms < 15.0, f"FastSHAP latency was {latency_ms:.2f} ms, target < 15.0 ms!"

    for item in attributions_list:
        assert item.direction in ["risk_factor", "protective_factor", "neutral"]


def test_clinical_narrative_engine_generation():
    engine = ClinicalNarrativeEngine()

    cls_out = ClassificationOutput(
        predicted_class="Moderate_Stress",
        predicted_class_id=2,
        probabilities={"Healthy": 0.1, "Mild_Stress": 0.2, "Moderate_Stress": 0.6, "Severe_Stress": 0.1},
    )

    reg_out = RegressionOutput(
        depression_score=15.5,
        anxiety_score=14.2,
        stress_score=22.8,
    )

    explainer = FastSHAPExplainer()
    v_in = torch.randn(1, 128)
    a_in = torch.randn(1, 256)
    t_scaled = np.zeros(18, dtype=np.float32)
    model = MultiTaskModel()
    model.eval()

    attributions_list, _, _ = explainer.explain(model, v_in, a_in, t_scaled)

    narrative = engine.generate_narrative(cls_out, reg_out, attributions_list)

    assert "moderate" in narrative.summary.lower()
    assert len(narrative.key_risk_factors) > 0
    assert len(narrative.clinical_recommendations) > 0


def test_server_xai_websocket_response():
    payload = {
        "visual_vector": {"values": [0.05] * 128},
        "acoustic_vector": {"values": [0.05] * 256},
        "tabular": {"values": [3.0] * 18},
    }

    with client.websocket_connect("/evaluate/ws") as websocket:
        websocket.send_text(json.dumps(payload))
        data_text = websocket.receive_text()
        data = json.loads(data_text)

        assert data["status"] == "success"
        assert "xai" in data
        assert data["xai"] is not None
        assert "attributions" in data["xai"]
        assert "narrative" in data["xai"]
        assert "cross_attention_weights" in data["xai"]
        assert len(data["xai"]["attributions"]) == 18
