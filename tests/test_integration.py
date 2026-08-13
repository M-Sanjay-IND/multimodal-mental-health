import json
import pytest
from fastapi.testclient import TestClient
from server.main import app, load_server_artifacts
from schemas.payload import SEVERITY_CLASSES


@pytest.fixture(scope="module", autouse=True)
def init_server():
    load_server_artifacts()


client = TestClient(app)


def test_end_to_end_rest_integration_roundtrip():
    """
    Validates complete REST API integration:
    Client Payload -> REST endpoint /evaluate/rest -> Quantized Inference -> FastSHAP XAI -> Narrative Engine.
    """
    payload = {
        "visual_vector": {"values": [0.15] * 128},
        "acoustic_vector": {"values": [-0.05] * 256},
        "tabular": {"values": [1.5] * 18},
    }

    response = client.post("/evaluate/rest", json=payload)
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"

    data = response.json()
    assert data["status"] == "success"
    assert "classification" in data
    assert "regression" in data
    assert "xai" in data
    assert "shap_attribution" in data
    assert "narrative" in data

    # 1. Classification contract assertions
    cls_data = data["classification"]
    assert cls_data["predicted_class"] in SEVERITY_CLASSES
    assert 0 <= cls_data["predicted_class_id"] <= 3
    assert len(cls_data["probabilities"]) == 4

    # 2. Regression contract assertions
    reg_data = data["regression"]
    assert 0.0 <= reg_data["depression_score"] <= 34.0
    assert 0.0 <= reg_data["anxiety_score"] <= 24.0
    assert 0.0 <= reg_data["stress_score"] <= 39.0

    # 3. FastSHAP XAI contract assertions
    assert "attributions" in data["shap_attribution"]
    assert len(data["shap_attribution"]["attributions"]) == 18

    # 4. Narrative Engine contract assertions
    assert data["narrative"] is not None
    if isinstance(data["narrative"], dict):
        assert len(data["narrative"]["summary"]) > 0
    else:
        assert len(str(data["narrative"])) > 0


def test_end_to_end_websocket_integration_roundtrip():
    """
    Validates complete WebSocket integration:
    WebSocket connect -> Frame Payload -> Real-time Inference -> Streamed JSON Response.
    """
    payload = {
        "visual_vector": {"values": [0.0] * 128},
        "acoustic_vector": {"values": [0.0] * 256},
        "tabular": {"values": [0.5] * 18},
    }

    with client.websocket_connect("/evaluate/ws") as websocket:
        websocket.send_text(json.dumps(payload))
        data_text = websocket.receive_text()
        data = json.loads(data_text)

        assert data["status"] == "success"
        assert data["classification"]["predicted_class"] in SEVERITY_CLASSES
        assert "depression_score" in data["regression"]
        assert len(data["shap_attribution"]["attributions"]) == 18
        assert data["narrative"] is not None
