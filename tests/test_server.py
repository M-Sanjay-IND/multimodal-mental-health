import json
import pytest
from fastapi.testclient import TestClient
from server.main import app, load_server_artifacts


@pytest.fixture(scope="module", autouse=True)
def init_server():
    load_server_artifacts()


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "websocket_endpoint" in data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True


def test_evaluate_rest_endpoint_valid():
    payload = {
        "visual_vector": {"values": [0.1] * 128},
        "acoustic_vector": {"values": [0.2] * 256},
        "tabular": {"values": [1.0] * 18},
    }
    response = client.post("/evaluate/rest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "classification" in data
    assert "regression" in data
    assert data["classification"]["predicted_class"] in [
        "Healthy",
        "Mild_Stress",
        "Moderate_Stress",
        "Severe_Stress",
    ]
    assert 0.0 <= data["regression"]["depression_score"] <= 34.0
    assert 0.0 <= data["regression"]["anxiety_score"] <= 24.0
    assert 0.0 <= data["regression"]["stress_score"] <= 39.0


def test_websocket_evaluation_valid():
    payload = {
        "visual_vector": {"values": [0.0] * 128},
        "acoustic_vector": {"values": [0.0] * 256},
        "tabular": {"values": [2.0] * 18},
    }
    with client.websocket_connect("/evaluate/ws") as websocket:
        websocket.send_text(json.dumps(payload))
        data_text = websocket.receive_text()
        data = json.loads(data_text)

        assert data["status"] == "success"
        assert "classification" in data
        assert "regression" in data


def test_websocket_evaluation_invalid_payload():
    invalid_payload = {
        "visual_vector": {"values": [0.0] * 100},  # Invalid dimension (100 instead of 128)
        "acoustic_vector": {"values": [0.0] * 256},
        "tabular": {"values": [1.0] * 18},
    }
    with client.websocket_connect("/evaluate/ws") as websocket:
        websocket.send_text(json.dumps(invalid_payload))
        data_text = websocket.receive_text()
        data = json.loads(data_text)

        assert data["status"] == "error"
        assert data["error_type"] == "validation_error"
