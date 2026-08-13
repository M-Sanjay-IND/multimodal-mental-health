import pytest
from pydantic import ValidationError
from schemas.payload import (
    VisualVector,
    AcousticVector,
    TabularVector,
    TabularFeatures,
    EvaluationPayload,
    ClassificationOutput,
    RegressionOutput,
    EvaluationResponse,
    TABULAR_FEATURE_NAMES,
)


def test_visual_vector_valid():
    v = VisualVector(values=[0.1] * 128)
    assert len(v.values) == 128


def test_visual_vector_invalid_length():
    with pytest.raises(ValidationError):
        VisualVector(values=[0.1] * 127)


def test_visual_vector_nan():
    with pytest.raises(ValidationError):
        VisualVector(values=[0.1] * 127 + [float("nan")])


def test_acoustic_vector_valid():
    a = AcousticVector(values=[0.5] * 256)
    assert len(a.values) == 256


def test_tabular_vector_valid():
    t = TabularVector(values=[1.0] * 18)
    assert len(t.values) == 18
    d = t.to_dict()
    assert len(d) == 18
    assert d["Sleep_Quality"] == 1.0


def test_evaluation_payload_valid():
    payload = EvaluationPayload(
        visual_vector=VisualVector(values=[0.0] * 128),
        acoustic_vector=AcousticVector(values=[0.0] * 256),
        tabular=TabularVector(values=[1.0] * 18),
    )
    assert isinstance(payload.tabular, TabularVector)
    assert len(payload.tabular.values) == 18


def test_evaluation_payload_list_input():
    payload = EvaluationPayload(
        visual_vector={"values": [0.0] * 128},
        acoustic_vector={"values": [0.0] * 256},
        tabular=[1.0] * 18,
    )
    assert isinstance(payload.tabular, TabularVector)
    assert len(payload.tabular.values) == 18


def test_evaluation_response_valid():
    resp = EvaluationResponse(
        classification=ClassificationOutput(
            predicted_class="Healthy",
            predicted_class_id=0,
            probabilities={"Healthy": 0.8, "Mild_Stress": 0.1, "Moderate_Stress": 0.05, "Severe_Stress": 0.05},
        ),
        regression=RegressionOutput(
            depression_score=5.0,
            anxiety_score=3.0,
            stress_score=8.0,
        ),
    )
    assert resp.status == "success"
    assert resp.classification.predicted_class == "Healthy"
