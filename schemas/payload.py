import math
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator

TABULAR_FEATURE_NAMES = [
    "Sleep_Quality",
    "Social_Engagement",
    "Daily_App_Usage_Min",
    "Typing_Speed_WPM",
    "Session_Frequency",
    "Idle_Time_Min",
    "Facial_Emotion_Variance",
    "Eye_Blink_Rate",
    "Smile_Intensity",
    "Head_Motion_Index",
    "MFCC_Mean",
    "MFCC_Variance",
    "Pitch_Mean",
    "Speech_Rate",
    "Heart_Rate_BPM",
    "HRV_Index",
    "Skin_Temperature",
    "GSR_Level",
]

SEVERITY_CLASSES = ["Healthy", "Mild_Stress", "Moderate_Stress", "Severe_Stress"]


def _validate_float_list(v: List[float], expected_len: int, name: str) -> List[float]:
    if len(v) != expected_len:
        raise ValueError(f"{name} must have dimension {expected_len}, got {len(v)}")
    for i, x in enumerate(v):
        if not isinstance(x, (int, float)) or math.isnan(x) or math.isinf(x):
            raise ValueError(f"{name}[{i}] must be a finite float, got {x}")
    return [float(x) for x in v]


class VisualVector(BaseModel):
    values: List[float] = Field(..., description="128-dimensional visual embedding vector")

    @field_validator("values")
    @classmethod
    def check_dimension_and_finite(cls, v: List[float]) -> List[float]:
        return _validate_float_list(v, 128, "VisualVector")


class AcousticVector(BaseModel):
    values: List[float] = Field(..., description="256-dimensional acoustic embedding vector")

    @field_validator("values")
    @classmethod
    def check_dimension_and_finite(cls, v: List[float]) -> List[float]:
        return _validate_float_list(v, 256, "AcousticVector")


class TabularVector(BaseModel):
    values: List[float] = Field(..., description="18-dimensional tabular feature vector")

    @field_validator("values")
    @classmethod
    def check_dimension_and_finite(cls, v: List[float]) -> List[float]:
        return _validate_float_list(v, 18, "TabularVector")

    def to_dict(self) -> Dict[str, float]:
        return dict(zip(TABULAR_FEATURE_NAMES, self.values))


class TabularFeatures(BaseModel):
    Sleep_Quality: float = Field(..., ge=1.0, le=5.0)
    Social_Engagement: float = Field(..., ge=1.0, le=5.0)
    Daily_App_Usage_Min: float = Field(..., ge=0.0)
    Typing_Speed_WPM: float = Field(..., ge=0.0)
    Session_Frequency: float = Field(..., ge=0.0)
    Idle_Time_Min: float = Field(..., ge=0.0)
    Facial_Emotion_Variance: float = Field(..., ge=0.0)
    Eye_Blink_Rate: float = Field(..., ge=0.0)
    Smile_Intensity: float = Field(..., ge=0.0, le=1.0)
    Head_Motion_Index: float = Field(..., ge=0.0)
    MFCC_Mean: float
    MFCC_Variance: float = Field(..., ge=0.0)
    Pitch_Mean: float = Field(..., ge=0.0)
    Speech_Rate: float = Field(..., ge=0.0)
    Heart_Rate_BPM: float = Field(..., ge=0.0)
    HRV_Index: float = Field(..., ge=0.0)
    Skin_Temperature: float
    GSR_Level: float = Field(..., ge=0.0)

    @model_validator(mode="after")
    def check_finite(self) -> "TabularFeatures":
        for field, val in self.__dict__.items():
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Tabular field {field} must be finite, got {val}")
        return self

    def to_list(self) -> List[float]:
        return [getattr(self, name) for name in TABULAR_FEATURE_NAMES]


class EvaluationPayload(BaseModel):
    visual_vector: VisualVector
    acoustic_vector: AcousticVector
    tabular: Union[TabularVector, TabularFeatures, List[float]]

    @field_validator("tabular")
    @classmethod
    def normalize_tabular(cls, v: Union[TabularVector, TabularFeatures, List[float]]) -> TabularVector:
        if isinstance(v, TabularVector):
            return v
        elif isinstance(v, TabularFeatures):
            return TabularVector(values=v.to_list())
        elif isinstance(v, list):
            return TabularVector(values=v)
        raise ValueError("Invalid tabular vector input")


class ClassificationOutput(BaseModel):
    predicted_class: str
    predicted_class_id: int = Field(..., ge=0, le=3)
    probabilities: Dict[str, float]


class RegressionOutput(BaseModel):
    depression_score: float = Field(..., ge=0.0, le=34.0)
    anxiety_score: float = Field(..., ge=0.0, le=24.0)
    stress_score: float = Field(..., ge=0.0, le=39.0)


class SHAPAttribution(BaseModel):
    attributions: Dict[str, float]


class FeatureAttributionItem(BaseModel):
    feature_name: str
    importance_score: float
    direction: str  # "risk_factor", "protective_factor", or "neutral"


class ClinicalNarrativePayload(BaseModel):
    summary: str
    modality_breakdown: str
    key_risk_factors: List[str]
    protective_factors: List[str]
    clinical_recommendations: List[str]


class XaiPayload(BaseModel):
    attributions: List[FeatureAttributionItem]
    narrative: ClinicalNarrativePayload
    cross_attention_weights: Dict[str, float]


class EvaluationResponse(BaseModel):
    status: str = "success"
    classification: ClassificationOutput
    regression: RegressionOutput
    shap_attribution: Optional[SHAPAttribution] = None
    narrative: Optional[str] = None
    xai: Optional[XaiPayload] = None
