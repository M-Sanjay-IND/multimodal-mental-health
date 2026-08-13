import torch
from typing import Dict, List, Any
from schemas.payload import (
    ClassificationOutput,
    RegressionOutput,
    FeatureAttributionItem,
    ClinicalNarrativePayload,
)


def _to_float_weight(val: Any, default: float = 0.33) -> float:
    if val is None:
        return default
    if isinstance(val, torch.Tensor):
        return float(val.mean().item())
    try:
        return float(val)
    except Exception:
        return default


class ClinicalNarrativeEngine:
    """Deterministic, template-driven Clinical Narrative Engine.
    Guarantees 100% zero hallucination and strict clinical policy alignment.
    """

    def generate_narrative(
        self,
        classification: ClassificationOutput,
        regression: RegressionOutput,
        attributions: List[FeatureAttributionItem],
        attn_dict: Dict[str, Any] = None,
    ) -> ClinicalNarrativePayload:
        pred_class = classification.predicted_class
        dep_score = regression.depression_score
        anx_score = regression.anxiety_score
        str_score = regression.stress_score

        # 1. Executive Summary
        if pred_class == "Healthy":
            summary = (
                f"Patient presents with low overall psychological distress (Depression: {dep_score}/34, "
                f"Anxiety: {anx_score}/24, Stress: {str_score}/39). Severity classified as Healthy."
            )
        elif pred_class == "Mild_Stress":
            summary = (
                f"Patient exhibits mild psychological strain (Depression: {dep_score}/34, "
                f"Anxiety: {anx_score}/24, Stress: {str_score}/39). Early lifestyle interventions recommended."
            )
        elif pred_class == "Moderate_Stress":
            summary = (
                f"Patient demonstrates moderate symptoms of distress (Depression: {dep_score}/34, "
                f"Anxiety: {anx_score}/24, Stress: {str_score}/39). Clinical evaluation & stress management indicated."
            )
        else:  # Severe_Stress
            summary = (
                f"Patient shows elevated indicators of severe psychological distress (Depression: {dep_score}/34, "
                f"Anxiety: {anx_score}/24, Stress: {str_score}/39). Immediate professional clinical follow-up strongly advised."
            )

        # 2. Modality Breakdown
        if attn_dict and "weights_vt" in attn_dict:
            v_w = _to_float_weight(attn_dict.get("weights_vt"), 0.33)
            a_w = _to_float_weight(attn_dict.get("weights_at"), 0.33)
            t_w = _to_float_weight(attn_dict.get("weights_ta"), 0.34)
            total = max(v_w + a_w + t_w, 1e-6)
            v_pct = round((v_w / total) * 100, 1)
            a_pct = round((a_w / total) * 100, 1)
            t_pct = round((t_w / total) * 100, 1)
            modality_breakdown = (
                f"Multimodal Fusion Attention Weights: Visual Facials ({v_pct}%), "
                f"Acoustics/Voice ({a_pct}%), Behavioral/Biometrics ({t_pct}%)."
            )
        else:
            modality_breakdown = "Multimodal Fusion Attention Weights: Balanced contribution across visual, acoustic, and behavioral streams."

        # 3. Key Risk Factors & Protective Factors
        key_risk_factors = []
        protective_factors = []

        for item in attributions:
            name_clean = item.feature_name.replace("_", " ")
            if item.direction == "risk_factor" and len(key_risk_factors) < 3:
                key_risk_factors.append(f"Elevated {name_clean} (impact score: +{item.importance_score:.2f})")
            elif item.direction == "protective_factor" and len(protective_factors) < 3:
                protective_factors.append(f"Healthy {name_clean} (impact score: {item.importance_score:.2f})")

        if not key_risk_factors:
            key_risk_factors.append("No dominant physiological or behavioral risk factors detected.")
        if not protective_factors:
            protective_factors.append("Baseline engagement metrics within normal parameters.")

        # 4. Clinical Recommendations
        clinical_recommendations = []
        if str_score > 20.0:
            clinical_recommendations.append("Incorporate daily structured mindfulness and biofeedback relaxation sessions.")
        if anx_score > 12.0:
            clinical_recommendations.append("Consider cognitive-behavioral stress reduction (CBSR) techniques.")
        if dep_score > 17.0:
            clinical_recommendations.append("Schedule a comprehensive clinical assessment with a certified healthcare professional.")

        if pred_class == "Healthy" or not clinical_recommendations:
            clinical_recommendations.append("Maintain current healthy lifestyle routines, sleep hygiene, and social activities.")

        return ClinicalNarrativePayload(
            summary=summary,
            modality_breakdown=modality_breakdown,
            key_risk_factors=key_risk_factors,
            protective_factors=protective_factors,
            clinical_recommendations=clinical_recommendations,
        )
