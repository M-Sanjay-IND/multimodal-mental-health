# 🧠 DCMF-Net — Multimodal Psychiatric Evaluation & Severity Estimation Platform

[![Streamlit](https://img.shields.io/badge/Streamlit-1.51%2B-FF4B4B.svg)](https://streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end clinical-grade AI platform integrating **Dynamic Cross-Modal Attention Fusion (DCMF-Net)**, **FastSHAP Explainable AI (XAI)**, and a zero-hallucination **Clinical Narrative Engine** for real-time mental health assessment across visual, acoustic, and mobile biometric streams.

---

## 🎨 Ultra-Minimalist Research Workspace (`app.py`)

The application features a high-contrast, modern black and white research workspace built in Streamlit (`app.py`):
- **Direct PyTorch Execution**: Runs `MultiTaskModel`, `FastSHAPExplainer`, and `ClinicalNarrativeEngine` directly in Python without client-side state wrappers or browser bloat.
- **Multimodal Embedding Derivation (`derive_multimodal_vectors`)**: Dynamically projects 18 biomarker inputs into 128D visual and 256D acoustic latent embeddings to prevent Transformer fusion zero-collapse when media streams are unprovided.
- **Interactive Benchmark Scenario Presets**:
  - `Optimal Healthy Baseline` (Healthy 99.5%)
  - `Mild Work Stress & Fatigue` (Mild Stress 98.6%)
  - `Moderate Depressive Affect` (Moderate Stress 97.3%)
  - `Severe Crisis & Agitation` (Severe/Moderate Stress 99.9%)
- **PyTorch Model Inference Output**:
  - Categorical Severity Classification (`Healthy`, `Mild_Stress`, `Moderate_Stress`, `Severe_Stress`) with Softmax Probabilities Bar Chart.
  - Continuous Symptom Regression Scores: Depression (PHQ-9 0-34), Anxiety (GAD-7 0-24), Stress (PSS 0-39).
  - FastSHAP Feature Impact Attributions Dataframe.
  - Deterministic Clinical Narrative Synthesis Summary.

---

## 🏗️ Architecture Overview

```mermaid
graph TD
    subgraph Streamlit Research Interface [app.py Workspace]
        Sliders[18 Biomarker Sliders / Benchmark Presets] --> Deriver[derive_multimodal_vectors]
        Deriver --> VisVec[128D Visual Embedding E_V]
        Deriver --> AcVec[256D Acoustic Embedding E_A]
        Deriver --> TabVec[18D Tabular Vector x_tab]
    end

    subgraph FastAPI REST Endpoint [http://localhost:8000/evaluate/rest]
        REST[FastAPI Gateway]
    end

    subgraph PyTorch Inference Engine [artifacts/model_state.pt]
        VisVec --> DCMF[DCMF-Net Cross-Modal Transformer]
        AcVec --> DCMF
        TabVec --> DCMF

        DCMF --> Trunk[Shared Neural Trunk 768 → 256]
        Trunk --> ClsHead[Classification Head — 4 Severity Classes]
        Trunk --> RegHead[Regression Head — Depression / Anxiety / Stress Scores]

        ClsHead --> FastSHAP[FastSHAP Feature Attribution Engine]
        RegHead --> FastSHAP
        FastSHAP --> Narrative[Deterministic Clinical Narrative Engine]
    end
```

---

## ⚡ Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Streamlit Research Workspace
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### 3. Launch FastAPI Server (Optional REST Endpoint)
```bash
cd server
python main.py
```
FastAPI server will listen on **`http://localhost:8000`**.

---

## 🔬 Repository Structure

```
multimodal-mental-health/
├── app.py                      # Main Streamlit research workspace (Black & White Theme)
├── artifacts/                  # Trained FP32 PyTorch checkpoint (model_state.pt) & RobustScaler
├── models/                     # PyTorch DCMF-Net model architecture
│   ├── dcmf_net.py            # Dynamic Cross-Modal Transformer Fusion
│   └── multi_task.py           # Shared trunk, Classification, & Regression heads
├── server/                     # FastAPI server (main.py)
├── schemas/                    # Pydantic payloads & feature definitions
├── xai/                        # FastSHAP explainer & Clinical Narrative Engine
├── reports/                    # Model Evaluation Benchmark Report (model_evaluation_report.md)
└── requirements.txt            # Python dependencies
```

---

## 📄 License
Distributed under the MIT License.
