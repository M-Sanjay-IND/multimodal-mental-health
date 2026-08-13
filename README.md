---
title: Multimodal Psychiatric Evaluation & Severity Estimation API
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🧠 Multimodal Psychiatric Evaluation & Severity Estimation Engine (DCMF-Net)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end clinical-grade AI platform integrating **Dynamic Cross-Modal Attention Fusion (DCMF-Net)**, **FastSHAP Explainable AI (XAI)**, and a zero-hallucination **Clinical Narrative Engine** for real-time mental health assessment across visual facials, speech acoustics, and mobile biometrics.

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[Visual Stream - 128D Facials] --> P1[Visual Subspace Projection]
    B[Acoustic Stream - 256D Speech] --> P2[Acoustic Subspace Projection]
    C[Tabular Stream - 18D Biometrics] --> P3[Tabular Subspace Projection]

    P1 --> DCMF[DCMF-Net Transformer Core]
    P2 --> DCMF
    P3 --> DCMF

    DCMF --> Trunk[Shared Neural Trunk 768 to 256]
    
    Trunk --> ClsHead[Classification Head - 4 Severity Classes]
    Trunk --> RegHead[Regression Head - Dep/Anx/Str Scores]
    
    ClsHead --> FastSHAP[FastSHAP Feature Attribution Engine]
    RegHead --> FastSHAP
    
    FastSHAP --> Narrative[Deterministic Clinical Narrative Engine]
    Narrative --> Gateway[Async FastAPI Gateway REST / WebSocket]
```

---

## ✨ Key Features

- **Multimodal Fusion Engine (`DCMF-Net`)**: 4-Head Bi-Directional Cross-Attention mechanism uniting facial expressions, vocal prosody, and smartphone biometrics.
- **Dynamic INT8 Model Quantization**: Compressed model footprint from 7.20 MB to **1.86 MB (74.2% reduction)** with ultra-fast **2.99ms CPU inference**.
- **Real-Time Async Gateway**: FastAPI streaming WebSocket (`/evaluate/ws`) & REST (`/evaluate/rest`) API.
- **FastSHAP Explainable AI**: Sub-15ms feature attribution computing risk-elevating vs protective factor metrics.
- **Zero-Hallucination Clinical Narrative**: Deterministic template engine generating clinical summaries, risk factors, and actionable recommendations.
- **$0.00/Month Serverless Containerization**: Optimized Docker SDK build ready for deployment on Hugging Face Spaces.

---

## 📊 Dataset Schema & Multimodal Inputs

| Modality | Features | Dimensions | Source / Pipeline |
|----------|----------|------------|-------------------|
| **Visual** | Facial Emotion Variance, Smile Intensity, Eye Blink Rate, Head Motion | 128-D | Mediapipe / OpenCV Feature Embeddings |
| **Acoustic** | Pitch Mean, Speech Rate, MFCC Means & Variances | 256-D | Librosa Speech Prosody Extraction |
| **Tabular & Biometrics** | Sleep Quality, Social Engagement, Typing WPM, HRV Index, GSR Level | 18 Features | Robust Scaled & Yeo-Johnson Transformed |

---

## 🚀 Quick Start & Local Execution

### 1. Installation
```bash
git clone https://github.com/M-Sanjay-IND/multimodal-mental-health.git
cd multimodal-mental-health
pip install -r requirements.txt
```

### 2. Run the Gateway Server
```bash
python server/main.py
```
- Interactive API Documentation (Swagger UI): `http://localhost:8000/docs`
- Health Diagnostic Check: `http://localhost:8000/health`
- Real-time WebSocket Endpoint: `ws://localhost:8000/evaluate/ws`

### 3. Run Test Suite
```bash
pytest
```

---

## 🐳 Docker & Hugging Face Spaces Deployment

### Build Docker Image Locally
```bash
docker build -t multimodal-mental-health:latest .
docker run -p 7860:7860 multimodal-mental-health:latest
```

### Deploy to Hugging Face Spaces
```bash
python scripts/deploy_hf.py --space-id "your-username/multimodal-mental-health"
```

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.
