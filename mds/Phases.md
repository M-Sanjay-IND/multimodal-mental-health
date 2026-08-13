# Phases — Backend ML Pipeline (Teammate B)

> **Read this preamble every time a new phase begins.** It contains the project context, hosting strategy, branching protocol, parallel-work boundaries, and commit discipline that govern every phase below.

---

## Preamble

### Project Context

This is a **hackathon project** — a live demo in front of judges is the endgame. Every architectural and hosting decision is optimized for a working, shareable, real-time demonstration. If it can't be demoed live, it doesn't count.

### Hosting & Deployment Stack

The entire system runs on **free-tier infrastructure** with zero operational cost.

| Layer | Technology | Why |
|-------|-----------|-----|
| **Backend hosting** | **Hugging Face Spaces** (Docker SDK, 2 vCPU, 16 GB RAM) | Free, always-on, no cold starts for Docker Spaces. Judges get a live `wss://` endpoint. |
| **Backend framework** | **FastAPI + Uvicorn** | Async WebSocket gateway — sub-85ms server-side latency. |
| **Inference engine** | **ONNX Runtime (CPU, INT8)** | No GPU needed. Quantized model (~108 MB) runs fast on 2 vCPUs. |
| **Frontend hosting** | **Vercel Edge Network** | Free tier, globally distributed CDN. Teammate A deploys separately. |
| **Frontend framework** | **Next.js 15 / React 19 + TypeScript** | WebAssembly edge processing — raw video/audio never leaves the browser. |
| **Communication** | **WebSocket (chunked streaming)** | Only ~2 KB feature vectors travel over the wire, not raw media. Works on mediocre WiFi. |

**Why this works for a live demo:**

- **Shareable URL** — judges open the link on their own device and see it work instantly.
- **No "works on my machine" risk** — both ends are deployed, not running off a laptop.
- **Sub-150ms end-to-end** — feels instant during a live demo. No loading spinners.
- **Privacy-preserving by design** — camera/mic data is processed client-side and discarded. Only extracted feature vectors are transmitted.

### Git Workflow & Parallel Development Protocol

### Branching Rules

| Rule | Detail |
|------|--------|
| **Source branch** | Every phase branch is created from `dev` — never from `main`. |
| **Naming convention** | `phase-<N>/<short-kebab-slug>` (e.g., `phase-1/schemas-synthetic-pipeline`). |
| **Merge target** | Phase branches merge back into `dev` via Pull Request. `main` is untouched during development. |
| **Merge to main** | Only when the full backend pipeline is production-validated and the team explicitly decides to cut a release. |
| **Commit style** | Conventional Commits — `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`. |
| **Commit cadence** | Every logical unit of work is a commit. No monolith commits spanning multiple deliverables. |

### Parallel Development — Frontend Teammate (Teammate A)

Teammate A is building the **React 19 + TypeScript client dashboard** on the `frontend` branch independently. The two workstreams share no source files during development.

**Boundaries:**

- Teammate B owns everything under `schemas/`, `models/`, `training/`, `xai/`, `server/`, `scripts/`, `tests/`, and `data/`.
- Teammate A owns everything under `frontend/`.
- The **WebSocket contract** (`/evaluate/ws` JSON schema) defined in Phase 1 is the single integration surface. Any change to this contract must be communicated and versioned before either side merges.
- Neither teammate merges into `main` unilaterally. A joint integration pass on `dev` precedes any `main` merge.

### Phase Checklist Protocol

Before starting any phase:

1. Pull latest `dev`.
2. Create the phase branch: `git checkout -b phase-<N>/<slug> dev`.
3. Re-read this preamble.
4. Confirm the previous phase's deliverables exist and pass.

Before closing any phase:

1. All deliverables listed in the phase's **Exit Criteria** are met.
2. All new files are committed with descriptive conventional commit messages.
3. PR is opened against `dev`, reviewed, and merged.
4. `dev` is pushed to `origin`.

---

## Phase 1 — Schemas, Data Transformation & Synthetic Pipeline

**Branch:** `phase-1/schemas-synthetic-pipeline`

### Objective

Establish Pydantic v2 request/response contracts matching client payloads, build a statistically conditioned tabular feature pipeline, and generate a stratified synthetic dataset for downstream model training and validation.

### Tasks

#### 1.1 Schema Definitions (`schemas/payload.py`)

- Define Pydantic v2 `BaseModel` classes for:
  - `VisualVector` — validates $E_V \in \mathbb{R}^{128}$.
  - `AcousticVector` — validates $E_A \in \mathbb{R}^{256}$.
  - `TabularVector` — validates $\mathbf{x}_{\text{tab}} \in \mathbb{R}^{18}$.
  - `EvaluationPayload` — composite wrapper encapsulating all three.
- Enforce strict boundary checks rejecting `NaN`, `Inf`, and out-of-range values.
- Define response models for classification probabilities, regression scores, SHAP attributions, and narrative text.

#### 1.2 Tabular Feature Preprocessing (`scripts/preprocess_tabular.py`)

- **IQR Winsorization**: clamp continuous values outside $[Q_1 - 2.5 \times \text{IQR},\ Q_3 + 2.5 \times \text{IQR}]$.
- **Yeo-Johnson Power Transformation**: apply $\psi(x, \lambda)$ to skewed variables (`GSR_Level`, `Daily_App_Usage_Min`, etc.).
- **Robust Scaling**: scale using median and IQR — $x_{\text{scaled}} = \frac{x - \text{Median}(x)}{\text{IQR}(x)}$.
- Fit `PowerTransformer(method='yeo-johnson')` and `RobustScaler()` on training data.
- Serialize preprocessor artifacts: `artifacts/scaler.joblib`, `artifacts/transformer.joblib`.

#### 1.3 Multi-Task Label Generation

- **Categorical**: 4 severity classes — `Healthy` (0), `Mild` (1), `Moderate` (2), `Severe` (3).
- **Continuous targets**: Depression (0–34), Anxiety (0–24), Stress (0–39).

#### 1.4 Synthetic Data Generation (`scripts/generate_synthetic_data.py`)

- Generate 1,000 synthetic evaluation vectors using correlated Gaussian distributions conditioned on severity class distributions from `datasets/Numerics/mental_health_multimodal.csv`.
- Produce visual ($\mathbb{R}^{128}$) and acoustic ($\mathbb{R}^{256}$) synthetic embeddings.

#### 1.5 Dataset Splitting

- Export processed data into `data/train.parquet` (70%), `data/val.parquet` (15%), `data/test.parquet` (15%).
- Use `StratifiedKFold` on `Mental_Health_Status` to preserve class distribution.

### Deliverables

| Artifact | Validation |
|----------|------------|
| `schemas/payload.py` | Passes unit tests for raw array inputs, rejects `NaN`/`Inf`. |
| `artifacts/scaler.joblib`, `artifacts/transformer.joblib` | Serialized, loadable, invertible. |
| `data/train.parquet`, `data/val.parquet`, `data/test.parquet` | Zero missing values, zero infinite values, tabular columns median-centered near 0.0. |
| `scripts/generate_synthetic_data.py` | Produces 1,000 valid synthetic samples. |

### Exit Criteria

- [x] All schema validators pass with valid and invalid test payloads.
- [x] Preprocessor artifacts saved and load correctly.
- [x] Parquet splits exist with correct row ratios and zero data anomalies.
- [x] All work committed and PR merged into `dev`.

---

## Phase 2 — Dynamic Cross-Modal Attention Transformer (DCMF-Net)

**Branch:** `phase-2/dcmf-net`

### Objective

Construct the multimodal fusion engine in PyTorch — subspace projections, multi-head cross-attention, modality-aware residual gating — capable of handling degraded or missing data streams.

### Tasks

#### 2.1 Subspace Projection Layers (`models/projections.py`)

- Linear projections mapping raw dimensions (128, 256, 18) into unified embedding space $d_m = 256$:

$$E_m = \mathbf{W}_m^{\text{proj}} E_m^{\text{edge}} + \mathbf{b}_m^{\text{proj}} \quad \forall m \in \{V, A, T\}$$

#### 2.2 Multi-Head Cross-Attention (`models/attention.py`)

- $H = 4$ heads, head dimension $d_k = 64$.
- Bi-directional cross-attention: tabular queries ($Q_T$) attend to visual and acoustic key-value streams.

$$\text{Head}_h(i, j) = \text{softmax}\left( \frac{Q_i^{(h)} (K_j^{(h)})^T}{\sqrt{d_k}} \right) V_j^{(h)}$$

#### 2.3 Modality-Aware Residual Gating & Adaptive Dropout (`models/gating.py`)

- **Adaptive Modality Dropout**: randomly zero individual modality streams during training with $p_m \in [0.05, 0.20]$ (target 0.15).
- **Dynamic gating**:

$$g_m = \sigma\left( \mathbf{W}_{g,m} [E_m \| \text{MHCA}(m, \cdot)] + \mathbf{b}_{g,m} \right)$$
$$\tilde{E}_m = \text{LayerNorm}\left( E_m + g_m \odot \text{MHCA}(m, \cdot) \right)$$

#### 2.4 Fused Representation Assembly (`models/dcmf_net.py`)

- Concatenate gated vectors: $Z_{\text{fused}} = \text{Concat}(\tilde{E}_V, \tilde{E}_A, \tilde{E}_T) \in \mathbb{R}^{768}$.
- Single `DCMFNet` `nn.Module` composing all sub-modules.

#### 2.5 Unit Tests (`tests/test_dcmf_net.py`)

- Forward-pass shape assertion: output $[B, 768]$ for arbitrary batch size $B$.
- Missing-modality test: zeroed input matrices must not raise runtime errors.
- Gradient flow test: ensure gradients propagate through all branches.

### Deliverables

| Artifact | Validation |
|----------|------------|
| `models/dcmf_net.py` (+ sub-modules) | Functional PyTorch `DCMFNet` module. |
| `tests/test_dcmf_net.py` | All forward-pass and missing-input tests pass. |

### Exit Criteria

- [ ] `DCMFNet` forward pass produces $[B, 768]$ tensors.
- [ ] Zeroed-modality inputs handled gracefully.
- [ ] All work committed and PR merged into `dev`.

---

## Phase 3 — Multi-Task Neural Trunk, Loss Formulation & Training Loop

**Branch:** `phase-3/multi-task-training`

### Objective

Wire the fused representation to shared multi-task heads, implement GradNorm-balanced composite losses, train on synthetic data until convergence, and produce a checkpoint.

### Tasks

#### 3.1 Shared Trunk & Task Heads (`models/multi_task.py`)

- **Shared trunk**: two residual blocks ($768 \to 512 \to 256$) with LayerNorm, GELU, Dropout ($p = 0.30$).
- **Classification head**: $256 \to 128 \to 4$ logits → Softmax.
- **Regression head**: $256 \to 128 \to 3$ outputs → bounded Sigmoid:

$$\hat{y}_{\text{dep}} = 34 \cdot \sigma(z_{\text{dep}}),\quad \hat{y}_{\text{anx}} = 24 \cdot \sigma(z_{\text{anx}}),\quad \hat{y}_{\text{str}} = 39 \cdot \sigma(z_{\text{str}})$$

#### 3.2 Composite Loss & GradNorm (`training/losses.py`)

- **Asymmetric Focal Loss** ($\mathcal{L}_{\text{AF}}$) with $\gamma_{\text{pos}} = 1.0$, $\gamma_{\text{neg}} = 2.0$.
- **Smooth L1 Loss** ($\mathcal{L}_{\text{SL1}}$) over continuous outputs.
- **GradNorm dynamic balancing** — scale task weights $w_k(t)$ via $L_2$ gradient norms to prevent regression from overwhelming classification.

#### 3.3 Training Loop (`training/trainer.py`)

- AdamW optimizer with learning rate scheduling (cosine annealing).
- Evaluation logging: loss curves, per-task gradient norms, classification and regression metrics on val split.
- Best-checkpoint saving to `artifacts/model_state.pt`.

#### 3.4 Train on Synthetic Data

- Run training loop on parquet splits from Phase 1.
- Confirm convergence — balanced gradient magnitudes across both task branches.

### Deliverables

| Artifact | Validation |
|----------|------------|
| `models/multi_task.py` | Integrates `DCMFNet` + trunk + heads. |
| `training/losses.py` | Focal loss + Smooth L1 + GradNorm. |
| `training/trainer.py` | Full training loop with eval. |
| `artifacts/model_state.pt` | Trained checkpoint file. |

### Exit Criteria

- [ ] Multi-task model trains without NaN/Inf losses.
- [ ] Gradient magnitudes balanced across classification and regression.
- [ ] Checkpoint saved and loadable.
- [ ] All work committed and PR merged into `dev`.

---

## Phase 4 — Async FastAPI WebSocket Gateway

**Branch:** `phase-4/fastapi-gateway`

### Objective

Serve model predictions via an asynchronous FastAPI WebSocket endpoint, wiring schema validation, model inference, and JSON streaming into a production-ready gateway.

### Tasks

#### 4.1 Server Setup (`server/main.py`)

- FastAPI application managed by Uvicorn.
- WebSocket endpoint at `/evaluate/ws`.
- Non-blocking queue for real-time payload ingestion and streaming responses.

#### 4.2 Inference Pipeline Integration

- Load `model_state.pt` and preprocessor artifacts at startup.
- Parse incoming JSON against Pydantic schemas from Phase 1.
- Run `DCMFNet` → multi-task heads → return classification probabilities and regression scores.

#### 4.3 Health & Diagnostics

- `/health` REST endpoint returning server status, model load state, and uptime.
- Structured JSON logging for all WebSocket events.

#### 4.4 Integration Tests (`tests/test_server.py`)

- WebSocket round-trip test with valid payload.
- Schema rejection test with malformed payload.
- Concurrent client stress test (10 simultaneous connections).

### Deliverables

| Artifact | Validation |
|----------|------------|
| `server/main.py` | Functional WebSocket server at `ws://localhost:8000/evaluate/ws`. |
| `tests/test_server.py` | All integration tests pass. |

### Exit Criteria

- [ ] Server starts, accepts connections, returns valid predictions.
- [ ] Malformed payloads are rejected with structured errors.
- [ ] All work committed and PR merged into `dev`.

---

## Phase 5 — ONNX INT8 Quantization & Acceleration

**Branch:** `phase-5/onnx-quantization`

### Objective

Convert the trained PyTorch model to ONNX, apply dynamic INT8 quantization, and swap the server inference engine to ONNX Runtime for CPU-bound deployment.

### Tasks

#### 5.1 ONNX Export (`scripts/export_onnx.py`)

- Trace the full `MultiTaskModel` graph with dummy inputs.
- Export to `artifacts/model.onnx`.

#### 5.2 INT8 Dynamic Quantization

- Apply `onnxruntime.quantization.quantize_dynamic` to produce `artifacts/model_int8.onnx`.
- Target footprint: ~108 MB (74% reduction from FP32).

#### 5.3 ONNX Runtime Integration

- Replace PyTorch inference in `server/main.py` with `onnxruntime.InferenceSession`.
- Validate numerical parity between PyTorch and ONNX outputs (tolerance $< 1e^{-3}$).

#### 5.4 Latency Benchmarking (`tests/test_latency.py`)

- Measure single-sample inference time on CPU.
- Target: $< 45$ ms per inference.

### Deliverables

| Artifact | Validation |
|----------|------------|
| `artifacts/model_int8.onnx` | ~108 MB quantized model. |
| `scripts/export_onnx.py` | Reproducible export pipeline. |
| `tests/test_latency.py` | Inference $< 45$ ms on CPU. |

### Exit Criteria

- [ ] ONNX INT8 model produces numerically equivalent outputs to PyTorch.
- [ ] Inference latency target met.
- [ ] Server updated to use ONNX Runtime.
- [ ] All work committed and PR merged into `dev`.

---

## Phase 6 — FastSHAP Explainer & Deterministic Narrative Engine

**Branch:** `phase-6/xai-narrative`

### Objective

Build the explainability layer — a FastSHAP surrogate network for tabular attributions and a deterministic clinical narrative generator that eliminates LLM hallucination risk.

### Tasks

#### 6.1 FastSHAP Explainer Network (`xai/fast_shap.py`)

- Train a lightweight surrogate $S_\theta(\mathbf{x}_{\text{tab}})$ estimating Shapley values for all 18 tabular features in a single forward pass ($< 15$ ms).
- Enforce Shapley efficiency constraint:

$$\sum_{i=1}^{18} \phi_i = f(\mathbf{x}_{\text{tab}}) - \mathbb{E}[f(\mathbf{X})]$$

- Serialize trained explainer to `artifacts/fastshap.pt`.

#### 6.2 Deterministic Narrative Engine (`xai/narrative_engine.py`)

- Pure-Python clinical rule graph — no LLM, no stochastic generation.
- **Severity cutoff matrix**:
  - Minimal: Depression $\le 9$, Anxiety $\le 7$, Stress $\le 14$.
  - Mild: Depression 10–13, Anxiety 8–9, Stress 15–18.
  - Moderate: Depression 14–20, Anxiety 10–14, Stress 19–25.
  - Severe: Depression $\ge 21$, Anxiety $\ge 15$, Stress $\ge 26$.
- Rank top 3 positive risk contributors ($+\phi_i$) and top 2 protective markers ($-\phi_i$).
- Parameterized string interpolation into clinically validated templates in $< 5$ ms.

#### 6.3 XAI Integration into Server

- Wire FastSHAP and narrative engine into the WebSocket response pipeline.
- Response JSON includes: classification, regression scores, SHAP attributions, and narrative text.

#### 6.4 Tests (`tests/test_xai.py`)

- Shapley efficiency assertion.
- Narrative determinism test — same inputs always produce identical narrative strings.
- Latency assertions: FastSHAP $< 15$ ms, narrative $< 5$ ms.

### Deliverables

| Artifact | Validation |
|----------|------------|
| `xai/fast_shap.py` | Trained explainer, single-pass attributions. |
| `xai/narrative_engine.py` | Zero-hallucination deterministic narratives. |
| `artifacts/fastshap.pt` | Serialized explainer weights. |
| `tests/test_xai.py` | All efficiency, determinism, and latency tests pass. |

### Exit Criteria

- [ ] FastSHAP attributions satisfy efficiency constraint.
- [ ] Narrative engine is fully deterministic with $< 5$ ms latency.
- [ ] Server returns complete evaluation response (predictions + explanations + narrative).
- [ ] All work committed and PR merged into `dev`.

---

## Phase 7 — Containerization & Serverless Deployment

**Branch:** `phase-7/deployment`

### Objective

Containerize the full backend, deploy to Hugging Face Spaces, and validate production readiness under load.

### Tasks

#### 7.1 Dockerfile

- Python 3.11 slim base.
- Install ONNX Runtime CPU, FastAPI, Uvicorn, and all pipeline dependencies.
- Copy model artifacts, preprocessor artifacts, and server code.
- Expose port 7860 (HF Spaces default).

#### 7.2 Hugging Face Spaces Deployment

- Push Docker container to Hugging Face Spaces (2 vCPU, 16 GB RAM).
- Configure WebSocket routing via `app.py` / `README.md` metadata.
- Verify live endpoint: `wss://<space-url>/evaluate/ws`.

#### 7.3 Latency Budget Validation

| Pipeline Stage | Target |
|----------------|--------|
| Acoustic processing | 20 ms |
| ONNX INT8 inference | 45 ms |
| FastSHAP attribution | 15 ms |
| Narrative synthesis | 5 ms |
| **Total server-side** | **< 85 ms** |

#### 7.4 Load Testing (`tests/test_benchmark.py`)

- Simulate 50 concurrent WebSocket clients using `locust` or `asyncio`.
- Assert total round-trip server latency stays under budget.

### Deliverables

| Artifact | Validation |
|----------|------------|
| `Dockerfile` | Builds and runs locally. |
| Live HF Spaces endpoint | `wss://` accepting real-time evaluations. |
| `tests/test_benchmark.py` | Latency $< 85$ ms under 50 concurrent clients. |

### Exit Criteria

- [ ] Container builds cleanly.
- [ ] Live endpoint accessible and functional.
- [ ] Latency budget met under concurrent load.
- [ ] All work committed and PR merged into `dev`.

---

## Phase 8 — Benchmark Validation & Model Performance Certification

**Branch:** `phase-8/benchmark-validation`

### Objective

Run the full evaluation suite against the test split and certify that model performance meets or exceeds all target benchmarks before declaring production readiness.

### Tasks

#### 8.1 Classification Metrics Validation

| Metric | Target |
|--------|--------|
| Classification Accuracy | $\ge 93.6\%$ |
| Macro F1-Score | $\ge 0.924$ |
| ROC-AUC Score | $\ge 0.978$ |

#### 8.2 Regression Metrics Validation

| Metric | Target |
|--------|--------|
| MAE (Depression) | $\le 1.08$ |
| MAE (Anxiety) | $\le 0.82$ |
| MAE (Stress) | $\le 1.15$ |
| Overall $R^2$ | $\ge 0.931$ |

#### 8.3 Benchmark Report (`reports/benchmark_report.md`)

- Generate automated report with all metric values, comparison against targets, latency profiles, and pass/fail status.
- Include confusion matrix, per-class F1 breakdown, and regression scatter plots.

#### 8.4 Cross-Validation Protocol Verification

- Audio: `GroupKFold` ($K=5$) by `Actor_ID`.
- Visual: `StratifiedKFold` ($K=5$) by emotion class.
- Tabular: 70/15/15 stratified by `Mental_Health_Status`.

### Deliverables

| Artifact | Validation |
|----------|------------|
| `reports/benchmark_report.md` | All metrics meet targets. |
| `tests/test_benchmark.py` (extended) | Automated metric assertions. |

### Exit Criteria

- [ ] All classification and regression metrics meet or exceed targets.
- [ ] Cross-validation protocol verified (no data leakage).
- [ ] Benchmark report generated and committed.
- [ ] All work committed and PR merged into `dev`.

---

## Phase 9 — Integration Testing, Documentation & Release Preparation

**Branch:** `phase-9/integration-release`

### Objective

End-to-end integration test with the frontend team's dashboard, finalize documentation, and prepare `dev` for merge into `main`.

### Tasks

#### 9.1 Frontend Integration Test

- Coordinate with Teammate A to connect the React dashboard to the live backend WebSocket.
- Validate full round-trip: client edge extraction → WebSocket payload → server inference → response rendering.
- Confirm schema contract compatibility.

#### 9.2 Documentation

- Update `README.md` with architecture overview, setup instructions, deployment guide, and API reference.
- Ensure all modules have inline documentation where non-obvious.

#### 9.3 Release Preparation

- Final code review across all phase PRs.
- Tag `dev` with a release candidate version.
- Open PR from `dev` → `main` for final team review.

### Deliverables

| Artifact | Validation |
|----------|------------|
| Passing end-to-end integration test | Frontend ↔ Backend round-trip works. |
| `README.md` | Comprehensive, production-grade documentation. |
| `dev` → `main` PR | Ready for final merge. |

### Exit Criteria

- [ ] End-to-end integration with frontend verified.
- [ ] Documentation complete and reviewed.
- [ ] `dev` → `main` PR opened and approved.
- [ ] `main` merge executed. Repository tagged with release version.

---

## Phase Dependency Graph

```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6 ──► Phase 7 ──► Phase 8 ──► Phase 9
  │                                    │                        │                        │           │
  │         Schemas & Data             │   Training & Server    │    XAI & Narrative      │  Metrics  │
  │                                    │                        │                        │           │
  └────────────────────────────────────┴────────────────────────┴────────────────────────┴───────────┘
                                                                                                    │
                                                                                          dev → main merge
                                                                                        (joint with Teammate A)
```

---

## Quick Reference — Branch Cheatsheet

| Phase | Branch Name | Core Deliverable |
|-------|-------------|------------------|
| 1 | `phase-1/schemas-synthetic-pipeline` | Pydantic schemas, preprocessors, parquet splits |
| 2 | `phase-2/dcmf-net` | Cross-modal attention transformer |
| 3 | `phase-3/multi-task-training` | Multi-task heads, GradNorm losses, trained checkpoint |
| 4 | `phase-4/fastapi-gateway` | Async WebSocket server |
| 5 | `phase-5/onnx-quantization` | INT8 ONNX model, CPU-optimized inference |
| 6 | `phase-6/xai-narrative` | FastSHAP explainer, deterministic narrative engine |
| 7 | `phase-7/deployment` | Docker container, HF Spaces live endpoint |
| 8 | `phase-8/benchmark-validation` | Benchmark report, metric certification |
| 9 | `phase-9/integration-release` | Frontend integration, docs, `main` merge |

---

> **Reminder**: This file is the source of truth for phase execution order and branching discipline. Re-read the **Preamble** at the start of every phase.
