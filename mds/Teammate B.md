## Phase 1: Schemas, Data Transformation & Synthetic Pipeline

### Objective

Establish Pydantic v2 request/response contracts matching client payloads, build a statistically conditioned tabular feature pipeline, and generate a stratified synthetic dataset to train and validate downstream models.

  

### Detailed Specifications

1. **Schema Definitions (`schemas/payload.py`)**:
    
      
    - Vector dimension verification: $E_V \in \mathbb{R}^{128}$ (Visual), $E_A \in \mathbb{R}^{256}$ (Acoustic), and $\mathbf{x}_{\text{tab}} \in \mathbb{R}^{18}$ (Tabular).
        
          
        
    - Strict boundary checks preventing non-numeric (`NaN`/`Inf`) values.
        
          
        
2. **Tabular Feature Preprocessing**:
    
      
    - **IQR Winsorization**: Clamp continuous values outside $[Q_1 - 2.5 \times \text{IQR}, Q_3 + 2.5 \times \text{IQR}]$.
        
          
        
    - **Yeo-Johnson Power Transformation**: Apply $\psi(x, \lambda)$ to non-Gaussian variables (e.g., $\text{GSR\_Level}$, $\text{Daily\_App\_Usage\_Min}$) to stabilize variance:
        
          
        
        $$\psi(x, \lambda) = \begin{cases} \frac{(x + 1)^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0, x \ge 0 \\ \log(x + 1) & \text{if } \lambda = 0, x \ge 0 \end{cases}$$
        
    - **Robust Scaling**: Scale variables using median and IQR: $x_{\text{scaled}} = \frac{x - \text{Median}(x)}{\text{IQR}(x)}$.
        
          
        
3. **Multi-Task Label Generation**:
    
      
    - **Categorical**: 4 severity classes (`Healthy`: 0, `Mild`: 1, `Moderate`: 2, `Severe`: 3).
        
          
        
    - **Continuous Target Ranges**: Depression score ($0\text{--}34$), Anxiety score ($0\text{--}24$), Stress score ($0\text{--}39$).
        
          
        

### Execution Procedure

1. Define Pydantic models for individual vectors and the combined payload wrapper.
    
      
    
2. Write `scripts/generate_synthetic_data.py` to produce $1,000$ synthetic evaluation vectors using correlated Gaussian distributions.
    
      
    
3. Fit `PowerTransformer(method='yeo-johnson')` and `RobustScaler()` on training data, then serialize the preprocessor artifacts using `joblib`.
    
      
    
4. Export the processed dataset into `train.parquet` ($70\%$), `val.parquet` ($15\%$), and `test.parquet` ($15\%$) using `StratifiedKFold` on target mental health classes.
    
      
    

### Phase Deliverables & Success Metrics

- `schemas/payload.py` passes validation tests for raw array inputs.
    
      
    
- Saved transformation artifacts (`scaler.joblib`, `transformer.joblib`).
    
      
    
- `.parquet` dataset splits with zero missing values, zero infinite values, and a median centered near $0.0$ across tabular columns.
    
      
    

## Phase 2: Dynamic Cross-Modal Attention Transformer (DCMF-Net)

```
       Query (Q_T) ──┐
       Key   (K_V) ──┼──► Scaled Dot-Product ──► Attention Weights (α_VT) ──┐
       Value (V_V) ──┘                                                      │
                                                                            ├──► Gated Residual (MARG) ──► Fused Vector Z
       Query (Q_T) ──┐                                                      │
       Key   (K_A) ──┼──► Scaled Dot-Product ──► Attention Weights (α_AT) ──┘
       Value (V_A) ──┘
```

### Objective

Construct the multimodal fusion engine in PyTorch that projects heterogeneous input vectors into a unified subspace, calculates cross-modal attention, and handles degraded/missing data streams via residual gating.

  

### Detailed Specifications

1. **Subspace Projection**:
    
      
    - Linear projections mapping raw dimensions ($128, 256, 18$) into unified embedding space $d_m = 256$:
        
          
        
        $$E_m = \mathbf{W}_m^{\text{proj}} E_m^{\text{edge}} + \mathbf{b}_m^{\text{proj}} \quad \forall m \in \{V, A, T\}$$
        
          
        
2. **Multi-Head Cross-Attention (MHCA)**:
    
      
    - $H = 4$ attention heads with head dimension $d_k = d_m / H = 64$.
        
          
        
    - Bi-directional cross-attention between tabular queries ($Q_T$) and visual/acoustic key-value streams ($K_V, V_V, K_A, V_A$):
        
          
        
        $$\text{Head}_h(i, j) = \text{softmax}\left( \frac{Q_i^{(h)} (K_j^{(h)})^T}{\sqrt{d_k}} \right) V_j^{(h)}$$
        
          
        
3. **Modality-Aware Residual Gating (MARG) & Adaptive Dropout**:
    
      
    - **Adaptive Modality Dropout**: Randomly zero out individual input modalities during training with probability $p_m \in [0.05, 0.20]$ (target $0.15$).
        
          
        
    - **Dynamic Gating**:
        
          
        
        $$g_m = \sigma\left( \mathbf{W}_{g, m} [E_m \parallel \text{MHCA}(m, \cdot)] + \mathbf{b}_{g, m} \right)$$
        
          
        
        $$\tilde{E}_m = \text{LayerNorm}\left( E_m + g_m \odot \text{MHCA}(m, \cdot) \right)$$
        
          
        
4. **Fused Representation**:
    
      
    - Concatenate gated vectors into $Z_{\text{fused}} = \text{Concat}(\tilde{E}_V, \tilde{E}_A, \tilde{E}_T) \in \mathbb{R}^{768}$.
        
          
        

### Execution Procedure

1. Create `models/dcmf_net.py` containing PyTorch `nn.Module` classes for projection layers, multi-head cross-attention, and MARG blocks.
    
      
    
2. Implement custom training hooks for Adaptive Modality Dropout to simulate missing client streams.
    
      
    
3. Write forward-pass unit tests confirming output shape $[B, 768]$ for arbitrary batch size $B$.
    
      
    

### Phase Deliverables & Success Metrics

- Functional PyTorch `DCMFNet` model module.
    
      
    
- Successful forward pass tests handling missing inputs (zeroed matrices) without throwing runtime matrix mismatch errors.
    
      
    

## Phase 3: Multi-Task Neural Trunk, Loss Formulation & Async Gateway

```
                             Fused Latent Vector (Z_fused ∈ ℝ^768)
                                               │
                                               v
                                  Shared Deep Neural Trunk
                               (LayerNorm + GELU + Residuals)
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       │                                               │
                       v                                               v
        Classification Task Head                       Regression Task Head
      Dense (768 -> 256 -> 128 -> 4)                 Dense (768 -> 256 -> 128 -> 3)
```

### Objective

Connect the fused representation to shared multi-task neural network heads, train using dynamically balanced multi-task losses (GradNorm), and serve predictions via an asynchronous FastAPI WebSocket gateway.

  

### Detailed Specifications

1. **Shared Neural Trunk & Task Heads**:
    
      
    - **Shared Trunk**: Two residual blocks ($768 \rightarrow 512 \rightarrow 256$) with Layer Normalization, GELU activations, and Dropout ($p = 0.30$).
        
          
        
    - **Classification Head**: $256 \rightarrow 128 \rightarrow 4$ logits converted to class probabilities $\hat{p}_c$ via Softmax.
        
          
        
    - **Regression Head**: $256 \rightarrow 128 \rightarrow 3$ outputs scaled by Sigmoid bounds:
        
          
        
        $$\hat{y}_{\text{dep}} = 34 \cdot \sigma(z_{\text{dep}}), \quad \hat{y}_{\text{anx}} = 24 \cdot \sigma(z_{\text{anx}}), \quad \hat{y}_{\text{str}} = 39 \cdot \sigma(z_{\text{str}})$$
        
          
        
2. **Composite Loss Functions & GradNorm**:
    
      
    - **Asymmetric Focal Loss ($\mathcal{L}_{\text{AF}}$)**:
        
          
        
        $$\mathcal{L}_{\text{AF}} = -\sum_{c=1}^{4} \alpha_c (1 - \hat{p}_c)^{\gamma_c} \log(\hat{p}_c) \quad (\gamma_{\text{pos}} = 1.0, \, \gamma_{\text{neg}} = 2.0)$$
        
          
        
    - **Smooth L1 Loss ($\mathcal{L}_{\text{SL1}}$)**: Calculated over continuous outputs $(\hat{y}_{\text{dep}}, \hat{y}_{\text{anx}}, \hat{y}_{\text{str}})$.
        
          
        
    - **GradNorm Dynamic Balancing**: Scale task weights $w_k(t)$ based on $L_2$ gradient norms $G_W^{(k)}(t) = \Vert \nabla_W w_k(t) \mathcal{L}_k(t) \Vert_2$ relative to mean norm $\bar{G}_W(t)$ to prevent regression gradients from overwhelming classification backpropagation.
        
          
        
3. **Async Web Service (`server/main.py`)**:
    
      
    - FastAPI async gateway managed by Uvicorn, listening on `/evaluate/ws`.
        
          
        
    - Non-blocking queue handling real-time payload ingestion and streaming model responses.
        
          
        

### Execution Procedure

1. Build `models/multi_task.py` joining `DCMFNet` with the shared trunk and prediction heads.
    
      
    
2. Build `training/trainer.py` implementing the training loop with GradNorm loss scaling, AdamW optimizer, and evaluation logging.
    
      
    
3. Train the PyTorch model on synthetic parquet data until convergence.
    
      
    
4. Implement `server/main.py` setting up the FastAPI WebSocket endpoint with JSON parsing and model execution loops.
    
      
    

### Phase Deliverables & Success Metrics

- Trained PyTorch checkpoint file `model_state.pt`.
    
      
    
- Functional FastAPI WebSocket server running locally on `ws://localhost:8000/evaluate/ws`.
    
      
    
- Multi-task training convergence showing balanced gradient magnitudes across classification and regression branches.
    
      
    

## Phase 4: ONNX INT8 Acceleration, FastSHAP & Deterministic Narrative Engine

```
[Target Probabilities & Continuous Scores]
                   │
                   ├─► FastSHAP Matrix Network ──────► Tabular Feature Attributions (ϕ_i)
                   ├─► Action Unit Grad-CAM ─────────► Spatial Facial Saliency Heatmap
                   └─► Attention Weight Extractor ───► Cross-Modal Interaction Ratios
                                   │
                                   v
             ┌───────────────────────────────────────────┐
             │    DETERMINISTIC CLINICAL RULE GRAPH      │
             │   - Severity Cutoff Logic Matrix          │
             │   - Indicator Rank & Attribution Selector │
             │   - Structured Clinical Narrative Mapping │
             └─────────────────────┬─────────────────────┘
                                   │
                                   v
             Verified, Non-Hallucinating Clinical Evaluation Report
```

### Objective

Quantize the PyTorch model to INT8 ONNX format to meet CPU latency constraints, construct a fast tabular explainer network, and build a deterministic narrative generator that eliminates LLM hallucinations.

  

### Detailed Specifications

1. **ONNX INT8 Compilation**:
    
      
    - Convert PyTorch model to ONNX format and apply dynamic 8-bit quantization via `onnxruntime.quantization`:
        
          
        
        $$\mathbf{W}_{\text{quant}} = \text{round}\left( \frac{\mathbf{W}_{\text{fp32}}}{\text{Scale}} \right) + \text{ZeroPoint}$$
        
          
        
    - Model footprint reduction target: $74\%$ ($420\text{ MB} \rightarrow 108\text{ MB}$).
        
          
        
    - Target model inference execution time: $< 45\text{ ms}$ on 2 vCPU serverless instances.
        
          
        
2. **FastSHAP Matrix Explainer (`xai/fast_shap.py`)**:
    
      
    - Train a lightweight explicit surrogate explainer network $S_\theta(\mathbf{x}_{\text{tab}})$ estimating exact Shapley values $\phi_i$ for all 18 tabular features in a single pass ($< 15\text{ ms}$):
        
          
        
        $$\sum_{i=1}^{18} \phi_i = f(\mathbf{x}_{\text{tab}}) - \mathbb{E}[f(\mathbf{X})]$$
        
          
        
3. **Deterministic Narrative Engine (`xai/narrative_engine.py`)**:
    
      
    - Implement a non-LLM clinical rule graph using deterministic cutoff logic:
        
          
        - **Minimal**: Depression $\le 9$, Anxiety $\le 7$, Stress $\le 14$
            
              
            
              
            
        - **Mild**: Depression $10\text{--}13$, Anxiety $8\text{--}9$, Stress $15\text{--}18$
            
              
            
              
            
        - **Moderate**: Depression $14\text{--}20$, Anxiety $10\text{--}14$, Stress $19\text{--}25$
            
              
            
              
            
        - **Severe**: Depression $\ge 21$, Anxiety $\ge 15$, Stress $\ge 26$
            
              
            
              
            
    - Rank top 3 positive risk contributors ($+\phi_i$) and top 2 protective markers ($-\phi_i$), interpolating values into predefined clinical templates in $< 5\text{ ms}$.
        
          
        

### Execution Procedure

1. Run `scripts/export_onnx.py` to trace PyTorch model graphs, export to `.onnx`, and execute dynamic INT8 quantization.
    
      
    
2. Train the FastSHAP explainer network against tabular feature outputs and verify Shapley efficiency constraints.
    
      
    
3. Construct `xai/narrative_engine.py` using pure Python string templates and severity matrices.
    
      
    
4. Integrate ONNX model runtime, FastSHAP, and narrative generation into the FastAPI WebSocket handler.
    
      
    

### Phase Deliverables & Success Metrics

- Quantized `model_int8.onnx` file ($108\text{ MB}$).
    
      
    
- FastSHAP output vector providing local attributions for 18 tabular variables.
    
      
    
- Verified, zero-hallucination structured narrative string output returned in $< 5\text{ ms}$.
    
      
    

## Phase 5: Integration, Serverless Deployment & Benchmark Validation

### Objective

Deploy the backend to Hugging Face Spaces via Docker and validate system performance against clinical accuracy targets and latency budgets under concurrent load.

  

### Detailed Specifications

1. **Serverless Deployment Setup**:
    
      
    - Containerize FastAPI application, ONNX INT8 runtime dependencies, preprocessor artifacts, and narrative engines using Docker.
        
          
        
    - Host on Hugging Face Spaces (2 vCPU cores, 16 GB RAM).
        
          
        
2. **Server Latency Target Budget**:
    
      
    - Acoustic processing execution: $20\text{ ms}$
        
          
        
          
        
    - Multimodal ONNX INT8 inference: $45\text{ ms}$
        
          
        
          
        
    - FastSHAP feature attribution calculation: $15\text{ ms}$
        
          
        
          
        
    - Dynamic narrative synthesis: $5\text{ ms}$
        
          
        
          
        
    - **Total Server-Side Processing Latency**: **$< 85\text{ ms}$**
        
          
        
          
        
3. **Model Benchmark Validation Matrix**:
    
      
    

|**Performance Dimension**|**Target Metric MD**|
|---|---|
|**Classification Accuracy**|$\ge 93.6\%$|
|**Macro F1-Score**|$\ge 0.924$|
|**ROC-AUC Score**|$\ge 0.978$|
|**MAE (Depression Score)**|$\le 1.08$|
|**MAE (Anxiety Score)**|$\le 0.82$|
|**MAE (Stress Score)**|$\le 1.15$|
|**Overall $R^2$ Score**|$\ge 0.931$|

### Execution Procedure

1. Write a optimized `Dockerfile` leveraging Python 3.11 slim base images and ONNX Runtime CPU packages.
    
      
    
2. Deploy the container to Hugging Face Spaces and configure WebSocket routing.
    
      
    
3. Write `tests/test_benchmark.py` using `locust` or `asyncio` test scripts to simulate 50 concurrent WebSocket streaming clients.
    
      
    
4. Measure total round-trip server latencies and assert diagnostic metric criteria against test datasets.
    
      
    

### Phase Deliverables & Success Metrics

- Live, operational backend running on Hugging Face Spaces (`wss://...`).
    
      
    
- Benchmark report confirming total server-side latency $< 85\text{ ms}$ under load.
    
      
    
- Evaluation metrics satisfying target validation benchmarks ($\text{Macro F1} \ge 0.924, R^2 \ge 0.931$).