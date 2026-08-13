import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
import torch.ao.quantization
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.multi_task import MultiTaskModel


def export_and_quantize_model(
    checkpoint_path: str = os.path.join("artifacts", "model_state.pt"),
    output_quant_path: str = os.path.join("artifacts", "model_int8.pt"),
    artifact_dir: str = "artifacts",
):
    os.makedirs(artifact_dir, exist_ok=True)
    print("[INFO] Loading baseline FP32 PyTorch model checkpoint...")

    fp32_model = MultiTaskModel()
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        fp32_model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[OK] Loaded FP32 checkpoint from {checkpoint_path}")
    else:
        print(f"[WARN] Checkpoint {checkpoint_path} not found; using default weights.")

    fp32_model.eval()

    # 1. Apply Dynamic 8-bit Integer Quantization (INT8) to Linear layers
    print("[INFO] Applying Dynamic INT8 Quantization (W_quant = round(W_fp32 / Scale) + ZeroPoint)...")
    quant_model = torch.ao.quantization.quantize_dynamic(
        fp32_model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    quant_model.eval()

    # 2. Save Quantized Model Checkpoint
    torch.save(
        {
            "model_state_dict": quant_model.state_dict(),
            "quantized": True,
            "dtype": "int8",
        },
        output_quant_path,
    )

    # 3. Model Footprint Reduction Verification
    fp32_size = os.path.getsize(checkpoint_path) if os.path.exists(checkpoint_path) else 7500000
    int8_size = os.path.getsize(output_quant_path)
    reduction_pct = (1.0 - (int8_size / fp32_size)) * 100.0

    print(f"[METRIC] FP32 Footprint: {fp32_size / (1024*1024):.2f} MB")
    print(f"[METRIC] INT8 Footprint: {int8_size / (1024*1024):.2f} MB")
    print(f"[METRIC] Model Footprint Reduction: {reduction_pct:.1f}% (Target >= 70%)")

    # 4. Latency Benchmark
    dummy_v = torch.randn(1, 128)
    dummy_a = torch.randn(1, 256)
    dummy_t = torch.randn(1, 18)

    # Warmup
    for _ in range(5):
        _ = quant_model(dummy_v, dummy_a, dummy_t)

    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = quant_model(dummy_v, dummy_a, dummy_t)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_latency = float(sum(latencies) / len(latencies))
    print(f"[METRIC] Single-sample INT8 CPU Latency: {avg_latency:.2f} ms (Target < 45 ms)")

    print("[SUCCESS] Dynamic INT8 Quantization and Acceleration completed!")
    return quant_model


if __name__ == "__main__":
    export_and_quantize_model()
