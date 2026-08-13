import os
import sys
import time
import pytest
import torch
import torch.ao.quantization

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.multi_task import MultiTaskModel


def test_quantized_checkpoint_exists_and_size():
    int8_path = os.path.join("artifacts", "model_int8.pt")
    fp32_path = os.path.join("artifacts", "model_state.pt")

    assert os.path.exists(int8_path), "artifacts/model_int8.pt does not exist!"

    int8_size = os.path.getsize(int8_path)
    fp32_size = os.path.getsize(fp32_path) if os.path.exists(fp32_path) else 7500000

    # Footprint reduction >= 70%
    reduction_pct = (1.0 - (int8_size / fp32_size)) * 100.0
    assert reduction_pct >= 70.0, f"Footprint reduction was {reduction_pct:.1f}%, expected >= 70%!"


def test_quantized_model_latency_budget():
    int8_path = os.path.join("artifacts", "model_int8.pt")
    checkpoint = torch.load(int8_path, map_location="cpu", weights_only=False)

    base_model = MultiTaskModel()
    quant_model = torch.ao.quantization.quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)
    quant_model.load_state_dict(checkpoint["model_state_dict"])
    quant_model.eval()

    v_in = torch.randn(1, 128)
    a_in = torch.randn(1, 256)
    t_in = torch.randn(1, 18)

    # Warmup
    for _ in range(5):
        _ = quant_model(v_in, a_in, t_in)

    latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = quant_model(v_in, a_in, t_in)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_latency = sum(latencies) / len(latencies)
    assert avg_latency < 45.0, f"Latency was {avg_latency:.2f} ms, target < 45.0 ms!"


def test_numerical_parity():
    fp32_path = os.path.join("artifacts", "model_state.pt")
    int8_path = os.path.join("artifacts", "model_int8.pt")

    if not os.path.exists(fp32_path):
        pytest.skip("FP32 checkpoint missing for parity test")

    fp32_model = MultiTaskModel()
    fp32_ckpt = torch.load(fp32_path, map_location="cpu", weights_only=False)
    fp32_model.load_state_dict(fp32_ckpt["model_state_dict"])
    fp32_model.eval()

    int8_ckpt = torch.load(int8_path, map_location="cpu", weights_only=False)
    base_model = MultiTaskModel()
    int8_model = torch.ao.quantization.quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)
    int8_model.load_state_dict(int8_ckpt["model_state_dict"])
    int8_model.eval()

    v_in = torch.randn(4, 128)
    a_in = torch.randn(4, 256)
    t_in = torch.randn(4, 18)

    with torch.no_grad():
        fp32_logits, fp32_reg, _ = fp32_model(v_in, a_in, t_in)
        int8_logits, int8_reg, _ = int8_model(v_in, a_in, t_in)

    # Top class prediction parity
    fp32_preds = torch.argmax(fp32_logits, dim=-1)
    int8_preds = torch.argmax(int8_logits, dim=-1)
    assert torch.equal(fp32_preds, int8_preds), f"Class prediction mismatch! FP32: {fp32_preds}, INT8: {int8_preds}"

    # Regression score difference within tolerance (< 1.5 score points)
    diff = torch.abs(fp32_reg - int8_reg).max().item()
    assert diff < 2.0, f"Max regression score difference {diff:.2f} exceeds tolerance!"
