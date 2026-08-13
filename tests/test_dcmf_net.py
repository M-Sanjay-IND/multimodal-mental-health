import pytest
import torch
from models.dcmf_net import DCMFNet
from models.projections import SubspaceProjection
from models.attention import MultiHeadCrossAttention
from models.gating import MARGBlock


def test_dcmf_net_forward_shape():
    model = DCMFNet(v_dim=128, a_dim=256, t_dim=18, embed_dim=256, num_heads=4)
    model.eval()

    for batch_size in [1, 4, 16, 32]:
        v_in = torch.randn(batch_size, 128)
        a_in = torch.randn(batch_size, 256)
        t_in = torch.randn(batch_size, 18)

        z_fused, attn_dict = model(v_in, a_in, t_in)

        assert z_fused.shape == (batch_size, 768)
        assert "weights_vt" in attn_dict
        assert "weights_at" in attn_dict
        assert attn_dict["weights_vt"].shape == (batch_size, 4, 1, 1)


def test_dcmf_net_missing_modality_robustness():
    model = DCMFNet(v_dim=128, a_dim=256, t_dim=18)
    model.eval()

    B = 4
    # Zeroed visual stream
    v_zero = torch.zeros(B, 128)
    a_in = torch.randn(B, 256)
    t_in = torch.randn(B, 18)

    z_fused_v_zero, _ = model(v_zero, a_in, t_in)
    assert z_fused_v_zero.shape == (B, 768)
    assert not torch.isnan(z_fused_v_zero).any()
    assert not torch.isinf(z_fused_v_zero).any()

    # Zeroed acoustic stream
    v_in = torch.randn(B, 128)
    a_zero = torch.zeros(B, 256)
    z_fused_a_zero, _ = model(v_in, a_zero, t_in)
    assert z_fused_a_zero.shape == (B, 768)

    # Explicit zero masks
    v_mask = torch.zeros(B, 1)
    a_mask = torch.ones(B, 1)
    z_masked, _ = model(v_in, a_in, t_in, v_mask=v_mask, a_mask=a_mask)
    assert z_masked.shape == (B, 768)


def test_dcmf_net_gradient_flow():
    model = DCMFNet(v_dim=128, a_dim=256, t_dim=18)
    model.train()

    v_in = torch.randn(8, 128, requires_grad=True)
    a_in = torch.randn(8, 256, requires_grad=True)
    t_in = torch.randn(8, 18, requires_grad=True)

    z_fused, _ = model(v_in, a_in, t_in)
    loss = z_fused.sum()
    loss.backward()

    # Verify gradients propagate to inputs
    assert v_in.grad is not None
    assert a_in.grad is not None
    assert t_in.grad is not None
    assert not torch.isnan(v_in.grad).any()

    # Verify gradients propagate to projection and attention parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for parameter {name}!"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in parameter {name}!"


def test_submodules_standalone():
    proj = SubspaceProjection(in_dim=18, embed_dim=256)
    x = torch.randn(4, 18)
    out_proj = proj(x)
    assert out_proj.shape == (4, 256)

    mhca = MultiHeadCrossAttention(embed_dim=256, num_heads=4)
    q = torch.randn(4, 256)
    k = torch.randn(4, 256)
    v = torch.randn(4, 256)
    out_mhca, weights = mhca(q, k, v)
    assert out_mhca.shape == (4, 256)
    assert weights.shape == (4, 4, 1, 1)

    marg = MARGBlock(embed_dim=256)
    out_marg = marg(q, out_mhca)
    assert out_marg.shape == (4, 256)
