import os
import pytest
import torch
from models.multi_task import MultiTaskModel, SharedNeuralTrunk, ClassificationHead, RegressionHead
from training.losses import AsymmetricFocalLoss, MultiTaskLoss, GradNormLossBalancer
from training.dataset import MultimodalParquetDataset


def test_multi_task_model_forward():
    model = MultiTaskModel()
    model.eval()

    B = 8
    v_in = torch.randn(B, 128)
    a_in = torch.randn(B, 256)
    t_in = torch.randn(B, 18)

    logits, reg_scores, attn_dict = model(v_in, a_in, t_in)

    assert logits.shape == (B, 4)
    assert reg_scores.shape == (B, 3)
    assert "weights_vt" in attn_dict


def test_regression_head_bounds():
    reg_head = RegressionHead(in_dim=256)
    reg_head.eval()

    # Pass extreme inputs
    z_shared = torch.randn(10, 256) * 10.0
    scores = reg_head(z_shared)  # [10, 3]

    # Bounds: Depression <= 34, Anxiety <= 24, Stress <= 39
    assert (scores[:, 0] >= 0.0).all() and (scores[:, 0] <= 34.0).all()
    assert (scores[:, 1] >= 0.0).all() and (scores[:, 1] <= 24.0).all()
    assert (scores[:, 2] >= 0.0).all() and (scores[:, 2] <= 39.0).all()


def test_multi_task_losses():
    loss_fn = MultiTaskLoss()
    balancer = GradNormLossBalancer()

    logits = torch.randn(4, 4)
    reg_scores = torch.tensor([[10.0, 5.0, 15.0]] * 4)
    target_cls = torch.tensor([0, 1, 2, 3])
    target_reg = torch.tensor([[9.0, 6.0, 14.0]] * 4)

    loss_cls, loss_reg = loss_fn(logits, reg_scores, target_cls, target_reg)
    total_loss = balancer.get_weighted_loss(loss_cls, loss_reg)

    assert loss_cls.item() > 0.0
    assert loss_reg.item() > 0.0
    assert total_loss.item() > 0.0


def test_checkpoint_exists_and_loadable():
    checkpoint_path = os.path.join("artifacts", "model_state.pt")
    assert os.path.exists(checkpoint_path), "artifacts/model_state.pt checkpoint does not exist!"

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in checkpoint

    model = MultiTaskModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    v_in = torch.randn(2, 128)
    a_in = torch.randn(2, 256)
    t_in = torch.randn(2, 18)

    logits, reg_scores, _ = model(v_in, a_in, t_in)
    assert logits.shape == (2, 4)
    assert reg_scores.shape == (2, 3)
