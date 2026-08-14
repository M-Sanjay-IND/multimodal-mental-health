import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, r2_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.multi_task import MultiTaskModel
from training.dataset import create_dataloaders
from training.losses import MultiTaskLoss, GradNormLossBalancer


def train_model(
    train_path: str = "data/train.parquet",
    val_path: str = "data/val.parquet",
    artifact_dir: str = "artifacts",
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cpu",
):
    os.makedirs(artifact_dir, exist_ok=True)
    print(f"[INFO] Training MultiTaskModel on {device.upper()} for {epochs} epochs...", flush=True)

    train_loader, val_loader = create_dataloaders(train_path, val_path, batch_size=batch_size)

    model = MultiTaskModel().to(device)
    # Class weights to handle severe class imbalance (Severe_Stress is ~3.2% of dataset)
    alpha_cls = torch.tensor([0.6, 0.8, 1.0, 7.8], dtype=torch.float32).to(device)
    loss_fn = MultiTaskLoss(alpha_cls=alpha_cls)
    balancer = GradNormLossBalancer().to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(balancer.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(artifact_dir, "model_state.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_cls_losses, train_reg_losses, train_total_losses = [], [], []

        for batch in train_loader:
            v_in = batch["visual"].to(device)
            a_in = batch["acoustic"].to(device)
            t_in = batch["tabular"].to(device)
            target_cls = batch["target_class"].to(device)
            target_reg = batch["target_reg"].to(device)

            optimizer.zero_grad()
            logits, reg_scores, _ = model(v_in, a_in, t_in)

            loss_cls, loss_reg = loss_fn(logits, reg_scores, target_cls, target_reg)
            total_loss = balancer.get_weighted_loss(loss_cls, loss_reg)

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_cls_losses.append(loss_cls.item())
            train_reg_losses.append(loss_reg.item())
            train_total_losses.append(total_loss.item())

        scheduler.step()

        # Validation loop
        model.eval()
        val_cls_losses, val_reg_losses = [], []
        all_pred_cls, all_true_cls = [], []
        all_pred_reg, all_true_reg = [], []

        with torch.no_grad():
            for batch in val_loader:
                v_in = batch["visual"].to(device)
                a_in = batch["acoustic"].to(device)
                t_in = batch["tabular"].to(device)
                target_cls = batch["target_class"].to(device)
                target_reg = batch["target_reg"].to(device)

                logits, reg_scores, _ = model(v_in, a_in, t_in)
                loss_cls, loss_reg = loss_fn(logits, reg_scores, target_cls, target_reg)

                val_cls_losses.append(loss_cls.item())
                val_reg_losses.append(loss_reg.item())

                pred_cls = torch.argmax(logits, dim=-1).cpu().numpy()
                all_pred_cls.extend(pred_cls)
                all_true_cls.extend(target_cls.cpu().numpy())

                all_pred_reg.append(reg_scores.cpu().numpy())
                all_true_reg.append(target_reg.cpu().numpy())

        val_cls_mean = np.mean(val_cls_losses)
        val_reg_mean = np.mean(val_reg_losses)
        val_total = val_cls_mean + val_reg_mean

        acc = accuracy_score(all_true_cls, all_pred_cls)
        macro_f1 = f1_score(all_true_cls, all_pred_cls, average="macro")

        all_pred_reg_arr = np.concatenate(all_pred_reg, axis=0)
        all_true_reg_arr = np.concatenate(all_true_reg, axis=0)
        mae_dep = mean_absolute_error(all_true_reg_arr[:, 0], all_pred_reg_arr[:, 0])
        mae_anx = mean_absolute_error(all_true_reg_arr[:, 1], all_pred_reg_arr[:, 1])
        mae_str = mean_absolute_error(all_true_reg_arr[:, 2], all_pred_reg_arr[:, 2])
        r2_overall = r2_score(all_true_reg_arr, all_pred_reg_arr)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] - "
            f"Train Loss: {np.mean(train_total_losses):.4f} | "
            f"Val Loss: {val_total:.4f} | "
            f"Acc: {acc*100:.1f}% | "
            f"Macro F1: {macro_f1:.3f} | "
            f"MAEs (Dep/Anx/Str): [{mae_dep:.2f}, {mae_anx:.2f}, {mae_str:.2f}] | "
            f"R2: {r2_overall:.3f}",
            flush=True,
        )

        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_total,
                    "accuracy": acc,
                    "macro_f1": macro_f1,
                    "r2": r2_overall,
                },
                best_checkpoint_path,
            )

    print(f"[SUCCESS] MultiTaskModel training finished. Saved best checkpoint to {best_checkpoint_path}")
    return model


if __name__ == "__main__":
    train_model()
