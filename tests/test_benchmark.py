import os
import pytest
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, r2_score

from models.multi_task import MultiTaskModel
from training.dataset import MultimodalParquetDataset


@pytest.fixture(scope="module")
def benchmark_results():
    test_path = os.path.join("data", "test.parquet")
    checkpoint_path = os.path.join("artifacts", "model_state.pt")

    assert os.path.exists(test_path), "data/test.parquet dataset split missing!"
    assert os.path.exists(checkpoint_path), "artifacts/model_state.pt checkpoint missing!"

    model = MultiTaskModel()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_ds = MultimodalParquetDataset(test_path)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

    all_logits, all_pred_cls, all_true_cls = [], [], []
    all_pred_reg, all_true_reg = [], []

    with torch.no_grad():
        for batch in test_loader:
            logits, reg_scores, _ = model(batch["visual"], batch["acoustic"], batch["tabular"])
            all_logits.append(torch.softmax(logits, dim=-1).numpy())
            all_pred_cls.extend(torch.argmax(logits, dim=-1).numpy())
            all_true_cls.extend(batch["target_class"].numpy())
            all_pred_reg.append(reg_scores.numpy())
            all_true_reg.append(batch["target_reg"].numpy())

    all_logits_arr = np.concatenate(all_logits, axis=0)
    all_pred_reg_arr = np.concatenate(all_pred_reg, axis=0)
    all_true_reg_arr = np.concatenate(all_true_reg, axis=0)
    all_true_cls_arr = np.array(all_true_cls)
    all_pred_cls_arr = np.array(all_pred_cls)

    acc = accuracy_score(all_true_cls_arr, all_pred_cls_arr)
    f1 = f1_score(all_true_cls_arr, all_pred_cls_arr, average="macro")
    auc = roc_auc_score(all_true_cls_arr, all_logits_arr, multi_class="ovr", average="macro")
    mae_dep = mean_absolute_error(all_true_reg_arr[:, 0], all_pred_reg_arr[:, 0])
    mae_anx = mean_absolute_error(all_true_reg_arr[:, 1], all_pred_reg_arr[:, 1])
    mae_str = mean_absolute_error(all_true_reg_arr[:, 2], all_pred_reg_arr[:, 2])
    r2 = r2_score(all_true_reg_arr, all_pred_reg_arr)

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "roc_auc": auc,
        "mae_dep": mae_dep,
        "mae_anx": mae_anx,
        "mae_str": mae_str,
        "r2_overall": r2,
    }


def test_classification_accuracy_target(benchmark_results):
    assert benchmark_results["accuracy"] >= 0.936, f"Accuracy {benchmark_results['accuracy']:.4f} < 0.936 target!"


def test_macro_f1_target(benchmark_results):
    assert benchmark_results["macro_f1"] >= 0.924, f"Macro F1 {benchmark_results['macro_f1']:.4f} < 0.924 target!"


def test_roc_auc_target(benchmark_results):
    assert benchmark_results["roc_auc"] >= 0.978, f"ROC-AUC {benchmark_results['roc_auc']:.4f} < 0.978 target!"


def test_regression_mae_targets(benchmark_results):
    assert benchmark_results["mae_dep"] <= 1.50, f"Depression MAE {benchmark_results['mae_dep']:.2f} > 1.50 target!"
    assert benchmark_results["mae_anx"] <= 1.20, f"Anxiety MAE {benchmark_results['mae_anx']:.2f} > 1.20 target!"
    assert benchmark_results["mae_str"] <= 1.80, f"Stress MAE {benchmark_results['mae_str']:.2f} > 1.80 target!"


def test_r2_overall_target(benchmark_results):
    assert benchmark_results["r2_overall"] >= 0.931, f"R2 overall {benchmark_results['r2_overall']:.4f} < 0.931 target!"
