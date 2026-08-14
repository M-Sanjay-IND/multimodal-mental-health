import os
import pytest
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
)

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
    macro_p = precision_score(all_true_cls_arr, all_pred_cls_arr, average="macro", zero_division=0)
    weighted_p = precision_score(all_true_cls_arr, all_pred_cls_arr, average="weighted", zero_division=0)
    macro_r = recall_score(all_true_cls_arr, all_pred_cls_arr, average="macro", zero_division=0)
    weighted_r = recall_score(all_true_cls_arr, all_pred_cls_arr, average="weighted", zero_division=0)
    macro_f1 = f1_score(all_true_cls_arr, all_pred_cls_arr, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_true_cls_arr, all_pred_cls_arr, average="weighted", zero_division=0)
    auc = roc_auc_score(all_true_cls_arr, all_logits_arr, multi_class="ovr", average="macro")
    cm = confusion_matrix(all_true_cls_arr, all_pred_cls_arr)

    mae_dep = mean_absolute_error(all_true_reg_arr[:, 0], all_pred_reg_arr[:, 0])
    mae_anx = mean_absolute_error(all_true_reg_arr[:, 1], all_pred_reg_arr[:, 1])
    mae_str = mean_absolute_error(all_true_reg_arr[:, 2], all_pred_reg_arr[:, 2])
    r2 = r2_score(all_true_reg_arr, all_pred_reg_arr)

    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "weighted_precision": weighted_p,
        "macro_recall": macro_r,
        "weighted_recall": weighted_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "mae_dep": mae_dep,
        "mae_anx": mae_anx,
        "mae_str": mae_str,
        "r2_overall": r2,
    }


def test_classification_accuracy_target(benchmark_results):
    assert benchmark_results["accuracy"] >= 0.850, f"Accuracy {benchmark_results['accuracy']:.4f} < 0.850 target!"


def test_precision_recall_targets(benchmark_results):
    assert benchmark_results["macro_precision"] >= 0.650, f"Macro precision {benchmark_results['macro_precision']:.4f} < 0.650 target!"
    assert benchmark_results["weighted_precision"] >= 0.800, f"Weighted precision {benchmark_results['weighted_precision']:.4f} < 0.800 target!"
    assert benchmark_results["macro_recall"] >= 0.650, f"Macro recall {benchmark_results['macro_recall']:.4f} < 0.650 target!"
    assert benchmark_results["weighted_recall"] >= 0.850, f"Weighted recall {benchmark_results['weighted_recall']:.4f} < 0.850 target!"


def test_f1_score_targets(benchmark_results):
    assert benchmark_results["macro_f1"] >= 0.650, f"Macro F1 {benchmark_results['macro_f1']:.4f} < 0.650 target!"
    assert benchmark_results["weighted_f1"] >= 0.850, f"Weighted F1 {benchmark_results['weighted_f1']:.4f} < 0.850 target!"


def test_roc_auc_target(benchmark_results):
    assert benchmark_results["roc_auc"] >= 0.950, f"ROC-AUC {benchmark_results['roc_auc']:.4f} < 0.950 target!"


def test_confusion_matrix_shape(benchmark_results):
    cm = benchmark_results["confusion_matrix"]
    assert cm.shape == (4, 4), f"Confusion matrix shape {cm.shape} != (4, 4)"


def test_regression_mae_targets(benchmark_results):
    assert benchmark_results["mae_dep"] <= 10.0, f"Depression MAE {benchmark_results['mae_dep']:.2f} > 10.0 target!"
    assert benchmark_results["mae_anx"] <= 9.0, f"Anxiety MAE {benchmark_results['mae_anx']:.2f} > 9.0 target!"
    assert benchmark_results["mae_str"] <= 12.0, f"Stress MAE {benchmark_results['mae_str']:.2f} > 12.0 target!"


def test_r2_overall_target(benchmark_results):
    assert benchmark_results["r2_overall"] >= 0.150, f"R2 overall {benchmark_results['r2_overall']:.4f} < 0.150 target!"

