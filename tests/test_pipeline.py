import os
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import PowerTransformer, RobustScaler
from schemas.payload import TABULAR_FEATURE_NAMES, SEVERITY_CLASSES
from scripts.preprocess_tabular import TabularPreprocessor
from scripts.generate_synthetic_data import generate_synthetic_evaluation_vectors


def test_preprocessor_artifacts_exist():
    assert os.path.exists(os.path.join("artifacts", "scaler.joblib"))
    assert os.path.exists(os.path.join("artifacts", "transformer.joblib"))
    assert os.path.exists(os.path.join("artifacts", "preprocessor.joblib"))


def test_preprocessor_artifacts_load_and_transform():
    scaler = joblib.load(os.path.join("artifacts", "scaler.joblib"))
    transformer = joblib.load(os.path.join("artifacts", "transformer.joblib"))
    preprocessor = joblib.load(os.path.join("artifacts", "preprocessor.joblib"))

    assert isinstance(scaler, RobustScaler)
    assert isinstance(transformer, PowerTransformer)
    assert isinstance(preprocessor, TabularPreprocessor)

    sample_input = np.ones((5, 18), dtype=np.float64)
    transformed = preprocessor.transform(sample_input)
    assert transformed.shape == (5, 18)
    assert not np.isnan(transformed).any()
    assert not np.isinf(transformed).any()


def test_parquet_splits_exist_and_valid():
    train_path = os.path.join("data", "train.parquet")
    val_path = os.path.join("data", "val.parquet")
    test_path = os.path.join("data", "test.parquet")

    assert os.path.exists(train_path)
    assert os.path.exists(val_path)
    assert os.path.exists(test_path)

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    total_rows = len(train_df) + len(val_df) + len(test_df)
    assert total_rows == 4000
    assert abs(len(train_df) / total_rows - 0.70) < 0.01
    assert abs(len(val_df) / total_rows - 0.15) < 0.01
    assert abs(len(test_df) / total_rows - 0.15) < 0.01

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        tabular_vals = df[TABULAR_FEATURE_NAMES].values
        assert not np.isnan(tabular_vals).any(), f"NaN found in {name}!"
        assert not np.isinf(tabular_vals).any(), f"Inf found in {name}!"

        # Check visual and acoustic embedding shapes
        assert len(df["visual_vector"].iloc[0]) == 128
        assert len(df["acoustic_vector"].iloc[0]) == 256

        # Check targets presence
        assert "target_class" in df.columns
        assert "Depression_Score" in df.columns
        assert "Anxiety_Score" in df.columns
        assert "Stress_Score" in df.columns

        # Check median centering near 0.0 across tabular columns
        medians = np.median(tabular_vals, axis=0)
        assert np.allclose(medians, 0.0, atol=0.25), f"Medians not centered near 0 in {name}: {medians}"


def test_synthetic_evaluation_vectors():
    syn_df = generate_synthetic_evaluation_vectors(n_samples=1000, seed=42)
    assert len(syn_df) == 1000
    assert len(syn_df["visual_vector"].iloc[0]) == 128
    assert len(syn_df["acoustic_vector"].iloc[0]) == 256
    for col in TABULAR_FEATURE_NAMES:
        assert col in syn_df.columns
        assert not syn_df[col].isna().any()
