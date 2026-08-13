import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, RobustScaler
from schemas.payload import TABULAR_FEATURE_NAMES


class IQRWinsorizer(BaseEstimator, TransformerMixin):
    """
    Clamps continuous feature values outside [Q1 - factor * IQR, Q3 + factor * IQR].
    """

    def __init__(self, factor: float = 2.5):
        self.factor = factor
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=np.float64)
        q1 = np.percentile(X_arr, 25, axis=0)
        q3 = np.percentile(X_arr, 75, axis=0)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - self.factor * iqr
        self.upper_bounds_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)


class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """
    Statistically conditioned tabular feature pipeline:
    1. IQR Winsorization
    2. Yeo-Johnson Power Transformation
    3. Robust Scaling
    """

    def __init__(self, iqr_factor: float = 2.5):
        self.iqr_factor = iqr_factor
        self.winsorizer = IQRWinsorizer(factor=iqr_factor)
        self.transformer = PowerTransformer(method="yeo-johnson", standardize=False)
        self.scaler = RobustScaler()

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=np.float64)
        X_win = self.winsorizer.fit_transform(X_arr)
        X_trans = self.transformer.fit_transform(X_win)
        self.scaler.fit(X_trans)
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        X_win = self.winsorizer.transform(X_arr)
        X_trans = self.transformer.transform(X_win)
        return self.scaler.transform(X_trans)

    def inverse_transform(self, X_scaled):
        X_trans = self.scaler.inverse_transform(X_scaled)
        X_win = self.transformer.inverse_transform(X_trans)
        return X_win


def fit_and_save_preprocessor(
    csv_path: str = os.path.join("datasets", "Numerics", "mental_health_multimodal.csv"),
    artifact_dir: str = "artifacts",
) -> TabularPreprocessor:
    os.makedirs(artifact_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    X = df[TABULAR_FEATURE_NAMES].values

    preprocessor = TabularPreprocessor()
    preprocessor.fit(X)

    # Save artifact files as specified
    joblib.dump(preprocessor.scaler, os.path.join(artifact_dir, "scaler.joblib"))
    joblib.dump(preprocessor.transformer, os.path.join(artifact_dir, "transformer.joblib"))
    joblib.dump(preprocessor, os.path.join(artifact_dir, "preprocessor.joblib"))

    print(f"[OK] Fit TabularPreprocessor on {len(df)} rows. Artifacts saved to {artifact_dir}/")
    return preprocessor


if __name__ == "__main__":
    fit_and_save_preprocessor()
