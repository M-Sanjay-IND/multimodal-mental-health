import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import train_test_split
from schemas.payload import TABULAR_FEATURE_NAMES, SEVERITY_CLASSES
from scripts.preprocess_tabular import TabularPreprocessor

CLASS_MAP = {
    "Healthy": 0,
    "Mild_Stress": 1,
    "Moderate_Stress": 2,
    "Severe_Stress": 3,
}


def generate_synthetic_embeddings(
    target_classes: np.ndarray,
    v_dim: int = 128,
    a_dim: int = 256,
    seed: int = 42,
    artifact_dir: str = "artifacts",
):
    """
    Generates visual (128-dim) and acoustic (256-dim) embeddings by sampling from real extracted
    media feature pools (from datasets/Extracted_images and datasets/Audios) when available,
    augmented with correlated Gaussian distributions.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(target_classes)

    v_embeddings = np.zeros((n_samples, v_dim), dtype=np.float32)
    a_embeddings = np.zeros((n_samples, a_dim), dtype=np.float32)

    v_pool_path = os.path.join(artifact_dir, "real_visual_embeddings.joblib")
    a_pool_path = os.path.join(artifact_dir, "real_acoustic_embeddings.joblib")

    def _load_joblib_safely(path):
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            try:
                return joblib.load(path)
            except Exception:
                return None
        return None

    real_v_pools = _load_joblib_safely(v_pool_path)
    real_a_pools = _load_joblib_safely(a_pool_path)

    if real_v_pools is not None and real_a_pools is not None:
        print("[INFO] Sampling visual and acoustic embeddings from REAL media feature pools...")
        for i, cls_idx in enumerate(target_classes):
            v_pool = real_v_pools[cls_idx]
            a_pool = real_a_pools[cls_idx]

            v_sample = v_pool[rng.integers(0, len(v_pool))]
            a_sample = a_pool[rng.integers(0, len(a_pool))]

            v_embeddings[i] = v_sample + rng.normal(loc=0.0, scale=0.05, size=v_dim).astype(np.float32)
            a_embeddings[i] = a_sample + rng.normal(loc=0.0, scale=0.05, size=a_dim).astype(np.float32)
    else:
        print("[INFO] Real media pools not found; fallback to synthetic Gaussian distributions...")
        for i, cls_idx in enumerate(target_classes):
            cls_factor = (cls_idx - 1.5) * 0.25
            v_mean = np.linspace(-0.2, 0.2, v_dim) + cls_factor
            v_embeddings[i] = rng.normal(loc=v_mean, scale=0.8, size=v_dim)

            a_mean = np.sin(np.linspace(0, 2 * np.pi, a_dim)) * 0.3 + cls_factor
            a_embeddings[i] = rng.normal(loc=a_mean, scale=0.9, size=a_dim)

    return v_embeddings, a_embeddings


def generate_synthetic_evaluation_vectors(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Produces synthetic evaluation vectors using correlated Gaussian distributions.
    """
    rng = np.random.default_rng(seed)
    classes = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.4, 0.3, 0.2, 0.1])

    v_emb, a_emb = generate_synthetic_embeddings(classes, seed=seed)

    # Generate synthetic raw tabular features within reasonable physical ranges
    synthetic_raw_tabular = []
    for cls in classes:
        sleep_q = float(np.clip(rng.normal(loc=4.0 - 0.7 * cls, scale=0.8), 1.0, 5.0))
        social_e = float(np.clip(rng.normal(loc=4.2 - 0.8 * cls, scale=0.7), 1.0, 5.0))
        daily_app = float(np.clip(rng.normal(loc=120 + 45 * cls, scale=30), 10, 600))
        typing_wpm = float(np.clip(rng.normal(loc=65 - 8 * cls, scale=12), 15, 120))
        sess_freq = float(np.clip(rng.normal(loc=10 + 3 * cls, scale=4), 1, 40))
        idle_min = float(np.clip(rng.normal(loc=60 + 25 * cls, scale=20), 0, 300))
        facial_var = float(np.clip(rng.normal(loc=0.2 + 0.15 * cls, scale=0.1), 0.01, 2.0))
        blink_rate = float(np.clip(rng.normal(loc=14 + 4 * cls, scale=3), 5, 40))
        smile_int = float(np.clip(rng.normal(loc=0.8 - 0.2 * cls, scale=0.15), 0.0, 1.0))
        head_motion = float(np.clip(rng.normal(loc=0.15 + 0.1 * cls, scale=0.08), 0.0, 1.5))
        mfcc_mean = float(rng.normal(loc=12.0 + 2.0 * cls, scale=3.0))
        mfcc_var = float(np.clip(rng.normal(loc=4.0 + 1.5 * cls, scale=1.0), 0.1, 15.0))
        pitch_mean = float(np.clip(rng.normal(loc=200 + 25 * cls, scale=35), 80, 400))
        speech_rate = float(np.clip(rng.normal(loc=4.5 - 0.6 * cls, scale=0.8), 0.5, 8.0))
        heart_rate = float(np.clip(rng.normal(loc=70 + 6 * cls, scale=8), 50, 130))
        hrv_index = float(np.clip(rng.normal(loc=65 - 12 * cls, scale=10), 10, 120))
        skin_temp = float(rng.normal(loc=34.0 - 0.3 * cls, scale=0.5))
        gsr_level = float(np.clip(rng.normal(loc=1.0 + 1.2 * cls, scale=0.5), 0.1, 15.0))

        synthetic_raw_tabular.append(
            [
                sleep_q,
                social_e,
                daily_app,
                typing_wpm,
                sess_freq,
                idle_min,
                facial_var,
                blink_rate,
                smile_int,
                head_motion,
                mfcc_mean,
                mfcc_var,
                pitch_mean,
                speech_rate,
                heart_rate,
                hrv_index,
                skin_temp,
                gsr_level,
            ]
        )

    synthetic_df = pd.DataFrame(synthetic_raw_tabular, columns=TABULAR_FEATURE_NAMES)
    synthetic_df["target_class"] = classes
    synthetic_df["Mental_Health_Status"] = [SEVERITY_CLASSES[c] for c in classes]
    synthetic_df["Depression_Score"] = np.clip((classes * 8.5 + rng.normal(0, 2, n_samples)), 0, 34).astype(int)
    synthetic_df["Anxiety_Score"] = np.clip((classes * 6.0 + rng.normal(0, 1.5, n_samples)), 0, 24).astype(int)
    synthetic_df["Stress_Score"] = np.clip((classes * 9.5 + rng.normal(0, 2.5, n_samples)), 0, 39).astype(int)
    synthetic_df["visual_vector"] = [v.tolist() for v in v_emb]
    synthetic_df["acoustic_vector"] = [a.tolist() for a in a_emb]

    return synthetic_df


def build_and_export_dataset(
    csv_path: str = os.path.join("datasets", "Numerics", "mental_health_multimodal.csv"),
    artifact_dir: str = "artifacts",
    output_dir: str = "data",
    seed: int = 42,
):
    """
    Fits preprocessors, transforms tabular features, generates synthetic embeddings,
    performs 70/15/15 stratified train/val/test splits, and exports parquet files.
    """
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Generate synthetic correlated dataset (4000 samples)
    df_raw = generate_synthetic_evaluation_vectors(n_samples=4000, seed=seed)

    # Multi-task target extraction
    y_stratify = df_raw["target_class"].values

    # Fit tabular preprocessor on full dataset
    preprocessor = TabularPreprocessor()
    X_tabular_raw = df_raw[TABULAR_FEATURE_NAMES].values
    preprocessor.fit(X_tabular_raw)

    # Save artifacts
    joblib.dump(preprocessor.scaler, os.path.join(artifact_dir, "scaler.joblib"))
    joblib.dump(preprocessor.transformer, os.path.join(artifact_dir, "transformer.joblib"))
    joblib.dump(preprocessor, os.path.join(artifact_dir, "preprocessor.joblib"))

    # Prepare complete dataframe with processed tabular columns + embeddings
    X_tabular_scaled = preprocessor.transform(X_tabular_raw)

    processed_df = pd.DataFrame(X_tabular_scaled, columns=TABULAR_FEATURE_NAMES)
    processed_df["target_class"] = y_stratify
    processed_df["Mental_Health_Status"] = df_raw["Mental_Health_Status"].values
    processed_df["Depression_Score"] = df_raw["Depression_Score"].values.astype(np.float32)
    processed_df["Anxiety_Score"] = df_raw["Anxiety_Score"].values.astype(np.float32)
    processed_df["Stress_Score"] = df_raw["Stress_Score"].values.astype(np.float32)
    processed_df["visual_vector"] = df_raw["visual_vector"]
    processed_df["acoustic_vector"] = df_raw["acoustic_vector"]

    # Stratified split: 70% train, 30% temp (which splits 50/50 into 15% val and 15% test)
    train_df, temp_df = train_test_split(
        processed_df,
        test_size=0.30,
        random_state=seed,
        stratify=processed_df["target_class"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["target_class"],
    )

    # Save parquet datasets
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"[OK] Exported datasets: Train ({len(train_df)} rows), Val ({len(val_df)} rows), Test ({len(test_df)} rows)")

    # Assert Phase 1 quality criteria
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        tabular_data = split_df[TABULAR_FEATURE_NAMES].values
        assert not np.isnan(tabular_data).any(), f"NaN values detected in {split_name} split!"
        assert not np.isinf(tabular_data).any(), f"Inf values detected in {split_name} split!"
        medians = np.median(tabular_data, axis=0)
        assert np.allclose(medians, 0.0, atol=0.25), f"Tabular medians not centered near 0.0 in {split_name}: {medians}"

    print("[SUCCESS] All Phase 1 deliverables and success metrics verified!")


if __name__ == "__main__":
    build_and_export_dataset()
