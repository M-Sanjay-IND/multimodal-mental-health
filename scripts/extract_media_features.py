import os
import sys
import glob
import joblib
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import scipy.io.wavfile as wav
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas.payload import SEVERITY_CLASSES

# Emotion to Mental Health Status Severity Mapping
EMOTION_TO_SEVERITY_MAP = {
    "Neutral": "Healthy",
    "Happy": "Healthy",
    "Surprise": "Mild_Stress",
    "Sad": "Mild_Stress",
    "Fear": "Moderate_Stress",
    "Disgust": "Moderate_Stress",
    "Angry": "Severe_Stress",
}

SEVERITY_TO_INT = {
    "Healthy": 0,
    "Mild_Stress": 1,
    "Moderate_Stress": 2,
    "Severe_Stress": 3,
}

# RAVDESS Emotion Code (3rd part of filename) to Severity
RAVDESS_EMOTION_MAP = {
    "01": "Healthy",        # neutral
    "02": "Healthy",        # calm
    "03": "Healthy",        # happy
    "04": "Mild_Stress",     # sad
    "05": "Severe_Stress",   # angry
    "06": "Moderate_Stress", # fearful
    "07": "Moderate_Stress", # disgust
    "08": "Mild_Stress",     # surprised
}


class LightweightVisualEncoder(nn.Module):
    """
    CNN Feature Encoder projecting 48x48 facial images into 128-dim visual embedding vectors.
    """

    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 24x24
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 12x12
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # 4x4
        )
        self.fc = nn.Linear(128 * 4 * 4, out_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        out = self.fc(h)
        return out


class LightweightAcousticEncoder(nn.Module):
    """
    Neural Feature Encoder projecting 1D speech spectrograms/features into 256-dim acoustic vectors.
    """

    def __init__(self, in_features=64, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def extract_audio_spectral_features(wav_path: str, n_bins: int = 64) -> np.ndarray:
    """
    Extracts spectral log-energy parameters from WAV audio files using SciPy FFT.
    """
    try:
        sr, signal = wav.read(wav_path)
        if signal.ndim > 1:
            signal = signal.mean(axis=1)

        signal = signal.astype(np.float32)
        if np.max(np.abs(signal)) > 0:
            signal /= np.max(np.abs(signal))

        # Short-Time Fourier Transform (STFT) log-mel energy summary
        frame_len = int(sr * 0.025)
        hop_len = int(sr * 0.010)

        if len(signal) < frame_len:
            signal = np.pad(signal, (0, frame_len - len(signal)))

        # Take FFT over windowed frames
        n_frames = max(1, (len(signal) - frame_len) // hop_len)
        specs = []
        for i in range(min(n_frames, 100)):
            frame = signal[i * hop_len : i * hop_len + frame_len] * np.hanning(frame_len)
            fft_mag = np.abs(np.fft.rfft(frame, n=128))
            specs.append(fft_mag[:n_bins])

        if specs:
            feat = np.mean(specs, axis=0)
        else:
            feat = np.zeros(n_bins, dtype=np.float32)

        if len(feat) < n_bins:
            feat = np.pad(feat, (0, n_bins - len(feat)))

        return feat.astype(np.float32)
    except Exception as e:
        return np.zeros(n_bins, dtype=np.float32)


def process_and_cache_media_embeddings(
    images_dir: str = os.path.join("datasets", "Extracted_images"),
    audios_dir: str = os.path.join("datasets", "Audios"),
    artifact_dir: str = "artifacts",
    max_images_per_cls: int = 200,
    seed: int = 42,
):
    """
    Processes real image and audio datasets to produce real visual (128-dim) and acoustic (256-dim)
    embedding pools grouped by severity status class.
    """
    os.makedirs(artifact_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    v_encoder = LightweightVisualEncoder(out_dim=128)
    a_encoder = LightweightAcousticEncoder(in_features=64, out_dim=256)
    v_encoder.eval()
    a_encoder.eval()

    visual_pools = {0: [], 1: [], 2: [], 3: []}
    acoustic_pools = {0: [], 1: [], 2: [], 3: []}

    print("[INFO] Processing real facial images from datasets/Extracted_images...", flush=True)
    for emotion_dir in os.listdir(images_dir):
        full_dir = os.path.join(images_dir, emotion_dir)
        if not os.path.isdir(full_dir) or emotion_dir not in EMOTION_TO_SEVERITY_MAP:
            continue

        severity = EMOTION_TO_SEVERITY_MAP[emotion_dir]
        cls_id = SEVERITY_TO_INT[severity]

        img_files = glob.glob(os.path.join(full_dir, "*.png"))[:max_images_per_cls]
        batch_imgs = []
        for img_path in img_files:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                batch_imgs.append(img.astype(np.float32) / 255.0)

            if len(batch_imgs) == 64:
                imgs_arr = np.array(batch_imgs)[:, np.newaxis, :, :]
                batch_tensor = torch.from_numpy(imgs_arr)
                with torch.no_grad():
                    embs = v_encoder(batch_tensor).numpy()
                visual_pools[cls_id].extend(embs)
                batch_imgs = []

        if batch_imgs:
            imgs_arr = np.array(batch_imgs)[:, np.newaxis, :, :]
            batch_tensor = torch.from_numpy(imgs_arr)
            with torch.no_grad():
                embs = v_encoder(batch_tensor).numpy()
            visual_pools[cls_id].extend(embs)

    print("[INFO] Processing real speech audio recordings from datasets/Audios...", flush=True)
    wav_files = glob.glob(os.path.join(audios_dir, "**", "*.wav"), recursive=True)
    for wav_path in wav_files:
        filename = os.path.basename(wav_path)
        parts = filename.split("-")
        if len(parts) >= 3 and parts[2] in RAVDESS_EMOTION_MAP:
            severity = RAVDESS_EMOTION_MAP[parts[2]]
            cls_id = SEVERITY_TO_INT[severity]
        else:
            cls_id = 0

        spec_feat = extract_audio_spectral_features(wav_path, n_bins=64)
        spec_tensor = torch.from_numpy(spec_feat).float().unsqueeze(0)
        with torch.no_grad():
            emb = a_encoder(spec_tensor).squeeze(0).numpy()
        acoustic_pools[cls_id].append(emb)

    # Convert to numpy arrays
    for c in range(4):
        if len(visual_pools[c]) == 0:
            visual_pools[c] = [np.random.randn(128).astype(np.float32)]
        if len(acoustic_pools[c]) == 0:
            acoustic_pools[c] = [np.random.randn(256).astype(np.float32)]
        visual_pools[c] = np.array(visual_pools[c], dtype=np.float32)
        acoustic_pools[c] = np.array(acoustic_pools[c], dtype=np.float32)
        print(f"  Class {c} ({SEVERITY_CLASSES[c]}): {len(visual_pools[c])} visual vectors, {len(acoustic_pools[c])} acoustic vectors", flush=True)

    v_path = os.path.join(artifact_dir, "real_visual_embeddings.joblib")
    a_path = os.path.join(artifact_dir, "real_acoustic_embeddings.joblib")

    joblib.dump(visual_pools, v_path)
    joblib.dump(acoustic_pools, a_path)
    print(f"[OK] Saved real media embedding pools to {artifact_dir}/", flush=True)

    return visual_pools, acoustic_pools


if __name__ == "__main__":
    process_and_cache_media_embeddings()
