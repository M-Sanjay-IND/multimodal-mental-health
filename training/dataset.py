import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from schemas.payload import TABULAR_FEATURE_NAMES


class MultimodalParquetDataset(Dataset):
    """
    PyTorch Dataset for loading multimodal parquet data (Visual, Acoustic, Tabular)
    and multi-task targets (Classification class + Continuous Regression scores).
    """

    def __init__(self, parquet_path: str):
        self.df = pd.read_parquet(parquet_path)

        # Extract features
        self.v_embs = np.array(self.df["visual_vector"].tolist(), dtype=np.float32)  # [N, 128]
        self.a_embs = np.array(self.df["acoustic_vector"].tolist(), dtype=np.float32)  # [N, 256]
        self.t_embs = self.df[TABULAR_FEATURE_NAMES].values.astype(np.float32)  # [N, 18]

        # Extract targets
        self.target_class = self.df["target_class"].values.astype(np.int64)  # [N]
        self.reg_targets = self.df[["Depression_Score", "Anxiety_Score", "Stress_Score"]].values.astype(
            np.float32
        )  # [N, 3]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        return {
            "visual": torch.from_numpy(self.v_embs[idx]),
            "acoustic": torch.from_numpy(self.a_embs[idx]),
            "tabular": torch.from_numpy(self.t_embs[idx]),
            "target_class": torch.tensor(self.target_class[idx], dtype=torch.long),
            "target_reg": torch.from_numpy(self.reg_targets[idx]),
        }


def create_dataloaders(
    train_path: str = "data/train.parquet",
    val_path: str = "data/val.parquet",
    batch_size: int = 32,
    num_workers: int = 0,
):
    train_ds = MultimodalParquetDataset(train_path)
    val_ds = MultimodalParquetDataset(val_path)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
