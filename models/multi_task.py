import torch
import torch.nn as nn
from models.dcmf_net import DCMFNet


class SharedNeuralTrunk(nn.Module):
    """
    Shared deep neural trunk (768 -> 512 -> 256) with Layer Normalization,
    GELU activations, residual connections, and Dropout (p = 0.30).
    """

    def __init__(self, in_dim: int = 768, hidden_dim: int = 512, out_dim: int = 256, dropout: float = 0.30):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, z_fused: torch.Tensor) -> torch.Tensor:
        h1 = self.block1(z_fused)
        h2 = self.block2(h1)
        return h2


class ClassificationHead(nn.Module):
    """
    Categorical Classification Head (256 -> 128 -> 4 logits).
    Target classes: Healthy (0), Mild_Stress (1), Moderate_Stress (2), Severe_Stress (3).
    """

    def __init__(self, in_dim: int = 256, num_classes: int = 4, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, z_shared: torch.Tensor) -> torch.Tensor:
        return self.net(z_shared)


class RegressionHead(nn.Module):
    """
    Multi-Output Continuous Regression Head (256 -> 128 -> 3 bounded outputs).
    Outputs bounded symptom scores via scaled Sigmoids:
        Depression (0-34), Anxiety (0-24), Stress (0-39).
    """

    def __init__(self, in_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )
        # Clinical upper bounds
        self.register_buffer("bounds", torch.tensor([34.0, 24.0, 39.0], dtype=torch.float32))

    def forward(self, z_shared: torch.Tensor) -> torch.Tensor:
        raw_logits = self.net(z_shared)  # [B, 3]
        sigmoids = torch.sigmoid(raw_logits)
        scaled_scores = sigmoids * self.bounds  # [B, 3] bounded
        return scaled_scores


class MultiTaskModel(nn.Module):
    """
    Unified Multi-Task Model integrating:
    1. DCMFNet (Central Multimodal Fusion Transformer)
    2. Shared Neural Trunk
    3. Categorical Classification Head (4-class)
    4. Continuous Regression Head (3-variable bounded)
    """

    def __init__(
        self,
        v_dim: int = 128,
        a_dim: int = 256,
        t_dim: int = 18,
        embed_dim: int = 256,
        num_heads: int = 4,
        p_m: float = 0.15,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.dcmf_net = DCMFNet(
            v_dim=v_dim,
            a_dim=a_dim,
            t_dim=t_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            p_m=p_m,
            dropout=dropout,
        )
        self.shared_trunk = SharedNeuralTrunk(in_dim=embed_dim * 3, hidden_dim=512, out_dim=256, dropout=dropout)
        self.classifier = ClassificationHead(in_dim=256, num_classes=4, dropout=dropout)
        self.regressor = RegressionHead(in_dim=256, dropout=dropout)

    def forward(
        self,
        visual_x: torch.Tensor,
        acoustic_x: torch.Tensor,
        tabular_x: torch.Tensor,
        v_mask: torch.Tensor = None,
        a_mask: torch.Tensor = None,
    ):
        """
        Returns:
            logits:     [B, 4] categorical classification logits
            reg_scores: [B, 3] continuous regression scores (Depression, Anxiety, Stress)
            attn_dict:  Dict of cross-attention weights
        """
        z_fused, attn_dict = self.dcmf_net(visual_x, acoustic_x, tabular_x, v_mask=v_mask, a_mask=a_mask)
        z_shared = self.shared_trunk(z_fused)

        logits = self.classifier(z_shared)
        reg_scores = self.regressor(z_shared)

        return logits, reg_scores, attn_dict
