import torch
import torch.nn as nn


class SubspaceProjection(nn.Module):
    """
    Projects raw input embedding vectors into a unified latent subspace dimension d_m = 256.
    Includes Layer Normalization, GELU activations, and Dropout.
    """

    def __init__(self, in_dim: int, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, in_dim] -> [B, embed_dim]
        out = self.proj(x)
        out = self.norm(out)
        out = self.act(out)
        return self.dropout(out)


class ModalityProjections(nn.Module):
    """
    Container for Visual (128->256), Acoustic (256->256), and Tabular (18->256) projections.
    """

    def __init__(
        self,
        v_dim: int = 128,
        a_dim: int = 256,
        t_dim: int = 18,
        embed_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.visual_proj = SubspaceProjection(v_dim, embed_dim, dropout)
        self.acoustic_proj = SubspaceProjection(a_dim, embed_dim, dropout)
        self.tabular_proj = SubspaceProjection(t_dim, embed_dim, dropout)

    def forward(self, visual_x: torch.Tensor, acoustic_x: torch.Tensor, tabular_x: torch.Tensor):
        e_v = self.visual_proj(visual_x)
        e_a = self.acoustic_proj(acoustic_x)
        e_t = self.tabular_proj(tabular_x)
        return e_v, e_a, e_t
