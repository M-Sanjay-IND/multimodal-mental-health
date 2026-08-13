import torch
import torch.nn as nn
from models.projections import ModalityProjections
from models.attention import MultiHeadCrossAttention
from models.gating import AdaptiveModalityDropout, MARGBlock


class DCMFNet(nn.Module):
    """
    Dynamic Cross-Modal Attention Transformer Network (DCMF-Net).
    Fuses heterogeneous Visual (128-dim), Acoustic (256-dim), and Tabular (18-dim) input streams
    into a unified 768-dimensional latent vector Z_fused.
    """

    def __init__(
        self,
        v_dim: int = 128,
        a_dim: int = 256,
        t_dim: int = 18,
        embed_dim: int = 256,
        num_heads: int = 4,
        p_m: float = 0.15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # 1. Subspace Projection Layers
        self.projections = ModalityProjections(
            v_dim=v_dim,
            a_dim=a_dim,
            t_dim=t_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        # 2. Adaptive Modality Dropout
        self.modality_dropout_v = AdaptiveModalityDropout(p_m=p_m)
        self.modality_dropout_a = AdaptiveModalityDropout(p_m=p_m)

        # 3. Bi-Directional Multi-Head Cross-Attention Units
        self.mhca_vt = MultiHeadCrossAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.mhca_at = MultiHeadCrossAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.mhca_ta = MultiHeadCrossAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        # 4. Modality-Aware Residual Gating (MARG) Units
        self.marg_v = MARGBlock(embed_dim=embed_dim)
        self.marg_a = MARGBlock(embed_dim=embed_dim)
        self.marg_t = MARGBlock(embed_dim=embed_dim)

    def forward(
        self,
        visual_x: torch.Tensor,
        acoustic_x: torch.Tensor,
        tabular_x: torch.Tensor,
        v_mask: torch.Tensor = None,
        a_mask: torch.Tensor = None,
    ):
        """
        Inputs:
            visual_x:   [B, 128]
            acoustic_x: [B, 256]
            tabular_x:  [B, 18]
            v_mask:     [B, 1] optional mask for visual stream
            a_mask:     [B, 1] optional mask for acoustic stream
        Returns:
            z_fused:    [B, 768] (Concat of gated e_v_tilde, e_a_tilde, e_t_tilde)
            attn_dict:  Dict of attention weights for XAI
        """
        # Apply modality dropout / masks
        v_in = self.modality_dropout_v(visual_x, mask=v_mask)
        a_in = self.modality_dropout_a(acoustic_x, mask=a_mask)
        t_in = tabular_x

        # Subspace Projections (all mapped to [B, 256])
        e_v, e_a, e_t = self.projections(v_in, a_in, t_in)

        # Bi-Directional Cross-Attentions
        # Tabular queries attend to Visual keys/values
        mhca_vt_out, weights_vt = self.mhca_vt(query=e_t, key=e_v, value=e_v)

        # Tabular queries attend to Acoustic keys/values
        mhca_at_out, weights_at = self.mhca_at(query=e_t, key=e_a, value=e_a)

        # Visual queries attend to Tabular keys/values
        mhca_ta_out, weights_ta = self.mhca_ta(query=e_v, key=e_t, value=e_t)

        # Modality-Aware Residual Gating
        e_v_tilde = self.marg_v(e_v, mhca_vt_out)
        e_a_tilde = self.marg_a(e_a, mhca_at_out)
        e_t_tilde = self.marg_t(e_t, mhca_ta_out)

        # Concatenate into 768-dim fused latent vector
        z_fused = torch.cat([e_v_tilde, e_a_tilde, e_t_tilde], dim=-1)  # [B, 768]

        attn_dict = {
            "weights_vt": weights_vt,
            "weights_at": weights_at,
            "weights_ta": weights_ta,
        }

        return z_fused, attn_dict
