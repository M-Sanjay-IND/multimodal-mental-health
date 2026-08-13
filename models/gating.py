import torch
import torch.nn as nn


class AdaptiveModalityDropout(nn.Module):
    """
    Randomly zeros out individual input modality streams during training
    with probability p_m in [0.05, 0.20] (default 0.15) to simulate missing or corrupted client data streams.
    """

    def __init__(self, p_m: float = 0.15):
        super().__init__()
        self.p_m = p_m

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            # Explicit missing mask: [B, 1] (0.0 for missing stream, 1.0 for present)
            if mask.dim() == 1:
                mask = mask.unsqueeze(1)
            return x * mask

        if self.training and self.p_m > 0.0:
            # Random Bernoulli drop per sample in batch
            B = x.size(0)
            bernoulli = torch.bernoulli(torch.full((B, 1), 1.0 - self.p_m, device=x.device))
            return x * bernoulli / (1.0 - self.p_m)

        return x


class MARGBlock(nn.Module):
    """
    Modality-Aware Residual Gating (MARG) Unit.
    Fuses unimodal embedding E_m with cross-attended output MHCA(m, .) via dynamic sigmoid gating:
        g_m = sigmoid( W_g [E_m || MHCA(m, .)] + b_g )
        E_tilde_m = LayerNorm( E_m + g_m * MHCA(m, .) )
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.gate_dense = nn.Linear(embed_dim * 2, embed_dim)
        self.gate_sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, e_m: torch.Tensor, mhca_out: torch.Tensor) -> torch.Tensor:
        # e_m: [B, 256], mhca_out: [B, 256]
        concat_feats = torch.cat([e_m, mhca_out], dim=-1)  # [B, 512]
        gate = self.gate_sigmoid(self.gate_dense(concat_feats))  # [B, 256]

        gated_mhca = gate * mhca_out
        e_tilde = self.norm(e_m + gated_mhca)
        return e_tilde
