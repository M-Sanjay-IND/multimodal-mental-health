import math
import torch
import torch.nn as nn


class MultiHeadCrossAttention(nn.Module):
    """
    Bi-directional Multi-Head Cross-Attention (MHCA) module.
    Calculates scaled dot-product cross-attention between Query stream Q (e.g., Tabular)
    and Key-Value streams K, V (e.g., Visual or Acoustic).
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.w_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """
        query: [B, embed_dim] or [B, 1, embed_dim]
        key: [B, embed_dim] or [B, 1, embed_dim]
        value: [B, embed_dim] or [B, 1, embed_dim]
        Returns:
            attn_output: [B, embed_dim]
            attn_weights: [B, num_heads, 1, 1]
        """
        B = query.size(0)

        if query.dim() == 2:
            query = query.unsqueeze(1)  # [B, 1, embed_dim]
        if key.dim() == 2:
            key = key.unsqueeze(1)  # [B, 1, embed_dim]
        if value.dim() == 2:
            value = value.unsqueeze(1)  # [B, 1, embed_dim]

        # Project and reshape into heads: [B, num_heads, 1, head_dim]
        q = self.w_q(query).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(key).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention: [B, num_heads, 1, 1]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights_drop = self.dropout(attn_weights)

        # Output context: [B, num_heads, 1, head_dim]
        context = torch.matmul(attn_weights_drop, v)

        # Reshape context back: [B, 1, embed_dim] -> [B, embed_dim]
        context = context.transpose(1, 2).contiguous().view(B, -1, self.embed_dim)
        attn_output = self.out_proj(context).squeeze(1)

        return attn_output, attn_weights
