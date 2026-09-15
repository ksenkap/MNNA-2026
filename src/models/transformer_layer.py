import torch
import torch.nn as nn

from src.models.attention import MultiHeadAttention
from src.models.ffn import FFN


class TransformerLayer(nn.Module):
    """
    Один слой трансформера в post-norm варианте:

        z1 = LayerNorm(x + Attention(x))
        z2 = LayerNorm(z1 + FFN(z1))

    post-norm — нормализация применяется ПОСЛЕ residual connection.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ffn = FFN(d_model=d_model, d_ff=d_ff, dropout=dropout, activation=activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, seq_ids: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        seq_ids: (batch, seq_len)
        """
        # Подслой внимания с residual + post-norm
        attn_out = self.attn(x, seq_ids)
        x = self.norm1(x + self.dropout(attn_out))

        # Подслой FFN с residual + post-norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x
