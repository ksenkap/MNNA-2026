import torch
import torch.nn as nn

from src.models.attention import MultiHeadAttention
from src.models.ffn import FFN


class TransformerLayer(nn.Module):
    """
    Один слой трансформера (post-norm), поддерживает KV-кэш.
    Возвращает (out, new_kv).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_kv_heads: int = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
        )
        self.ffn = FFN(d_model=d_model, d_ff=d_ff, dropout=dropout, activation=activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, seq_ids, past_kv=None):
        attn_out, new_kv = self.attn(x, seq_ids, past_kv=past_kv)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x, new_kv
