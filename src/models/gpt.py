import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.positional_encoding import SinusoidalPositionalEncoding
from src.models.transformer_layer import TransformerLayer


class GPT(nn.Module):
    """
    GPT-like модель с GQA и KV-кэшем.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        n_kv_heads: int = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.n_layers = n_layers

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=d_model, max_seq_len=max_seq_len, dropout=dropout
        )

        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=self.n_kv_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        seq_ids: torch.Tensor,
        past_kvs=None,
        position_offset: int = 0,
        use_cache: bool = False,
    ):
        """
        input_ids: (B, T_new)
        seq_ids: (B, T_new)
        past_kvs: list[(k, v)] * n_layers, или None
        position_offset: int — сдвиг позиций (для инференса с KV-кэшем)
        use_cache: bool — если True, возвращает new_past_kvs (для инференса)

        Возвращает:
            logits: (B, T_new, vocab_size)
            new_past_kvs: list[(k, v)] * n_layers, или None
        """
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x, seq_ids, position_offset=position_offset)

        new_past_kvs = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, new_kv = layer(x, seq_ids, past_kv=past_kv)
            if use_cache:
                new_past_kvs.append(new_kv)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, new_past_kvs

    def compute_loss(self, logits, targets, seq_ids):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = targets[:, 1:].contiguous()

        s_i = seq_ids[:, :-1]
        s_next = seq_ids[:, 1:]
        loss_mask = (s_i == s_next) & (s_i != 0)

        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_targets.view(-1),
            reduction="none",
        )
        loss_per_token = loss_per_token.view(shift_targets.size(0), -1)

        loss_mask = loss_mask.float()
        loss = (loss_per_token * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
        return loss
