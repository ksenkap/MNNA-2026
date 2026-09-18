import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    Многоглавовое маскированное внимание с поддержкой packed batching.

    seq_ids: (batch, seq_len) — номер объекта для каждого токена:
        0 — PAD (не участвует во внимании)
        1, 2, 3, ... — номера объектов

    Внимание между позициями i и j разрешено, только если:
        - seq_ids[i] == seq_ids[j] (один объект)
        - j <= i (маскирование будущего)
        - seq_ids[i] != 0 (PAD не участвует)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, seq_ids: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        seq_ids: (batch, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # 1. Проекции + разбиение на головы
        # (batch, n_heads, seq_len, d_head)
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # 2. Оценки внимания (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # 3. Маска
        s_i = seq_ids.unsqueeze(1).unsqueeze(-1)  # (batch, 1, seq_len, 1)
        s_j = seq_ids.unsqueeze(1).unsqueeze(-2)  # (batch, 1, 1, seq_len)

        same_object = (s_i == s_j)

        i_idx = torch.arange(seq_len, device=device).view(1, 1, seq_len, 1)
        j_idx = torch.arange(seq_len, device=device).view(1, 1, 1, seq_len)
        causal = (j_idx <= i_idx)

        not_pad = (s_i != 0)

        mask = same_object & causal & not_pad  # (batch, 1, seq_len, seq_len)

        # 4. Применяем маску
        scores = scores.masked_fill(~mask, float("-inf"))

        # 5. Softmax
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        # 6. Взвешенная сумма
        out = torch.matmul(attn, V)

        # 7. Склейка голов
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 8. Финальная проекция
        out = self.W_o(out)
        return out
