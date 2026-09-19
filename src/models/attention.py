import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    Многоголовое внимание с поддержкой:
      - packed batching (seq_ids),
      - GQA (n_kv_heads <= n_heads),
      - KV-кэш для инференса (past_kv).

    forward возвращает (out, new_kv), где new_kv = (k, v) — обновлённый кэш.
    Если past_kv=None — это первый вызов, кэш создаётся с нуля.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        if n_kv_heads is None:
            n_kv_heads = n_heads

        assert n_heads % n_kv_heads == 0, \
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.W_o = nn.Linear(n_heads * self.d_head, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_kv_heads, T, d_head) → (B, n_heads, T, d_head)."""
        if self.n_rep == 1:
            return x
        return x.repeat_interleave(self.n_rep, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        seq_ids: torch.Tensor,
        past_kv=None,
    ):
        """
        x: (B, T_new, d_model) — новые токены (при инференсе T_new=1).
        seq_ids: (B, T_new) — их seq_ids.
        past_kv: (past_k, past_v) или None.
            past_k: (B, n_kv_heads, T_past, d_head)
            past_v: (B, n_kv_heads, T_past, d_head)

        Возвращает (out, new_kv):
            out: (B, T_new, d_model)
            new_kv: (k_all, v_all), где k_all: (B, n_kv_heads, T_past+T_new, d_head)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # 1. Проекции (только для новых токенов!)
        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k_new = self.W_k(x).view(batch_size, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)
        v_new = self.W_v(x).view(batch_size, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)

        # 2. Приклеиваем новые K/V к кэшу
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k_new], dim=2)   # (B, n_kv_heads, T_past+T_new, d_head)
            v = torch.cat([past_v, v_new], dim=2)
        else:
            k = k_new
            v = v_new

        new_kv = (k, v)

        # 3. Расширяем K/V до n_heads (GQA)
        k_rep = self._repeat_kv(k)   # (B, n_heads, T_total, d_head)
        v_rep = self._repeat_kv(v)

        # 4. Оценки внимания
        # q: (B, n_heads, T_new, d_head), k_rep: (B, n_heads, T_total, d_head)
        scores = torch.matmul(q, k_rep.transpose(-2, -1)) / math.sqrt(self.d_head)
        # scores: (B, n_heads, T_new, T_total)

        # 5. Маска
        T_new = seq_len
        T_total = k.size(2)
        T_past = T_total - T_new

        # Позиции новых токенов в полной последовательности
        i_idx = T_past + torch.arange(T_new, device=device)   # (T_new,)
        j_idx = torch.arange(T_total, device=device)          # (T_total,)

        # Causal: j <= i
        causal = (j_idx.unsqueeze(0) <= i_idx.unsqueeze(1))    # (T_new, T_total)

        # Маска по seq_ids: только для режима обучения (packed batching)
        # При инференсе seq_ids = все единицы → маска по seq_ids не нужна
        if past_kv is None:
            # Packed batching: работаем как раньше
            s_i = seq_ids.unsqueeze(1).unsqueeze(-1)   # (B, 1, T_new, 1)
            s_j = seq_ids.unsqueeze(1).unsqueeze(-2)   # (B, 1, 1, T_new)
            same_object = (s_i == s_j)                 # (B, 1, T_new, T_new)
            not_pad = (s_i != 0)
            mask = same_object & causal.unsqueeze(0) & not_pad
        else:
            # Инференс: считаем, что все seq_ids одинаковые
            mask = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, self.n_heads, T_new, T_total)

        scores = scores.masked_fill(~mask, float("-inf"))

        # 6. Softmax
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        # 7. Взвешенная сумма
        out = torch.matmul(attn, v_rep)   # (B, n_heads, T_new, d_head)

        # 8. Склейка голов
        out = out.transpose(1, 2).contiguous().view(batch_size, T_new, self.d_model)

        # 9. Финальная проекция
        out = self.W_o(out)
        return out, new_kv
