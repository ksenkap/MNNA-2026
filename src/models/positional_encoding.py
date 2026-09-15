import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """
    Синусоидальное позиционное кодирование с поддержкой packed batching.

    seq_ids: (batch, seq_len) — номер объекта для каждого токена:
        0 — PAD
        1 — первый объект
        2 — второй объект
        ...
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, seq_ids: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        seq_ids: (batch, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        positions = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

        for b in range(batch_size):
            pos = 0
            for i in range(seq_len):
                sid = seq_ids[b, i].item()
                if sid == 0:
                    positions[b, i] = 0
                    pos = 0
                else:
                    if i > 0 and seq_ids[b, i - 1].item() == sid:
                        pos += 1
                    else:
                        pos = 0
                    positions[b, i] = pos

        pe = self.pe[0, positions]  # (batch, seq_len, d_model)
        return self.dropout(x + pe)
