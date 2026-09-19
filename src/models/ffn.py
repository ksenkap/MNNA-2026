import torch
import torch.nn as nn


class FFN(nn.Module):
    """
    Двухслойная полносвязная сеть с нелинейностью и dropout.

    FFN(x) = Dropout(act(x @ W1 + b1)) @ W2 + b2

    Внутренняя размерность d_ff обычно больше d_model (стандартно 4x),
    чтобы расширить пространство признаков.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
