import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.positional_encoding import SinusoidalPositionalEncoding
from src.models.transformer_layer import TransformerLayer


class GPT(nn.Module):
    """
    GPT-like модель для обучения на packed batching.

    Параметры:
        vocab_size — размер словаря
        d_model    — размерность эмбеддинга
        n_heads    — число голов внимания
        n_layers   — число слоёв трансформера
        d_ff       — размерность FFN
        max_seq_len— максимальная длина последовательности
        dropout    — dropout
        activation — активация в FFN (gelu/relu)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # 1. Эмбеддинги токенов
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 2. Позиционное кодирование
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=d_model, max_seq_len=max_seq_len, dropout=dropout
        )

        # 3. Стек слоёв трансформера
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(n_layers)
        ])

        # 4. Финальный LayerNorm
        self.final_norm = nn.LayerNorm(d_model)

        # 5. LM-head: проекция в логиты по словарю
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, seq_ids: torch.Tensor):
        """
        input_ids: (batch, seq_len) — индексы токенов
        seq_ids:   (batch, seq_len) — номера объектов (0 для PAD)

        Возвращает:
            logits: (batch, seq_len, vocab_size)
        """
        # Эмбеддинги
        x = self.token_embedding(input_ids)  # (batch, seq_len, d_model)

        # Позиционное кодирование (внутри каждого объекта)
        x = self.pos_encoding(x, seq_ids)

        # Стек трансформеров
        for layer in self.layers:
            x = layer(x, seq_ids)

        # Финальная нормализация
        x = self.final_norm(x)

        # Логиты
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        seq_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Считает cross-entropy loss с маской для packed batching.

        Логиты на позиции i предсказывают токен на позиции i+1.
        Переход между разными объектами (s_i != s_{i+1}) не учитывается.
        PAD-токены (s_i == 0) тоже не учитываются.

        logits:  (batch, seq_len, vocab_size)
        targets: (batch, seq_len) — те же input_ids, но loss считается со сдвигом
        seq_ids: (batch, seq_len)
        """
        # Сдвигаем: логиты [:, :-1] предсказывают targets [:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()    # (batch, seq_len-1, vocab)
        shift_targets = targets[:, 1:].contiguous()      # (batch, seq_len-1)

        # Маска: учитываем только переходы внутри одного объекта
        # s_i — seq_ids на позиции i (текущий токен)
        # s_{i+1} — seq_ids на позиции i+1 (следующий токен)
        s_i = seq_ids[:, :-1]      # (batch, seq_len-1)
        s_next = seq_ids[:, 1:]    # (batch, seq_len-1)

        # M_i^loss = (s_i == s_{i+1}) & (s_i != 0)
        loss_mask = (s_i == s_next) & (s_i != 0)  # (batch, seq_len-1)

        # Считаем cross-entropy без усреднения
        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_targets.view(-1),
            reduction="none",
        )  # (batch * (seq_len-1),)

        loss_per_token = loss_per_token.view(shift_targets.size(0), -1)

        # Применяем маску и усредняем
        loss_mask = loss_mask.float()
        loss = (loss_per_token * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

        return loss
