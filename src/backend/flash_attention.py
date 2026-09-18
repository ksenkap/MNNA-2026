import torch
import torch.nn as nn

from src.backend.flash_attention_forward import flash_attn_forward
from src.backend.flash_attention_backward import flash_attn_backward


class FlashAttnFunc(torch.autograd.Function):
    """
    Кастомная autograd-функция для Flash Attention.

    forward:  вызывает наш forward-кернел, сохраняет q, k, v, out, L
    backward: вызывает наш backward-кернел (два кернела внутри)
    """
    @staticmethod
    def forward(ctx, q, k, v, causal):
        out, L = flash_attn_forward(q, k, v, causal=causal)
        ctx.save_for_backward(q, k, v, out, L)
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, L = ctx.saved_tensors
        dq, dk, dv = flash_attn_backward(dout, q, k, v, out, L, causal=ctx.causal)
        return dq, dk, dv, None   # None для causal (это bool, не тензор)


class FlashAttention(nn.Module):
    """
    nn.Module-обёртка над FlashAttnFunc.

    Args:
        causal: bool — применять ли causal-маску
    """
    def __init__(self, causal: bool = True):
        super().__init__()
        self.causal = causal

    def forward(self, q, k, v):
        return FlashAttnFunc.apply(q, k, v, self.causal)
