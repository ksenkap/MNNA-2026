import torch
import triton
import triton.language as tl
import math


@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, Out, L,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    Z, H, N_CTX,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_offset = off_z * stride_qz + off_h * stride_qh
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    if IS_CAUSAL:
        hi = tl.cdiv((start_m + 1) * BLOCK_M, BLOCK_N)
    else:
        hi = tl.cdiv(N_CTX, BLOCK_N)

    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh

    for start_n in range(0, hi * BLOCK_N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = K + k_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k_mask = offs_n[None, :] < N_CTX
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        s = tl.dot(q, k) * scale

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask, s, float('-inf'))

        n_mask = offs_n[None, :] < N_CTX
        s = tl.where(n_mask, s, float('-inf'))

        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_mask = offs_n[:, None] < N_CTX
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)

        m_i = m_new

    acc = acc / l_i[:, None]
    L_i = m_i + tl.log(l_i)

    o_offset = off_z * stride_oz + off_h * stride_oh
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask)

    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, L_i, mask=offs_m < N_CTX)


def flash_attn_forward(q, k, v, causal=True):
    """
    q, k, v: (B, H, N, D), dtype fp16
    Возвращает: out (B, H, N, D), L (B, H, N) — logsumexp
    """
    B, H, N, D = q.shape
    assert D in {16, 32, 64, 128, 256}, f"D={D} not supported"
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    L = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
    scale = 1.0 / math.sqrt(D)

    # Подбор блоков под SMEM T4 (64 KB)
    if D <= 64:
        BLOCK_M, BLOCK_N = 128, 64
    elif D <= 128:
        BLOCK_M, BLOCK_N = 64, 64
    else:
        BLOCK_M, BLOCK_N = 64, 32

    grid = (triton.cdiv(N, BLOCK_M), B * H)

    _flash_attn_fwd_kernel[grid](
        q, k, v, out, L,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, N, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D,
        IS_CAUSAL=causal,
    )
    return out, L
