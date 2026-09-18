import torch
import triton
import triton.language as tl
import math


# ---------------------------------------------------------------------------
# Кернел 1: dK, dV. Внешний цикл по блокам Q (i), накопление dK, dV.
# ---------------------------------------------------------------------------
@triton.jit
def _flash_attn_bwd_dkdv_kernel(
    Q, K, V, dO, dK, dV, L, D,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    stride_dkz, stride_dkh, stride_dkn, stride_dkd,
    stride_dvz, stride_dvh, stride_dvn, stride_dvd,
    Z, H, N_CTX,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh

    k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    k_mask = offs_n[:, None] < N_CTX
    k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)

    v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    v_mask = offs_n[:, None] < N_CTX
    v_block = tl.load(v_ptrs, mask=v_mask, other=0.0)

    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    if IS_CAUSAL:
        lo = (start_n * BLOCK_N) // BLOCK_M
    else:
        lo = 0
    hi = tl.cdiv(N_CTX, BLOCK_M)

    q_offset = off_z * stride_qz + off_h * stride_qh
    o_offset = off_z * stride_oz + off_h * stride_oh

    for start_m in range(lo * BLOCK_M, hi * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + tl.arange(0, BLOCK_M)

        q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        q_mask = offs_m[:, None] < N_CTX
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        do_ptrs = dO + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        do_mask = offs_m[:, None] < N_CTX
        do = tl.load(do_ptrs, mask=do_mask, other=0.0)

        l_ptrs = L + off_hz * N_CTX + offs_m
        l_mask = offs_m < N_CTX
        L_i = tl.load(l_ptrs, mask=l_mask, other=0.0)

        d_ptrs = D + off_hz * N_CTX + offs_m
        delta_i = tl.load(d_ptrs, mask=l_mask, other=0.0)

        s = tl.dot(q, tl.trans(k_block)) * scale

        if IS_CAUSAL:
            causal = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal, s, float('-inf'))
        n_mask = offs_n[None, :] < N_CTX
        s = tl.where(n_mask, s, float('-inf'))
        m_mask = offs_m[:, None] < N_CTX
        s = tl.where(m_mask, s, float('-inf'))

        p = tl.exp(s - L_i[:, None])

        dv += tl.dot(tl.trans(p).to(do.dtype), do)

        dp = tl.dot(do, tl.trans(v_block))
        ds = p * (dp - delta_i[:, None])

        dk += tl.dot(tl.trans(ds).to(q.dtype), q) * scale

    dk_offset = off_z * stride_dkz + off_h * stride_dkh
    dk_ptrs = dK + dk_offset + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
    tl.store(dk_ptrs, dk.to(dK.dtype.element_ty), mask=k_mask)

    dv_offset = off_z * stride_dvz + off_h * stride_dvh
    dv_ptrs = dV + dv_offset + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
    tl.store(dv_ptrs, dv.to(dV.dtype.element_ty), mask=v_mask)


# ---------------------------------------------------------------------------
# Кернел 2: dQ. Внешний цикл по блокам K/V (j), накопление dQ.
# ---------------------------------------------------------------------------
@triton.jit
def _flash_attn_bwd_dq_kernel(
    Q, K, V, dO, dQ, L, D,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    stride_dqz, stride_dqh, stride_dqm, stride_dqd,
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
    o_offset = off_z * stride_oz + off_h * stride_oh

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    do_ptrs = dO + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    do_mask = offs_m[:, None] < N_CTX
    do = tl.load(do_ptrs, mask=do_mask, other=0.0)

    l_ptrs = L + off_hz * N_CTX + offs_m
    l_mask = offs_m < N_CTX
    L_i = tl.load(l_ptrs, mask=l_mask, other=0.0)

    d_ptrs = D + off_hz * N_CTX + offs_m
    delta_i = tl.load(d_ptrs, mask=l_mask, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    if IS_CAUSAL:
        hi = tl.cdiv((start_m + 1) * BLOCK_M, BLOCK_N)
    else:
        hi = tl.cdiv(N_CTX, BLOCK_N)

    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh

    for start_n in range(0, hi * BLOCK_N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k_mask = offs_n[:, None] < N_CTX
        k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)

        v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_mask = offs_n[:, None] < N_CTX
        v_block = tl.load(v_ptrs, mask=v_mask, other=0.0)

        s = tl.dot(q, tl.trans(k_block)) * scale

        if IS_CAUSAL:
            causal = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal, s, float('-inf'))
        n_mask = offs_n[None, :] < N_CTX
        s = tl.where(n_mask, s, float('-inf'))
        m_mask = offs_m[:, None] < N_CTX
        s = tl.where(m_mask, s, float('-inf'))

        p = tl.exp(s - L_i[:, None])

        dp = tl.dot(do, tl.trans(v_block))
        ds = p * (dp - delta_i[:, None])

        dq += tl.dot(ds.to(q.dtype), k_block) * scale

    dq_offset = off_z * stride_dqz + off_h * stride_dqh
    dq_ptrs = dQ + dq_offset + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(dQ.dtype.element_ty), mask=q_mask)


# ---------------------------------------------------------------------------
# Враппер backward
# ---------------------------------------------------------------------------
def flash_attn_backward(dout, q, k, v, out, L, causal=True):
    B, H, N, D = q.shape
    assert D in {16, 32, 64, 128, 256}

    dout = dout.contiguous()
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = out.contiguous()
    L = L.contiguous()

    delta = (dout.float() * out.float()).sum(dim=-1).contiguous()

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    scale = 1.0 / math.sqrt(D)

    # Подбор блоков под SMEM T4 (64 KB)
    if D <= 64:
        BLOCK_M, BLOCK_N = 64, 64
    elif D <= 128:
        BLOCK_M, BLOCK_N = 32, 32
    else:
        BLOCK_M, BLOCK_N = 16, 16

    # --- Кернел 1: dK, dV ---
    grid_dkdv = (triton.cdiv(N, BLOCK_N), B * H)
    _flash_attn_bwd_dkdv_kernel[grid_dkdv](
        q, k, v, dout, dk, dv, L, delta,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        B, H, N, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D,
        IS_CAUSAL=causal,
    )

    # --- Кернел 2: dQ ---
    grid_dq = (triton.cdiv(N, BLOCK_M), B * H)
    _flash_attn_bwd_dq_kernel[grid_dq](
        q, k, v, dout, dq, L, delta,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        B, H, N, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D,
        IS_CAUSAL=causal,
    )

    return dq, dk, dv
