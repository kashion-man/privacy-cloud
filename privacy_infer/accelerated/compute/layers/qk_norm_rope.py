import torch
import triton
import triton.language as tl


@triton.jit
def _qk_norm_rope(
    Q,
    K,
    QW,
    KW,
    POS,
    CACHE,
    QO,
    KO,
    QS: tl.constexpr,
    KS: tl.constexpr,
    N: tl.constexpr,
    QH: tl.constexpr,
    KH: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    valid = row < N * (QH + KH)
    token = row // (QH + KH)
    head = row % (QH + KH)
    is_query = head < QH
    source = tl.where(is_query, Q + token * QS + head * D, K + token * KS + (head - QH) * D)
    weight = tl.where(is_query, QW, KW)
    target = tl.where(is_query, QO + (token * QH + head) * D, KO + (token * KH + head - QH) * D)
    col = tl.arange(0, D)
    value = tl.load(source[:, None] + col[None, :], valid[:, None], other=0).to(tl.float32)
    gamma = tl.load(weight[:, None] + col[None, :]).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(value * value, 1) / D + EPS)
    # Keep the normalization output rounding before applying RoPE.
    normalized = (value * inv_rms[:, None] * gamma).to(QO.dtype.element_ty).to(tl.float32)
    half = tl.arange(0, D // 2)
    indices = tl.broadcast_to(half[None, :], (ROWS, D // 2))
    low = tl.gather(normalized, indices, 1)
    high = tl.gather(normalized, indices + D // 2, 1)
    position = tl.load(POS + token, valid, other=0)
    cosine = tl.load(CACHE + position[:, None] * D + half[None, :])
    sine = tl.load(CACHE + position[:, None] * D + half[None, :] + D // 2)
    tl.store(target[:, None] + half[None, :], tl.fma(low, cosine, -high * sine), valid[:, None])
    tl.store(
        target[:, None] + half[None, :] + D // 2, tl.fma(high, cosine, low * sine), valid[:, None]
    )


@torch.library.custom_op("privacy_cloud::qk_norm_rope", mutates_args=())
def qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    qo = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    ko = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    rows = 4
    _qk_norm_rope[(triton.cdiv(q.shape[0] * (q.shape[1] + k.shape[1]), rows),)](
        q,
        k,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        qo,
        ko,
        q.stride(0),
        k.stride(0),
        q.shape[0],
        q.shape[1],
        k.shape[1],
        q.shape[2],
        eps,
        rows,
        num_warps=4,
    )
    return qo, ko


@qk_norm_rope.register_fake
def _qk_norm_rope_fake(q, k, q_weight, k_weight, positions, cos_sin_cache, eps):
    return torch.empty(q.shape, dtype=q.dtype, device=q.device), torch.empty(
        k.shape, dtype=k.dtype, device=k.device
    )
