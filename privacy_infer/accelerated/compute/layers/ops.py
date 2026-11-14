import torch
from flashinfer import apply_rope_with_cos_sin_cache as _flashinfer_rope
from flashinfer.activation import silu_and_mul as _flashinfer_silu_and_mul
from flashinfer.norm import fused_add_rmsnorm as _flashinfer_fused_rmsnorm
from flashinfer.norm import rmsnorm as _flashinfer_rmsnorm


@torch.library.custom_op("privacy_cloud::silu_and_mul", mutates_args=())
def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    return _flashinfer_silu_and_mul(x)


@silu_and_mul.register_fake
def _silu_and_mul_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty((x.shape[0], x.shape[1] // 2), dtype=x.dtype, device=x.device)


@torch.library.custom_op("privacy_cloud::rotary_embedding", mutates_args=())
def rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n, n_qh = q.shape[:2]
    n_kh = k.shape[1]

    qf = q.reshape(n, n_qh * head_size).contiguous()
    kf = k.reshape(n, n_kh * head_size).contiguous()

    qo, ko = _flashinfer_rope(positions, qf, kf, head_size, cos_sin_cache, is_neox=True)

    return qo.view_as(q), ko.view_as(k)


@rotary_embedding.register_fake
def _rotary_embedding_fake(q, k, positions, cos_sin_cache, head_size):
    return q, k


@torch.library.custom_op("privacy_cloud::rmsnorm", mutates_args=())
def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return _flashinfer_rmsnorm(x, weight, eps, enable_pdl=False)


@rmsnorm.register_fake
def _rmsnorm_fake(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("privacy_cloud::fused_add_rmsnorm", mutates_args=("x", "residual"))
def fused_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    _flashinfer_fused_rmsnorm(x, residual, weight, eps, enable_pdl=False)


@fused_add_rmsnorm.register_fake
def _fused_add_rmsnorm_fake(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    pass
