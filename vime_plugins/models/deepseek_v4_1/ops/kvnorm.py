# Adapted from https://github.com/radixark/miles/blob/b133ce9cac3c8b4d9969d20d6a03117d545c2865/miles_plugins/models/deepseek_v4_1/ops/kvnorm.py
import torch

from vime_plugins.models.deepseek_v4_1.kernels.rope import apply_rotary_emb
from vime_plugins.models.deepseek_v4_1.ops.qat import fp8_simulate_qat

# The forward keeps the fake-quantized value; it does not replay a rollout engine's KV cache kernels bit for bit.


def kv_norm_rope(
    kv: torch.Tensor, weight: torch.Tensor, eps: float, freqs_cis: torch.Tensor, rope_dim: int
) -> torch.Tensor:
    x = kv.float()
    data = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.float()).contiguous()
    apply_rotary_emb(data[..., -rope_dim:], freqs_cis)
    return data.to(kv.dtype)


def kv_norm_rope_fp8(
    kv: torch.Tensor, weight: torch.Tensor, eps: float, freqs_cis: torch.Tensor, rope_dim: int
) -> torch.Tensor:
    out = kv_norm_rope(kv, weight, eps, freqs_cis, rope_dim).clone()
    out[..., :-rope_dim] = fp8_simulate_qat(out[..., :-rope_dim], 64)
    return out


def compressed_kv_stored(latent: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """The compressed latent as the engine's fp8 pool returns it: the fp4/E4M3 values are requantized to fp8 per 64."""
    out = latent.clone()
    out[..., :-rope_dim] = fp8_simulate_qat(out[..., :-rope_dim], 64)
    return out
