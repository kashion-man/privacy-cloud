import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_size, rotary_dim, max_position_embeddings, base):
        super().__init__()
        if not 0 < rotary_dim <= head_size or rotary_dim % 2:
            raise ValueError("rotary_dim must be positive, even and no larger than head_size")
        if rotary_dim != head_size:
            raise ValueError("Fused RoPE requires full head rotation")
        self.head_size, self.rotary_dim = head_size, rotary_dim
        frequencies = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        angles = torch.outer(
            torch.arange(max_position_embeddings, dtype=torch.float32), frequencies
        )
        self.register_buffer(
            "cos_sin_cache", torch.cat((angles.cos(), angles.sin()), -1), persistent=False
        )

    def _apply(self, fn, recurse=True):
        cache = self.cos_sin_cache
        super()._apply(fn, recurse)
        self.cos_sin_cache = cache.to(device=self.cos_sin_cache.device)
        return self

    @classmethod
    def from_config(cls, config):
        parameters = (
            getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
        )
        if parameters.get("rope_type", parameters.get("type", "default")) != "default":
            raise ValueError("This tutorial implements default RoPE only")
        head_dim = (
            getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        )
        theta = parameters.get("rope_theta", getattr(config, "rope_theta", 1_000_000))
        return cls(head_dim, head_dim, config.max_position_embeddings, theta)

    def forward(self, positions, query, key):
        return torch.ops.privacy_cloud.rotary_embedding(
            query, key, positions, self.cos_sin_cache, self.head_size
        )
