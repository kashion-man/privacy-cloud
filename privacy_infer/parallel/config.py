from __future__ import annotations

import os
from dataclasses import dataclass, field

from transformers import AutoConfig, PretrainedConfig


@dataclass
class EngineConfig:
    model: str

    context_len: int = 4096
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    mix_prefill_decode: bool = True

    kvcache_block_size: int = 256
    gpu_memory_utilization: float = 0.9
    num_kvcache_blocks: int = 0
    prefix_backend: str = "hash"

    enforce_eager: bool = True
    torch_compile: bool = False
    compile_mode: str = "reduce-overhead"
    compile_dynamic: bool = True
    dtype: str = "auto"
    enable_zos: bool = False

    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    nccl_port: int = 29500
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    device_ids: list[int] | None = None
    nccl_ifname: str = ""
    pp_layer_counts: list[int] | None = None
    rendezvous_path: str | None = None
    timeout: int = 120

    hf_config: PretrainedConfig = field(init=False)

    def __post_init__(self) -> None:
        if self.enable_zos and self.tp_size * self.pp_size * self.dp_size != 1:
            raise ValueError(
                "ZOS currently supports one GPU; distributed TP/PP/DP need rank-coordinated tickets"
            )
        self.model = os.path.expanduser(self.model)
        if not os.path.isdir(self.model):
            raise FileNotFoundError(
                f"Model path not found: {self.model}. "
                "Use a local directory (e.g. ~/huggingface/Qwen3-0.6B)."
            )
        if self.kvcache_block_size % 256 != 0:
            raise ValueError("kvcache_block_size must be a multiple of 256 (FA2 paged)")
        if self.prefix_backend not in ("none", "hash", "radix"):
            raise ValueError(f"prefix_backend must be none|hash|radix, got {self.prefix_backend!r}")
        if self.tp_size < 1 or self.pp_size < 1 or self.dp_size < 1:
            raise ValueError("tp_size / pp_size / dp_size must be >= 1")
        self.hf_config = AutoConfig.from_pretrained(
            self.model, trust_remote_code=False, local_files_only=True
        )
        hf = self.hf_config
        if hf.model_type != "qwen3":
            raise ValueError("Multi-GPU accelerated inference requires dense Qwen3")
        if any(value % self.tp_size for value in (
            hf.num_attention_heads, hf.num_key_value_heads, hf.intermediate_size
        )):
            raise ValueError("Tensor parallel size must divide Q heads, KV heads and MLP width")
        if self.pp_size > hf.num_hidden_layers or (
            self.pp_layer_counts is not None and (
                len(self.pp_layer_counts) != self.pp_size or min(self.pp_layer_counts) < 1
                or sum(self.pp_layer_counts) != hf.num_hidden_layers
            )
        ):
            raise ValueError("Invalid pipeline layer allocation")
        self.context_len = min(
            self.context_len, getattr(self.hf_config, "max_position_embeddings", self.context_len)
        )

    @property
    def world_size(self) -> int:
        return self.tp_size * self.pp_size

    @property
    def num_gpus(self) -> int:
        return self.dp_size * self.tp_size * self.pp_size

    @property
    def block_size(self) -> int:
        return self.kvcache_block_size
