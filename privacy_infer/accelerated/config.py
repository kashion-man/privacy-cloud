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
    device: str = "cuda:0"
    enable_zos: bool = False

    hf_config: PretrainedConfig = field(init=False)

    def __post_init__(self) -> None:
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
        self.hf_config = AutoConfig.from_pretrained(
            self.model, local_files_only=True, trust_remote_code=False
        )
        self.context_len = min(
            self.context_len, getattr(self.hf_config, "max_position_embeddings", self.context_len)
        )

    @property
    def block_size(self) -> int:
        return self.kvcache_block_size
