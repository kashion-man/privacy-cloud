from __future__ import annotations
import os
from dataclasses import dataclass, field

from transformers import AutoConfig, PretrainedConfig


@dataclass
class EngineConfig:
    model: str
    context_len: int = 2048
    enforce_eager: bool = True
    dtype: str = "auto"
    hf_config: PretrainedConfig = field(init=False)

    def __post_init__(self) -> None:
        self.model = os.path.expanduser(self.model)

        self.hf_config = AutoConfig.from_pretrained(
            self.model, trust_remote_code=False, local_files_only=True
        )
        if not 0 < self.context_len <= self.hf_config.max_position_embeddings:
            raise ValueError("context_len must be positive and within the model position limit")
