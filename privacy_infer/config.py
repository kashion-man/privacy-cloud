import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    model_path: str
    context_len: int = 4096
    ner_max_new_tokens: int = 512
    relevance_threshold: float = 0.35
    cloud_model: str = "deepseek-flash"
    cloud_timeout: float = 60
    cloud_max_tokens: int = 1024
    max_input_chars: int = 8000
    privacy_mode: str = "redact"
    stamp_base_epsilon: float = 50.0
    stamp_group_epsilons: tuple[float, float, float, float] | None = None
    stamp_max_tokens: int = 512
    stamp_decode_batch_size: int = 32
    stamp_vocab_chunk_size: int = 4096
    sensitive_importance: str = "prosan"
    prosan_threshold: float = 0.5
    prosan_reference_tokens: int = 64
    prosan_max_tokens: int = 1024
    prosan_checkpointing: bool = True
    answer_backend: str = "eager"
    dzyy_engine_path: str | None = None
    prefill_chunk_tokens: int = 512
    max_batch_size: int = 2
    prefix_cache: str = "none"
    cuda_graph: bool = True
    compile_prefill: bool = False
    async_decode: bool = True

    def __post_init__(self):
        if self.answer_backend not in {"eager", "accelerated"}:
            raise ValueError("Unknown local answer backend")
        if self.answer_backend == "accelerated" and not self.dzyy_engine_path:
            raise ValueError("Accelerated answers require the local dzyy-vllm engine path")
        if self.prefix_cache not in {"none", "hash", "radix"}:
            raise ValueError("Invalid accelerated engine prefix backend")
        if type(self.prefill_chunk_tokens) is not int or not 2 <= self.prefill_chunk_tokens <= 8192:
            raise ValueError("Invalid accelerated engine prefill token budget")
        if type(self.max_batch_size) is not int or not 1 <= self.max_batch_size <= 8:
            raise ValueError("Invalid accelerated engine batch size")
        if any(type(flag) is not bool for flag in (self.cuda_graph, self.compile_prefill, self.async_decode)):
            raise ValueError("Acceleration switches must be boolean")
        for value in (self.context_len, self.ner_max_new_tokens,
                      self.cloud_max_tokens, self.max_input_chars, self.stamp_max_tokens,
                      self.stamp_decode_batch_size, self.stamp_vocab_chunk_size,
                      self.prosan_reference_tokens, self.prosan_max_tokens):
            if type(value) is not int or value <= 0:
                raise ValueError("Limits must be positive integers")
        if self.ner_max_new_tokens >= self.context_len:
            raise ValueError("NER output budget must be below context length")
        if not math.isfinite(self.relevance_threshold) or not -1 <= self.relevance_threshold <= 1:
            raise ValueError("Invalid cosine threshold")
        if not math.isfinite(self.cloud_timeout) or self.cloud_timeout <= 0:
            raise ValueError("Timeout must be finite and positive")
        if not self.cloud_model.strip():
            raise ValueError("Cloud model must be nonempty")
        if self.privacy_mode not in {"redact", "stamp"}:
            raise ValueError("Unknown privacy mode")
        if self.sensitive_importance not in {"prosan", "cosine"}:
            raise ValueError("Unknown sensitive importance method")
        if not math.isfinite(self.prosan_threshold) or not 0 <= self.prosan_threshold <= 1:
            raise ValueError("ProSan threshold must be between zero and one")
        if type(self.prosan_checkpointing) is not bool:
            raise ValueError("ProSan checkpointing must be boolean")
        if self.prosan_reference_tokens >= self.context_len:
            raise ValueError("ProSan reference budget exceeds context limit")
        if isinstance(self.stamp_base_epsilon, bool) or not math.isfinite(self.stamp_base_epsilon) or self.stamp_base_epsilon < 0:
            raise ValueError("STAMP base epsilon must be finite and nonnegative")
        if self.stamp_group_epsilons is not None:
            values = tuple(self.stamp_group_epsilons)
            if len(values) != 4 or any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in values):
                raise ValueError("STAMP requires four finite nonnegative budgets")
            object.__setattr__(self, "stamp_group_epsilons", values)
        if not all(math.isfinite(v * self.stamp_max_tokens) for v in self.stamp_budgets().values()):
            raise ValueError("STAMP budget exceeds numerical range")

    def stamp_budgets(self):
        values = self.stamp_group_epsilons
        if values is None:
            values = tuple(self.stamp_base_epsilon * r for r in (2, 1, 4, 3))
        return dict(zip(("G1", "G2", "G3", "G4"), values))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
