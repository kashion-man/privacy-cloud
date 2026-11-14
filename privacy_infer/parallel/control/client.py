from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from ..config import EngineConfig
from ..data import EngineCoreOutput, Sequence
from .core import EngineCore


class EngineCoreClient(ABC):
    @abstractmethod
    def add_request(self, seq: Sequence) -> None: ...

    @abstractmethod
    def step(self) -> tuple[list[EngineCoreOutput], int]: ...

    @abstractmethod
    def has_unfinished(self) -> bool: ...

    @abstractmethod
    def destroy(self) -> None: ...

    @abstractmethod
    def clear_batch(self) -> None: ...


class InprocClient(EngineCoreClient):
    def __init__(self, engine_core: EngineCore) -> None:
        self.engine_core = engine_core

    def add_request(self, seq: Sequence) -> None:
        self.engine_core.add_request(seq)

    def step(self) -> tuple[list[EngineCoreOutput], int]:
        return self.engine_core.step()

    def has_unfinished(self) -> bool:
        return self.engine_core.has_unfinished()

    def destroy(self) -> None:
        self.engine_core.destroy()

    def clear_batch(self) -> None:
        self.engine_core.clear_batch()


def make_client(
    config: EngineConfig,
    *,
    eos_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
    executor=None,
) -> EngineCoreClient:
    if config.dp_size > 1:
        from .mp_client import MPClient

        if executor is not None:
            raise ValueError("dp_size>1 uses MPClient replicas; cannot inject a shared executor")
        return MPClient(config, eos_token_id=eos_token_id, device=device, dtype=dtype)
    core = EngineCore(
        config, eos_token_id=eos_token_id, device=device, dtype=dtype, executor=executor
    )
    return InprocClient(core)
