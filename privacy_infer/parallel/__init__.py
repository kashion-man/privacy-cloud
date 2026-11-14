from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__all__ = ["Engine", "EngineConfig", "SamplingParams"]

if TYPE_CHECKING:
    from .config import EngineConfig
    from .control.engine import Engine
    from .sampling_params import SamplingParams

_LAZY = {
    "Engine": (".control.engine", "Engine"),
    "EngineConfig": (".config", "EngineConfig"),
    "SamplingParams": (".sampling_params", "SamplingParams"),
}


def __getattr__(name: str) -> Any:
    try:
        module, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(module, __name__), attr)
