from .config import EngineConfig
from .sampling_params import SamplingParams

__all__ = ["Engine", "EngineConfig", "SamplingParams"]


def __getattr__(name):
    if name == "Engine":
        from .control.engine import Engine

        return Engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
