"""Request-local timings and model call counts without request content."""
from contextlib import contextmanager
from time import perf_counter


class RequestMetrics:
    def __init__(self):
        self.started = perf_counter()
        self.elapsed = {}
        self.calls = {}

    @contextmanager
    def stage(self, name, *, model_call=False):
        started = perf_counter()
        if model_call:
            self.calls[name] = self.calls.get(name, 0) + 1
        try:
            yield
        finally:
            self.elapsed[name] = self.elapsed.get(name, 0) + (perf_counter() - started)

    def summary(self):
        now = perf_counter()
        return {
            "total_ms": round((now - self.started) * 1000, 3),
            "stage_ms": {name: round(seconds * 1000, 3) for name, seconds in self.elapsed.items()},
            "model_calls": dict(self.calls),
        }
