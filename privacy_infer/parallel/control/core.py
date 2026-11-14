from __future__ import annotations

import gc

import torch

from ..config import EngineConfig
from ..data import EngineCoreOutput, KVManager, ScheduledBatch, Sequence
from .executor import create_executor
from .scheduler import Scheduler


class EngineCore:
    _MAX_IDLE_STEPS = 1024

    def __init__(
        self,
        config: EngineConfig,
        *,
        eos_token_id: int,
        device: torch.device,
        dtype: torch.dtype,
        executor=None,
    ) -> None:
        self.config = config
        self.eos_token_id = eos_token_id

        self.executor = executor or create_executor(config, device, dtype)

        self._idle_steps = 0
        self.async_decode = None
        try:
            self._init_kv()
        except BaseException:
            self.executor.destroy()
            raise
        if config.enable_zos:
            from .async_decode import CoreAsyncDecode

            self.async_decode = CoreAsyncDecode(self)

    def _build_scheduler(self) -> Scheduler:
        return Scheduler(
            self.kv,
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            eos_token_id=self.eos_token_id,
            mix_prefill_decode=self.config.mix_prefill_decode,
        )

    def _init_kv(self) -> None:
        self.kv: KVManager = self.executor.init_kv()
        self.scheduler = self._build_scheduler()
        self.last_num_cached_tokens = 0

    def add_request(self, seq: Sequence) -> None:
        self.scheduler.add(seq)

    def has_unfinished(self) -> bool:
        return not self.scheduler.is_finished()

    def step(self) -> tuple[list[EngineCoreOutput], int]:
        if self.async_decode is not None:
            return self.async_decode.step()
        return self._step_sync()

    def _step_sync(self) -> tuple[list[EngineCoreOutput], int]:
        sched: ScheduledBatch = self.scheduler.schedule()

        if sched.is_empty():
            if not self.scheduler.is_finished():
                self._idle_steps += 1
                if self._idle_steps >= self._MAX_IDLE_STEPS:
                    raise RuntimeError(
                        "scheduler returned empty for too long while requests remain "
                        f"(idle_steps={self._idle_steps}). Likely KV pool exhaustion "
                        f"(free_blocks={self.kv.num_free})."
                    )
            return [], 0

        self._idle_steps = 0

        n_p, n_d = self.executor.run(sched)
        self.last_num_cached_tokens = self.kv.last_num_cached_tokens

        prefill_toks = sum(s.num_scheduled_tokens for s in sched.prefills)
        metric = prefill_toks + len(sched.decodes)

        p_ids, d_ids = self.executor.read_samples(n_p, n_d)

        outputs = self.scheduler.update(sched, p_ids, d_ids)
        return outputs, metric

    def destroy(self) -> None:
        if self.async_decode is not None:
            self.async_decode.close()
        self.executor.destroy()
        self.kv = None
        self.scheduler = None
        gc.collect()
        torch.cuda.empty_cache()

    def clear_batch(self) -> None:
        self.executor.collective_rpc("clear_batch")
        self._init_kv()
        self._idle_steps = 0
