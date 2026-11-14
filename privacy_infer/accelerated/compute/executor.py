import copy
import gc
import io
from contextlib import redirect_stdout

import torch

from ..config import EngineConfig
from ..data import KVManager, ScheduledBatch
from .layers import RotaryEmbedding
from .model_runner import ModelRunner
from .models.loader import default_dtype_context, load_model, skip_param_init
from .models.qwen3 import Qwen3ForCausalLM
from .sampler import Sampler


class ModelExecutor:
    def __init__(self, config: EngineConfig, device: torch.device, dtype: torch.dtype) -> None:
        self.config = config
        self.device = device
        self.dtype = dtype
        self.model = self._load_model()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.sampler = Sampler()
        self._mem_baseline = torch.cuda.memory_reserved()

        self.kv: KVManager | None = None
        self.runner: ModelRunner | None = None

    def _load_model(self) -> torch.nn.Module:
        with default_dtype_context(self.dtype), skip_param_init():
            model = Qwen3ForCausalLM(self.config.hf_config)
        load_model(model, self.config.model)
        model = model.to(dtype=self.dtype, device=self.device).eval()
        for module in model.modules():
            if isinstance(module, RotaryEmbedding):
                module.cos_sin_cache = module.cos_sin_cache.float()
        return model

    def _make_kv(self, num_blocks: int) -> KVManager:
        return KVManager(
            self.config.hf_config,
            num_blocks=num_blocks,
            block_size=self.config.block_size,
            device=self.device,
            dtype=self.dtype,
            prefix_backend=self.config.prefix_backend,
        )

    def _resolve_num_blocks(self, reserve_bytes: int) -> int:
        return KVManager.auto_num_blocks(
            self.config.hf_config,
            block_size=self.config.block_size,
            dtype=self.dtype,
            device=self.device,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            reserve_bytes=reserve_bytes,
        )

    def _probe_cg_footprint(self) -> int:
        probe_config = copy.copy(self.config)
        probe_config.torch_compile = False
        kv = self._make_kv(num_blocks=8)
        with redirect_stdout(io.StringIO()):
            runner = ModelRunner(
                self.model, kv, self.sampler, config=probe_config, device=self.device
            )
        torch.cuda.synchronize()
        footprint = torch.cuda.memory_reserved() - self._mem_baseline

        runner.destroy()
        del runner, kv
        gc.collect()
        torch.cuda.empty_cache()

        return int(footprint)

    def init_kv(self) -> KVManager:
        use_cg = not self.config.enforce_eager
        if self.config.num_kvcache_blocks > 0:
            num_blocks = self.config.num_kvcache_blocks
        else:
            reserve = self._probe_cg_footprint() if use_cg else 0
            num_blocks = self._resolve_num_blocks(reserve)
            self.config.num_kvcache_blocks = num_blocks
            if use_cg:
                print(
                    f"CG footprint measured: {reserve / 2**30:.3f} GiB; "
                    f"KV pool sized to remainder ({num_blocks} blocks)"
                )
        self.kv = self._make_kv(num_blocks)
        self.runner = ModelRunner(
            self.model, self.kv, self.sampler, config=self.config, device=self.device
        )
        return self.kv

    def reset_kv(self, prefix_backend: str | None = None) -> KVManager:
        if self.runner is not None:
            self.runner.destroy()
            self.runner = None
        self.kv = None
        gc.collect()
        torch.cuda.empty_cache()
        if prefix_backend is not None:
            self.config.prefix_backend = prefix_backend
        return self.init_kv()

    def run(self, sched: ScheduledBatch) -> int:
        assert self.runner is not None, "init_kv() must be called before run()"
        return self.runner.run(sched)

    def read_samples(self, n: int) -> list[int]:
        return self.runner.read_samples(n)

    @torch.inference_mode()
    def run_decode(self, seqs, input_ids_gpu=None, *, refresh_block_tables=False):
        return self.runner._run_decode(
            seqs,
            input_ids_gpu,
            token_offset=int(input_ids_gpu is not None),
            refresh_block_tables=refresh_block_tables,
        )

    def destroy(self) -> None:
        if self.runner is not None:
            self.runner.destroy()
        self.runner = None
        self.sampler = None
        self.kv = None
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
