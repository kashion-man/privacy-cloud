import gc
import warnings
from collections.abc import Iterator
from dataclasses import replace
import torch
from tokenizers.decoders import DecodeStream
from transformers import AutoTokenizer, GenerationConfig

from ..compute.executor import ModelExecutor
from ..config import EngineConfig
from ..data import Sequence
from ..sampling_params import SamplingParams
from .scheduler import Scheduler


class Engine:
    _MAX_IDLE_STEPS = 1024

    def __init__(self, config: EngineConfig):
        self.config = config
        self.device = torch.device(config.device)
        torch.cuda.set_device(self.device)
        self.dtype = self._resolve_dtype()

        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        self.tokenizer = self._load_tokenizer()
        self.eos_token_id = int(self.tokenizer.eos_token_id)
        eos = GenerationConfig.from_pretrained(config.model, local_files_only=True).eos_token_id
        self.eos_token_ids = frozenset([eos] if isinstance(eos, int) else eos)
        self.context_len = config.context_len
        self.block_size = config.block_size

        self.executor = ModelExecutor(config, self.device, self.dtype)
        self._idle_steps = 0
        self._init_kv()
        self.async_decode = None
        if config.enable_zos:
            from .async_decode import AsyncDecode

            self.async_decode = AsyncDecode(self)

    def _resolve_dtype(self) -> torch.dtype:
        if self.config.dtype != "auto":
            return getattr(torch, self.config.dtype)
        dt = getattr(self.config.hf_config, "torch_dtype", None)
        return dt if isinstance(dt, torch.dtype) else torch.float16

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model, local_files_only=True, trust_remote_code=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _build_scheduler(self) -> Scheduler:
        return Scheduler(
            self.kv,
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            eos_token_id=self.eos_token_ids,
            mix_prefill_decode=self.config.mix_prefill_decode,
        )

    def _init_kv(self) -> None:
        self.kv = self.executor.init_kv()
        self.scheduler = self._build_scheduler()
        self.last_num_cached_tokens = 0

    def reset_kv(self, prefix_backend: str | None = None) -> None:
        if self.async_decode is not None and self.async_decode.pending is not None:
            raise RuntimeError("drain in-flight decode before resetting KV")
        if prefix_backend is not None and prefix_backend not in ("none", "hash", "radix"):
            raise ValueError(f"invalid prefix_backend: {prefix_backend!r}")
        self.scheduler = None
        self.kv = None
        self.kv = self.executor.reset_kv(prefix_backend)
        self.scheduler = self._build_scheduler()
        self.last_num_cached_tokens = 0
        torch.cuda.empty_cache()

    def add_request(
        self, prompt: str | list[int], sampling_params: SamplingParams | None = None
    ) -> Sequence:
        sp = sampling_params or SamplingParams()

        token_ids = list(prompt) if isinstance(prompt, list) else self.tokenizer.encode(prompt)
        if not token_ids:
            raise ValueError("empty prompt")

        if len(token_ids) > self.context_len:
            warnings.warn(
                f"prompt length {len(token_ids)} > context_len={self.context_len}; "
                f"truncating to last {self.context_len} tokens",
                stacklevel=2,
            )
            token_ids = token_ids[-self.context_len :]

        max_ok = max(0, self.context_len - len(token_ids))
        if sp.max_tokens > max_ok:
            warnings.warn(
                f"max_tokens={sp.max_tokens} + prompt={len(token_ids)} exceeds "
                f"context_len={self.context_len}; clamping max_tokens to {max_ok}",
                stacklevel=2,
            )
            sp = replace(sp, max_tokens=max_ok)

        seq = Sequence.from_prompt(token_ids, block_size=self.block_size, sampling=sp)
        self.scheduler.add(seq)
        return seq

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        if self.async_decode is not None:
            return self.async_decode.step()
        return self.execute_batch(self.scheduler.schedule())

    def execute_batch(self, sched) -> tuple[list[tuple[int, list[int]]], int]:
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

        self.last_num_cached_tokens = self.kv.last_num_cached_tokens

        prefill_toks = sum(s.num_scheduled_tokens for s in sched.prefills)
        metric = prefill_toks + len(sched.decodes)

        n = self.executor.run(sched)
        ids = self.executor.read_samples(n)
        finished = self.scheduler.postprocess(sched.sequences, ids)

        outputs = [(seq.seq_id, seq.token_ids[seq.num_prompt_tokens :]) for seq in finished]
        return outputs, metric

    def generate_batch(
        self,
        prompts: list[str | list[int]],
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
    ) -> list[str]:
        sps = self._normalize_sampling_params(prompts, sampling_params)
        seqs = [self.add_request(p, sp) for p, sp in zip(prompts, sps)]
        finished_text: dict[int, str] = {}

        while not self.scheduler.is_finished():
            outputs, _ = self.step()
            for seq_id, completion_ids in outputs:
                finished_text[seq_id] = self.tokenizer.decode(
                    completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
        return [finished_text[seq.seq_id] for seq in seqs]

    def generate(
        self,
        prompt: str | list[int],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
    ) -> Iterator[str]:
        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_new_tokens,
        )
        seq = self.add_request(prompt, sp)
        emitted = 0
        decoded_chars = 0
        decoder = DecodeStream(skip_special_tokens=True)

        while not seq.is_finished:
            self.step()
            while emitted < seq.num_completion_tokens:
                tid = seq.token_ids[seq.num_prompt_tokens + emitted]
                emitted += 1
                text = decoder.step(self.tokenizer.backend_tokenizer, tid)
                if text:
                    decoded_chars += len(text)
                    yield text

        text = self.tokenizer.decode(
            seq.token_ids[seq.num_prompt_tokens :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if tail := text[decoded_chars:]:
            yield tail

    @staticmethod
    def _normalize_sampling_params(
        prompts: list[str | list[int]],
        sampling_params: SamplingParams | list[SamplingParams] | None,
    ) -> list[SamplingParams]:
        if sampling_params is None:
            return [SamplingParams() for _ in prompts]
        if isinstance(sampling_params, SamplingParams):
            return [sampling_params for _ in prompts]
        if len(sampling_params) != len(prompts):
            raise ValueError("sampling_params length must match prompts")
        return list(sampling_params)

    def destroy(self) -> None:
        if self.async_decode is not None:
            self.async_decode.close()
        executor = getattr(self, "executor", None)
        if executor is not None:
            executor.destroy()
        self.kv = None
        self.scheduler = None
        gc.collect()
        torch.cuda.empty_cache()
