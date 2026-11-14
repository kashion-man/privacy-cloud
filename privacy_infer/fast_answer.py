"""Qwen3 answers through the bundled GPU inference engine."""
from contextlib import redirect_stdout
import json
import math
from pathlib import Path
import sys
from threading import RLock

from .errors import PrivacyError


# Upstream uses a module-level FlashAttention context and process-wide CUDA
# settings. Only one accelerated call may touch them at a time.
_ENGINE_LOCK = RLock()


class FastAnswer:
    def __init__(self, config):
        import torch

        if not torch.cuda.is_available():
            raise PrivacyError("The accelerated engine requires a CUDA device with FlashAttention and FlashInfer")
        model = Path(config.model_path).expanduser().resolve(strict=True)
        for name in ("config.json", "tokenizer_config.json"):
            metadata = json.loads((model / name).read_text(encoding="utf-8"))
            if metadata.get("auto_map") or name == "config.json" and metadata.get("model_type") != "qwen3":
                raise PrivacyError("The accelerated engine requires a standard local dense Qwen3 checkpoint")
        self.max_batch = config.max_batch_size
        self.prefix_backend = config.prefix_cache
        self.torch = torch
        self.engine = None
        with _ENGINE_LOCK, redirect_stdout(sys.stderr):
            from .accelerated.config import EngineConfig
            from .accelerated.control.engine import Engine

            settings = EngineConfig(
                model=str(model), context_len=config.context_len,
                max_num_seqs=self.max_batch, max_num_batched_tokens=config.prefill_chunk_tokens,
                num_kvcache_blocks=math.ceil(config.context_len / 256) * self.max_batch,
                prefix_backend=self.prefix_backend, enforce_eager=not config.cuda_graph,
                torch_compile=config.compile_prefill, enable_zos=config.async_decode, dtype="bfloat16",
            )
            if settings.context_len != config.context_len or settings.hf_config.model_type != "qwen3":
                raise PrivacyError("The accelerated engine requires dense Qwen3 and a valid context length")
            reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            try:
                self.engine = Engine(settings)
            finally:
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduction
        self.tokenizer = self.engine.tokenizer
        self.engine.scheduler.eos = frozenset([self.engine.eos_token_id])

    def encode(self, system, text):
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.tokenizer.encode(rendered, add_special_tokens=False)

    def generate(self, prompts, max_new_tokens):
        """Generate one explicitly grouped batch, with cache state scoped to this call."""
        from .accelerated.sampling_params import SamplingParams

        if not prompts or len(prompts) > self.max_batch or type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("Invalid accelerated engine request count or generation budget")
        engine = self.engine
        if any(not ids or len(ids) + max_new_tokens > engine.context_len for ids in prompts):
            raise ValueError("Input plus output exceeds context limit; truncation is disabled")
        if max_new_tokens == 0:
            return [""] * len(prompts)
        with _ENGINE_LOCK, self.torch.inference_mode(), redirect_stdout(sys.stderr):
            reduction = self.torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            self.torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            try:
                params = SamplingParams(max_tokens=max_new_tokens, temperature=0)
                sequences = [engine.add_request(ids, params) for ids in prompts]
                while not engine.scheduler.is_finished():
                    engine.step()
                return [self._decode(seq.token_ids[seq.num_prompt_tokens:]) for seq in sequences]
            finally:
                try:
                    self._clear_request_state()
                finally:
                    self.torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduction

    def _decode(self, ids):
        # Keep the legacy EOS and UTF-8 fragment behavior.
        text = []
        pending = []
        for token in ids:
            if token == self.engine.eos_token_id:
                break
            pending.append(token)
            piece = self.tokenizer.decode(pending, skip_special_tokens=False)
            if piece and not piece.endswith("\ufffd"):
                text.append(piece)
                pending.clear()
        if pending:
            text.append(self.tokenizer.decode(pending, skip_special_tokens=False))
        return "".join(text)

    def _clear_request_state(self):
        engine = self.engine
        if engine.async_decode is not None:
            engine.async_decode.close()
        self.torch.cuda.synchronize(engine.device)
        scheduler = engine.scheduler
        for seq in (*scheduler.waiting, *scheduler.running):
            engine.kv.deallocate(seq, publish=False)
        for block_id in tuple(engine.kv.blocks.cached_ids):
            engine.kv.blocks._evict_cached(block_id)
        # Drop the index as well as the page metadata, including radix nodes.
        engine.kv.index = type(engine.kv.index)()
        engine.kv.kv_cache.zero_()
        engine.scheduler = engine._build_scheduler()
        engine.scheduler.eos = frozenset([engine.eos_token_id])
        engine._idle_steps = 0
        if engine.async_decode is not None:
            engine.async_decode._tables_dirty = True
        self.torch.cuda.synchronize(engine.device)

    def close(self):
        with _ENGINE_LOCK:
            if self.engine is not None:
                self.engine.destroy()
                self.engine = None
