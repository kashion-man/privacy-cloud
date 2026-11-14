import torch

from ..config import EngineConfig
from ..data import Batch, KVManager, ScheduledBatch
from ..data.sequence import Sequence
from .cuda_graph import CUDAGraphRunner
from .decode_inputs import advance_decode
from .layers.attention import AttnInputs, AttnMetadata, attention_context
from .sampler import Sampler


class ModelRunner:
    def __init__(
        self, model, kv: KVManager, sampler: Sampler, *, config: EngineConfig, device: torch.device
    ) -> None:
        self.model = model
        self.kv = kv
        self.sampler = sampler.to(device=device)
        self.config = config
        self.device = device
        self.block_size = kv.block_size
        self.kv_cache = kv.kv_cache
        self.dtype = kv.dtype
        self._meta = AttnMetadata(kv_cache=self.kv_cache)

        max_bs = config.max_num_seqs
        max_blocks = (config.context_len + self.block_size - 1) // self.block_size
        self.batch = Batch(
            max_bs=max_bs,
            max_tokens=max(config.max_num_batched_tokens, max_bs),
            max_blocks=max_blocks,
            vocab_size=config.hf_config.vocab_size,
            device=device,
        )

        self._copy_stream = torch.cuda.Stream(device=device)

        self._copy_event = torch.cuda.Event()

        self._sample_cpu = torch.empty(2 * max_bs, dtype=torch.int64, pin_memory=True)

        self.cuda_graph_runner: CUDAGraphRunner | None = None
        if not config.enforce_eager:
            self.cuda_graph_runner = CUDAGraphRunner(
                model=self.model,
                batch=self.batch.decode,
                meta=self._meta,
                max_batch_size=min(max_bs, 256),
                vocab_size=config.hf_config.vocab_size,
                dtype=self.dtype,
                device=device,
            )
            self.cuda_graph_runner.capture()

        self.prefill_fn = None
        if config.torch_compile:
            self.prefill_fn = self._build_compiled_prefill()

    @torch.inference_mode()
    def run(self, sched: ScheduledBatch) -> int:
        seqs = sched.sequences
        samples = self._run_prefill(seqs) if sched.prefills else self._run_decode(seqs)
        n = len(seqs)
        self._copy_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._copy_stream):
            self._sample_cpu[:n].copy_(samples, non_blocking=True)
        self._copy_event.record(self._copy_stream)
        return n

    def read_samples(self, n: int) -> list[int]:
        self._copy_event.synchronize()
        return self._sample_cpu[:n].tolist()

    def _run_decode(
        self,
        seqs: list[Sequence],
        input_ids_gpu=None,
        *,
        token_offset=0,
        refresh_block_tables=False,
    ) -> torch.Tensor:
        decode = self.batch.decode
        bs = len(seqs)
        if input_ids_gpu is not None and token_offset == 1:
            if refresh_block_tables:
                decode.refresh_tables(seqs)
            advance_decode(decode, input_ids_gpu, bs, self.block_size)
        else:
            self.batch.build_decode(seqs, token_offset=token_offset, input_ids_gpu=input_ids_gpu)
        runner = self.cuda_graph_runner
        if runner is not None and runner.can_use(bs):
            logits, samples = runner.replay(bs)
            if self.batch.all_greedy:
                return samples
        else:
            attn = AttnInputs(
                slot_mapping=decode.slot_mapping[:bs],
                is_prefill=False,
                cache_seqlens=decode.cache_seqlens[:bs],
                block_tables=decode.block_tables[:bs],
            )
            with attention_context(self._meta):
                hidden = self.model(decode.input_ids[:bs], decode.positions[:bs], attn)
            logits = self.model.compute_logits(hidden)
        return self._sample(bs, logits)

    def _run_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        n_tokens, max_q, max_k = self.batch.build_prefill(seqs)
        bs = len(seqs)
        attn = AttnInputs(
            slot_mapping=self.batch.slot_mapping[:n_tokens],
            is_prefill=True,
            cu_seqlens_q=self.batch.cu_seqlens_q[: bs + 1],
            cu_seqlens_k=self.batch.cu_seqlens_k[: bs + 1],
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            block_tables=self.batch.block_tables[:bs],
        )
        forward = self.prefill_fn if self.prefill_fn is not None else self.model
        with attention_context(self._meta):
            hidden = forward(self.batch.input_ids[:n_tokens], self.batch.positions[:n_tokens], attn)
        if not any(s.need_sample for s in seqs):
            return torch.zeros(bs, dtype=torch.int64, device=self.device)
        last = self.batch.cu_seqlens_q[1 : bs + 1] - 1
        logits = self.model.compute_logits(hidden[last])
        return self._sample(bs, logits)

    def _sample(self, bs: int, logits: torch.Tensor) -> torch.Tensor:
        return self.sampler(
            logits,
            self.batch.temperatures[:bs],
            top_ps=self.batch.top_ps[:bs],
            top_ks=self.batch.top_ks[:bs],
            min_ps=self.batch.min_ps[:bs],
            all_greedy=self.batch.all_greedy,
            any_greedy=self.batch.any_greedy,
            use_top_k=self.batch.use_top_k,
            use_top_p=self.batch.use_top_p,
            use_min_p=self.batch.use_min_p,
            max_top_k=self.batch.max_top_k,
        )

    def _build_compiled_prefill(self):
        fn = torch.compile(
            self._prefill_forward,
            mode=self.config.compile_mode,
            dynamic=self.config.compile_dynamic,
        )

        n = min(256, max(1, self.config.context_len // 2))
        ids = torch.arange(1, n + 1, dtype=torch.int64, device=self.device)
        pos = torch.arange(n, dtype=torch.int64, device=self.device)
        cu = torch.tensor([0, n], dtype=torch.int32, device=self.device)
        sm = torch.arange(n, dtype=torch.int32, device=self.device)
        offset = min(n, self.config.context_len - n)
        cu_k = torch.tensor([0, n + offset], dtype=torch.int32, device=self.device)
        tables = torch.arange(
            (n + offset + self.block_size - 1) // self.block_size,
            dtype=torch.int32,
            device=self.device,
        ).unsqueeze(0)
        with attention_context(self._meta), torch.inference_mode():
            attn = AttnInputs(sm, True, cu, cu, n, n, tables)
            fn(ids, pos, attn)
            cached = AttnInputs(sm + offset, True, cu, cu_k, n, n + offset, tables)
            fn(ids, pos + offset, cached)
        torch.cuda.synchronize()
        self.kv_cache.zero_()
        return fn

    def _prefill_forward(self, input_ids, positions, attn):
        return self.model(input_ids, positions, attn)

    def destroy(self) -> None:
        if self.cuda_graph_runner is not None:
            self.cuda_graph_runner.destroy()
            self.cuda_graph_runner = None
        self.prefill_fn = None
