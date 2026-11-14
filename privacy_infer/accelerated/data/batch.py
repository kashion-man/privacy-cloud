from dataclasses import dataclass, field

import torch

from .sequence import Sequence, slots_for_range


@dataclass
class ScheduledBatch:
    prefills: list[Sequence] = field(default_factory=list)
    decodes: list[Sequence] = field(default_factory=list)

    @property
    def sequences(self) -> list[Sequence]:
        return self.decodes + self.prefills

    def is_empty(self) -> bool:
        return not self.prefills and not self.decodes


class DecodeInputs:
    """One packed metadata upload, with two guarded pinned staging buffers."""

    def __init__(self, max_bs: int, max_blocks: int, device: torch.device):
        self.max_bs = max_bs
        words = 2 * max_bs + ((2 + max_blocks) * max_bs + 1) // 2
        self.storage = torch.zeros(words, dtype=torch.int64, device=device)

        def views(storage):
            ints = storage[2 * max_bs :].view(torch.int32)
            return (
                storage[:max_bs],
                storage[max_bs : 2 * max_bs],
                ints[:max_bs],
                ints[max_bs : 2 * max_bs],
                ints[2 * max_bs : 2 * max_bs + max_bs * max_blocks].view(max_bs, max_blocks),
            )

        (
            self.input_ids,
            self.positions,
            self.slot_mapping,
            self.cache_seqlens,
            self.block_tables,
        ) = views(self.storage)
        self.slot_mapping.fill_(-1)
        self._cpu = [torch.empty(words, dtype=torch.int64, pin_memory=True) for _ in range(2)]
        self._views = [views(cpu) for cpu in self._cpu]
        self._arrays = [tuple(v.numpy() for v in fields) for fields in self._views]
        self._raw = [cpu.numpy() for cpu in self._cpu]
        self._copied = [torch.cuda.Event() for _ in range(2)]
        self._used = [False, False]
        self._slot = 0

    def build(
        self,
        seqs: list[Sequence],
        *,
        token_offset: int = 0,
        input_ids_gpu: torch.Tensor | None = None,
    ) -> None:
        slot = self._slot
        self._slot ^= 1
        # The CPU must not overwrite a pinned source while its H2D is in flight.
        if self._used[slot]:
            self._copied[slot].synchronize()
        self._raw[slot].fill(0)
        ids, positions, slots, lengths, tables = self._arrays[slot]
        slots.fill(-1)  # CUDA Graph padding must never write a retired row's KV.
        tables.fill(-1)
        for row, seq in enumerate(seqs):
            length = len(seq) + token_offset
            position = length - 1
            ids[row] = seq.last_token if input_ids_gpu is None else 0
            positions[row] = position
            slots[row] = (
                seq.block_table[position // seq.block_size] * seq.block_size
                + position % seq.block_size
            )
            lengths[row] = length
            tables[row, : len(seq.block_table)] = seq.block_table
        start = self.max_bs if input_ids_gpu is not None else 0
        self.storage[start:].copy_(self._cpu[slot][start:], non_blocking=True)
        self._copied[slot].record()
        self._used[slot] = True
        if input_ids_gpu is not None:
            self.input_ids[: len(seqs)].copy_(input_ids_gpu, non_blocking=True)

    def refresh_tables(self, seqs: list[Sequence]) -> None:
        slot = self._slot
        self._slot ^= 1
        if self._used[slot]:
            self._copied[slot].synchronize()
        tables = self._arrays[slot][4]
        tables.fill(-1)
        for row, seq in enumerate(seqs):
            tables[row, : len(seq.block_table)] = seq.block_table
        self.block_tables.copy_(self._views[slot][4], non_blocking=True)
        self._copied[slot].record()
        self._used[slot] = True


class Batch:
    def __init__(
        self,
        *,
        max_bs: int,
        max_tokens: int,
        max_blocks: int,
        vocab_size: int,
        device: torch.device,
    ) -> None:
        self.max_bs = max_bs
        self.max_tokens = max_tokens
        self.max_blocks = max_blocks
        self.vocab_size = vocab_size
        self.device = device

        words = self._input_words(max_tokens, max_bs)
        self._inputs = torch.empty(words, dtype=torch.int64, device=device)
        self._inputs_cpu = torch.empty(words, dtype=torch.int64, pin_memory=True)
        self.temperatures = torch.zeros(max_bs, dtype=torch.float32, device=device)
        self.top_ps = torch.zeros(max_bs, dtype=torch.float32, device=device)
        self.top_ks = torch.zeros(max_bs, dtype=torch.int32, device=device)
        self.min_ps = torch.zeros(max_bs, dtype=torch.float32, device=device)
        self.all_greedy = True
        self.any_greedy = False
        self.use_top_k = False
        self.use_top_p = False
        self.use_min_p = False
        self.max_top_k = 0
        self.decode = DecodeInputs(max_bs, max_blocks, device)

    @staticmethod
    def _cp(dst: torch.Tensor, data: list, dtype: torch.dtype) -> None:
        cpu = torch.tensor(data, dtype=dtype, pin_memory=True)
        dst.copy_(cpu, non_blocking=True)

    def build_decode(
        self,
        seqs: list[Sequence],
        *,
        token_offset: int = 0,
        input_ids_gpu: torch.Tensor | None = None,
    ) -> None:
        self.decode.build(seqs, token_offset=token_offset, input_ids_gpu=input_ids_gpu)
        self._fill_sampling(seqs)

    def _input_words(self, tokens: int, rows: int) -> int:
        ints = tokens + 2 * (rows + 1) + rows * self.max_blocks
        return 2 * tokens + (ints + 1) // 2

    def _input_views(self, storage, tokens, rows):
        ints = storage[2 * tokens :].view(torch.int32)
        cu_start = tokens
        table_start = cu_start + 2 * (rows + 1)
        return (
            storage[:tokens],
            storage[tokens : 2 * tokens],
            ints[:tokens],
            ints[cu_start : cu_start + rows + 1],
            ints[cu_start + rows + 1 : table_start],
            ints[table_start : table_start + rows * self.max_blocks].view(rows, self.max_blocks),
        )

    def build_prefill(self, seqs: list[Sequence]) -> tuple[int, int, int]:
        n = sum(seq.num_scheduled_tokens for seq in seqs)
        rows = len(seqs)
        fields = self._input_views(self._inputs_cpu, n, rows)
        ids, positions, slots, cu_q, cu_k, tables = (field.numpy() for field in fields)
        cu_q[0] = cu_k[0] = 0
        tables.fill(-1)
        max_q = max_k = offset = 0
        for row, seq in enumerate(seqs):
            start = seq.num_cached_tokens
            count = seq.num_scheduled_tokens
            end = start + count
            target = slice(offset, offset + count)
            ids[target] = seq.token_ids[start:end]
            positions[target] = range(start, end)
            slots[target] = slots_for_range(seq.block_table, start, end, seq.block_size)
            offset += count
            cu_q[row + 1] = offset
            cu_k[row + 1] = cu_k[row] + end
            tables[row, : len(seq.block_table)] = seq.block_table
            max_q, max_k = max(max_q, count), max(max_k, end)
        words = self._input_words(n, rows)
        self._inputs[:words].copy_(self._inputs_cpu[:words], non_blocking=True)
        (
            self.input_ids,
            self.positions,
            self.slot_mapping,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
            self.block_tables,
        ) = self._input_views(self._inputs, n, rows)
        self._fill_sampling(seqs)
        return n, max_q, max_k

    def _fill_sampling(self, seqs: list[Sequence]) -> None:
        bs = len(seqs)
        vocab = self.vocab_size
        temps = [s.temperature for s in seqs]
        if all(t <= 1e-5 for t in temps):
            # Argmax does not read any of the sampling parameter tensors.
            self.all_greedy = self.any_greedy = True
            self.use_top_k = self.use_top_p = self.use_min_p = False
            self.max_top_k = 0
            return
        top_ps = [1.0 if s.top_p is None else s.top_p for s in seqs]
        top_ks = [vocab if (s.top_k is None or s.top_k <= 0) else min(s.top_k, vocab) for s in seqs]
        min_ps = [0.0 if s.min_p is None else s.min_p for s in seqs]

        self._cp(self.temperatures[:bs], temps, torch.float32)
        self._cp(self.top_ps[:bs], top_ps, torch.float32)
        self._cp(self.top_ks[:bs], top_ks, torch.int32)
        self._cp(self.min_ps[:bs], min_ps, torch.float32)

        self.all_greedy = all(t <= 1e-5 for t in temps)
        self.any_greedy = any(t <= 1e-5 for t in temps)
        self.use_top_k = any(k < vocab for k in top_ks)
        self.use_top_p = any(p < 1.0 for p in top_ps)
        self.use_min_p = any(m > 0.0 for m in min_ps)
        self.max_top_k = max(top_ks) if self.use_top_k else 0
