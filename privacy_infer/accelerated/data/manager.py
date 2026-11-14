import weakref

import torch
from transformers import PretrainedConfig

from .block_manager import BlockManager
from .prefix import PrefixBackend, RadixPrefixIndex, make_prefix_index
from .sequence import Sequence


class KVManager:
    def __init__(
        self,
        hf_config: PretrainedConfig,
        *,
        num_blocks: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
        prefix_backend: PrefixBackend = "hash",
    ) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")

        self.block_size = block_size
        self.device = device
        self.dtype = dtype
        self.hf_config = hf_config
        self.prefix_backend: PrefixBackend = prefix_backend
        self.num_blocks_total = num_blocks
        self.blocks = BlockManager(num_blocks, block_size)
        self.index = make_prefix_index(prefix_backend)

        # Pool callbacks must not retain the manager and its GPU allocation.
        owner = weakref.ref(self)

        def invalidate(bid):
            manager = owner()
            if manager is not None:
                manager.index.invalidate(bid, manager.blocks)

        def reclaim(n):
            manager = owner()
            return [] if manager is None else manager.index.reclaim_pages(n, manager.blocks)

        self.blocks.on_evict = invalidate
        if isinstance(self.index, RadixPrefixIndex):
            self.blocks.reclaim_fn = reclaim
        self.kv_cache = self._alloc_pool(hf_config, num_blocks, block_size, device, dtype)
        self.last_num_cached_tokens = 0

    @staticmethod
    def _alloc_pool(
        hf: PretrainedConfig,
        num_blocks: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        nkv = hf.num_key_value_heads
        hd = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        layers = hf.num_hidden_layers
        block_bytes = 2 * layers * block_size * nkv * hd * dtype.itemsize
        kv = torch.empty(2, layers, num_blocks, block_size, nkv, hd, dtype=dtype, device=device)
        print(
            f"KV pool: {num_blocks} blocks × {block_size} tok "
            f"({num_blocks * block_bytes / (1024**3):.2f} GiB)"
        )
        return kv

    @staticmethod
    def block_bytes(hf, block_size, dtype):
        head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        return (
            2
            * hf.num_hidden_layers
            * block_size
            * hf.num_key_value_heads
            * head_dim
            * dtype.itemsize
        )

    @classmethod
    def auto_num_blocks(
        cls,
        hf: PretrainedConfig,
        *,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device,
        gpu_memory_utilization: float,
        reserve_bytes: int = 0,
    ) -> int:
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        block_bytes = cls.block_bytes(hf, block_size, dtype)

        free, total = torch.cuda.mem_get_info(device)
        available = int(total * gpu_memory_utilization) - (total - free)

        available -= reserve_bytes
        return max(4, available // block_bytes)

    @property
    def num_free(self) -> int:
        return self.blocks.num_free

    def _allocation(self, seq: Sequence, token_budget: int | None):
        hit = self.index.match(seq, self.blocks)
        cached = min(len(hit) * self.block_size, len(seq) - 1)
        end = len(seq) if token_budget is None else min(len(seq), cached + token_budget)
        pages = (end + self.block_size - 1) // self.block_size
        hit = hit[:pages]
        pinned = sum(bid in self.blocks.cached_ids for bid in hit)
        fits = pages - len(hit) <= self.blocks.available_slots - pinned
        return hit, cached, end, fits

    def allocate(self, seq: Sequence, token_budget: int | None = None) -> bool:
        hit, cached, end, fits = self._allocation(seq, token_budget)
        if not fits:
            return False
        for bid in hit:
            self.blocks.acquire_hit(bid)
        seq.block_table[:] = hit
        seq.num_cached_tokens = cached
        self.reserve([seq], [end])
        self.last_num_cached_tokens = len(hit) * self.block_size
        if isinstance(self.index, RadixPrefixIndex) and hit:
            self.index.pin_match(seq, len(hit))
        return True

    def reserve(self, seqs: list[Sequence], ends: list[int]) -> bool:
        required = [
            max(0, (end + self.block_size - 1) // self.block_size - len(seq.block_table))
            for seq, end in zip(seqs, ends)
        ]
        if sum(required) > self.blocks.available_slots:
            return False
        for seq, count in zip(seqs, required):
            seq.block_table.extend(self.blocks.allocate_fresh() for _ in range(count))
        return True

    def sync_prefix(self, seq: Sequence) -> None:
        if self.prefix_backend == "none" or seq.num_scheduled_tokens <= 0:
            return
        prev = seq.num_cached_tokens - seq.num_scheduled_tokens
        start = prev // self.block_size
        end = min(seq.num_full_blocks(), seq.num_cached_tokens // self.block_size)
        if start >= end:
            return
        self.index.publish(seq, end, self.blocks, start=start)

    def deallocate(self, seq: Sequence, *, publish: bool = True) -> None:
        n_full = (
            min(
                seq.num_full_blocks(),
                seq.num_cached_tokens // self.block_size,
                len(seq.block_table),
            )
            if publish
            else 0
        )
        if publish and n_full > 0 and self.prefix_backend != "none":
            self.index.publish(seq, n_full, self.blocks)

        self.index.release_seq(seq.seq_id)

        published = set(seq.block_table[:n_full]) if n_full > 0 else set()
        for bid in list(seq.block_table):
            keep = bid in published and self.prefix_backend != "none"

            if not keep and self.prefix_backend == "hash":
                self.index.invalidate(bid, self.blocks)
            self.blocks.release_to_cached_or_free(bid, keep_cached=keep)
            if not keep and self.prefix_backend == "radix":
                self.index.invalidate(bid, self.blocks)

        seq.block_table.clear()
        seq.num_cached_tokens = 0
