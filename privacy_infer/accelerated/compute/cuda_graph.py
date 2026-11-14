import torch

from .layers.attention import AttnInputs, AttnMetadata, attention_context


class CUDAGraphRunner:
    def __init__(
        self,
        model,
        batch,
        meta: AttnMetadata,
        *,
        max_batch_size: int,
        vocab_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.model = model
        self.batch = batch
        self._meta = meta
        self.device = device
        self.max_bs = max_batch_size
        self.bs_list = self._make_bs_list(max_batch_size)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._graph_pool = None
        self.logits = torch.empty(max_batch_size, vocab_size, dtype=dtype, device=device)
        self.samples = torch.empty(max_batch_size, dtype=torch.int64, device=device)

    def _forward(self, bs: int, attn: AttnInputs) -> None:
        hidden = self.model(self.batch.input_ids[:bs], self.batch.positions[:bs], attn)
        self.model.compute_logits(hidden, out=self.logits[:bs])
        torch.argmax(self.logits[:bs], dim=-1, out=self.samples[:bs])

    @torch.inference_mode()
    def capture(self) -> None:
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        self.batch.slot_mapping.fill_(-1)
        for bs in sorted(self.bs_list, reverse=True):
            attn = AttnInputs(
                slot_mapping=self.batch.slot_mapping[:bs],
                is_prefill=False,
                cache_seqlens=self.batch.cache_seqlens[:bs],
                block_tables=self.batch.block_tables[:bs],
            )
            with attention_context(self._meta):
                self._forward(bs, attn)

            graph = torch.cuda.CUDAGraph()
            with attention_context(self._meta):
                with torch.cuda.graph(graph, pool=self._graph_pool):
                    self._forward(bs, attn)
            if self._graph_pool is None:
                self._graph_pool = graph.pool()
            self.graphs[bs] = graph
        self.batch.slot_mapping.fill_(-1)
        torch.cuda.synchronize(self.device)

    def can_use(self, bs: int) -> bool:
        return bs <= self.max_bs

    def replay(self, bs: int) -> tuple[torch.Tensor, torch.Tensor]:
        pbs = self._pad_bs(bs)
        self.graphs[pbs].replay()
        return self.logits[:bs], self.samples[:bs]

    def _pad_bs(self, bs: int) -> int:
        for b in self.bs_list:
            if b >= bs:
                return b
        return bs

    @staticmethod
    def _make_bs_list(max_bs: int) -> list[int]:
        bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        bs = [b for b in bs if b <= max_bs]
        if max_bs not in bs:
            bs.append(max_bs)
        return bs

    def destroy(self) -> None:
        self.graphs.clear()
        self._graph_pool = None
        del self.logits, self.samples
