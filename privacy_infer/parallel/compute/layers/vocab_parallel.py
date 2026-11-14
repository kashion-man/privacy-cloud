from __future__ import annotations

import torch
from torch import nn

from ...distributed import (
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)


def _pad_vocab_size(vocab_size: int, tp_size: int, *, multiple: int = 16) -> int:
    per = (vocab_size + tp_size - 1) // tp_size

    per = ((per + multiple - 1) // multiple) * multiple

    return per * tp_size


def _load_weight(
    param: nn.Parameter, loaded_weight: torch.Tensor, *, padded_vocab: int, row_offset: int = 0
) -> None:
    tp = get_tp_group()

    per_partition = padded_vocab // tp.size

    start = per_partition * tp.group_rank

    seg_len = max(0, min(per_partition, loaded_weight.shape[0] - start))

    if seg_len > 0:
        param.data[row_offset : row_offset + seg_len].copy_(loaded_weight[start : start + seg_len])

    if seg_len < per_partition:
        param.data[row_offset + seg_len : row_offset + per_partition].zero_()


class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        tp = get_tp_group()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        padded = _pad_vocab_size(num_embeddings, tp.size)

        self.vocab_start_index, self.vocab_end_index = self._vocab_range(padded // tp.size, tp)

        self.weight = nn.Parameter(torch.zeros(padded // tp.size + 1, embedding_dim))

        setattr(self.weight, "weight_loader", self.weight_loader)

    def _vocab_range(self, per_partition: int, tp) -> tuple[int, int]:
        return (per_partition * tp.group_rank, per_partition * (tp.group_rank + 1))

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int = 0
    ) -> None:
        _load_weight(
            param,
            loaded_weight,
            padded_vocab=_pad_vocab_size(self.num_embeddings, get_tp_group().size),
            row_offset=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x < self.vocab_start_index) | (x >= self.vocab_end_index)

        local_idx = x - self.vocab_start_index + 1

        local_idx.masked_fill_(mask, 0)

        out = torch.nn.functional.embedding(local_idx, self.weight)
        return tensor_model_parallel_all_reduce(out)


class ParallelLMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, *, bias: bool = False):
        super().__init__()
        tp = get_tp_group()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.padded_vocab_size = _pad_vocab_size(vocab_size, tp.size)
        self.output_size_per_partition = self.padded_vocab_size // tp.size

        self.vocab_start_index = self.output_size_per_partition * tp.group_rank
        self.vocab_end_index = self.output_size_per_partition * (tp.group_rank + 1)

        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
        else:
            self.register_parameter("bias", None)

        setattr(self.weight, "weight_loader", self.weight_loader)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int = 0
    ) -> None:
        _load_weight(param, loaded_weight, padded_vocab=self.padded_vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = torch.nn.functional.linear(hidden_states, self.weight, self.bias)
        logits = tensor_model_parallel_all_gather(logits, dim=-1)
        return logits[..., : self.vocab_size]

    def argmax(self, hidden_states: torch.Tensor) -> torch.Tensor:
        tp = get_tp_group()
        per = self.output_size_per_partition

        local_logits = torch.nn.functional.linear(hidden_states, self.weight, self.bias)

        pad_start = max(0, self.vocab_size - self.vocab_start_index)
        if pad_start < per:
            local_logits[..., pad_start:] = float("-inf")

        vals, idx = local_logits.max(dim=-1)

        idx = idx + per * tp.group_rank

        vals = tensor_model_parallel_all_gather(vals, dim=-1)
        idx = tensor_model_parallel_all_gather(idx, dim=-1)

        bs = hidden_states.shape[0]
        vals = vals.reshape(tp.size, bs).t()
        idx = idx.reshape(tp.size, bs).t()

        best = vals.argmax(dim=-1)

        return idx.gather(-1, best.unsqueeze(-1)).squeeze(-1)
