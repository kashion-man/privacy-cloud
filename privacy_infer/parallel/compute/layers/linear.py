from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from ...distributed import (
    divide,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)


def _shard(weight: torch.Tensor, dim: int) -> torch.Tensor:
    tp = get_tp_group()

    size = weight.shape[dim] // tp.size

    offset = size * tp.group_rank

    return weight.narrow(dim, offset, size)


class ColumnParallelLinear(nn.Module):
    def __init__(
        self, in_size: int, out_size: int, *, bias: bool = False, gather_output: bool = True
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.gather_output = gather_output

        self.output_size_per_partition = divide(out_size, get_tp_group().size)

        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, in_size))

        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            setattr(self.bias, "weight_loader", self.weight_loader)
        else:
            self.register_parameter("bias", None)

        setattr(self.weight, "weight_loader", self.weight_loader)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int = 0
    ) -> None:
        param.data.copy_(_shard(loaded_weight, dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.linear(x, self.weight, self.bias)

        if self.gather_output:
            return tensor_model_parallel_all_gather(out, dim=-1)
        return out


class RowParallelLinear(nn.Module):
    def __init__(
        self, in_size: int, out_size: int, *, bias: bool = False, input_is_parallel: bool = False
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.input_is_parallel = input_is_parallel

        self.input_size_per_partition = divide(in_size, get_tp_group().size)
        self.weight = nn.Parameter(torch.empty(out_size, self.input_size_per_partition))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_size))
            setattr(self.bias, "weight_loader", self.weight_loader)
        else:
            self.register_parameter("bias", None)
        setattr(self.weight, "weight_loader", self.weight_loader)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int = 0
    ) -> None:
        if param is self.bias:
            param.data.copy_(loaded_weight)
            return
        param.data.copy_(_shard(loaded_weight, dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            tp = get_tp_group()
            shard = self.input_size_per_partition
            x = x.narrow(-1, tp.group_rank * shard, shard).contiguous()

        out = torch.nn.functional.linear(x, self.weight)

        out = tensor_model_parallel_all_reduce(out)

        if self.bias is not None:
            out = out + self.bias
        return out


class QKVParallelLinear(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        *,
        bias: bool = False,
    ):
        super().__init__()
        tp_size = get_tp_group().size
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_heads_total = total_num_heads
        self.num_kv_heads_total = total_num_kv_heads

        self.num_heads = divide(total_num_heads, tp_size)

        if tp_size > total_num_kv_heads:
            self.num_kv_heads = 1

            self.num_kv_head_replicas = divide(tp_size, total_num_kv_heads)
        else:
            self.num_kv_heads = divide(total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        self.q_size = self.num_heads * head_size
        self.kv_size = self.num_kv_heads * head_size

        self.weight = nn.Parameter(torch.empty(self.q_size + 2 * self.kv_size, hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.q_size + 2 * self.kv_size))
            setattr(self.bias, "weight_loader", self.weight_loader)
        else:
            self.register_parameter("bias", None)
        setattr(self.weight, "weight_loader", self.weight_loader)

    def _shard_qkv(self, loaded_weight: torch.Tensor, shard_id: str) -> tuple[int, torch.Tensor]:
        tp_rank = get_tp_group().group_rank
        if shard_id == "q":
            begin = 0
            return begin, loaded_weight[tp_rank * self.q_size : (tp_rank + 1) * self.q_size]
        if shard_id == "k":
            begin = self.q_size
        else:
            begin = self.q_size + self.kv_size

        if self.num_kv_head_replicas > 1:
            head_idx = tp_rank // self.num_kv_head_replicas
            start = head_idx * self.kv_size
        else:
            start = tp_rank * self.kv_size
        return begin, loaded_weight[start : start + self.kv_size]

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: str
    ) -> None:
        assert shard_id in ("q", "k", "v", "bias")
        if shard_id == "bias":
            self._load_bias(loaded_weight)
            return
        begin, shard = self._shard_qkv(loaded_weight, shard_id)
        param.data[begin : begin + shard.shape[0]].copy_(shard)

    def _load_bias(self, loaded_weight: torch.Tensor) -> None:
        q_total = self.num_heads_total * self.head_size
        kv_total = self.num_kv_heads_total * self.head_size

        _, q_shard = self._shard_qkv(loaded_weight[:q_total], "q")
        self.bias.data[: self.q_size].copy_(q_shard)
        k_seg = loaded_weight[q_total : q_total + kv_total]
        v_seg = loaded_weight[q_total + kv_total : q_total + 2 * kv_total]
        for idx, (seg, shard_id) in enumerate(((k_seg, "k"), (v_seg, "v"))):
            _, shard = self._shard_qkv(seg, shard_id)
            lo = self.q_size + idx * self.kv_size
            self.bias.data[lo : lo + self.kv_size].copy_(shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(nn.Module):
    def __init__(self, in_size: int, output_sizes: list[int], *, bias: bool = False):
        super().__init__()
        self.in_size = in_size
        self.output_sizes = output_sizes
        self.output_sizes_per_partition = [divide(s, get_tp_group().size) for s in output_sizes]
        total = sum(self.output_sizes_per_partition)
        self.weight = nn.Parameter(torch.empty(total, in_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(total))
            setattr(self.bias, "weight_loader", self.weight_loader)
        else:
            self.register_parameter("bias", None)
        setattr(self.weight, "weight_loader", self.weight_loader)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int
    ) -> None:
        shard_offset = sum(self.output_sizes_per_partition[:shard_id])
        shard_size = self.output_sizes_per_partition[shard_id]
        param.data[shard_offset : shard_offset + shard_size].copy_(_shard(loaded_weight, dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


def _default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    param.data.copy_(loaded_weight)


def get_weight_loader(param: nn.Parameter) -> Callable:
    return getattr(param, "weight_loader", _default_weight_loader)
