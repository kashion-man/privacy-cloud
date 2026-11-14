"""Advance stable decode rows using the previous GPU token and resident metadata."""

import triton
import triton.language as tl


@triton.jit
def _advance(
    tokens,
    ids,
    positions,
    slots,
    lengths,
    tables,
    BS: tl.constexpr,
    WIDTH: tl.constexpr,
    PAGE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.arange(0, TILE)
    valid = row < BS
    position = tl.load(positions + row, valid, other=0) + 1
    page = tl.load(tables + row * WIDTH + position // PAGE, valid, other=0)
    token = tl.load(tokens + row, valid, other=0)
    tl.store(ids + row, token, valid)
    tl.store(positions + row, position, valid)
    tl.store(lengths + row, position + 1, valid)
    tl.store(slots + row, page * PAGE + position % PAGE, valid)


def advance_decode(data, sampled_gpu, batch_size, block_size):
    _advance[(1,)](
        sampled_gpu,
        data.input_ids,
        data.positions,
        data.slot_mapping,
        data.cache_seqlens,
        data.block_tables,
        batch_size,
        data.block_tables.shape[1],
        block_size,
        triton.next_power_of_2(batch_size),
        num_warps=4,
    )
