from __future__ import annotations

from dataclasses import dataclass, field

import msgspec

from .sequence import Sequence


@dataclass
class ScheduledBatch:
    prefills: list[Sequence] = field(default_factory=list)
    decodes: list[Sequence] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.prefills and not self.decodes

    def to_wire(self) -> "SchedulerOutput":
        return SchedulerOutput(
            prefills=[_prefill_schedule(s) for s in self.prefills],
            decodes=[_decode_schedule(s) for s in self.decodes],
        )


def _sampling_fields(s: Sequence) -> tuple[float, float | None, int | None, float | None]:
    return s.temperature, s.top_p, s.top_k, s.min_p


def _decode_schedule(s: Sequence) -> "SeqSchedule":
    assert len(s.token_ids) == s.num_cached_tokens + 1, (
        f"decode invariant: len(token_ids)={len(s.token_ids)} vs "
        f"num_cached_tokens+1={s.num_cached_tokens + 1}"
    )
    temperature, top_p, top_k, min_p = _sampling_fields(s)
    return SeqSchedule(
        seq_id=s.seq_id,
        block_table=s.block_table,
        num_cached_tokens=s.num_cached_tokens,
        token_ids=[s.last_token],
        need_sample=False,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )


def _prefill_schedule(s: Sequence) -> "SeqSchedule":
    start = s.num_cached_tokens
    end = start + s.num_scheduled_tokens
    temperature, top_p, top_k, min_p = _sampling_fields(s)
    return SeqSchedule(
        seq_id=s.seq_id,
        block_table=s.block_table,
        num_cached_tokens=s.num_cached_tokens,
        token_ids=s.token_ids[start:end],
        need_sample=s.need_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )


class SeqSchedule(msgspec.Struct):
    seq_id: int
    block_table: list[int]
    num_cached_tokens: int
    token_ids: list[int]
    need_sample: bool = False
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None


class SchedulerOutput(msgspec.Struct):
    prefills: list[SeqSchedule] = msgspec.field(default_factory=list)
    decodes: list[SeqSchedule] = msgspec.field(default_factory=list)


class EngineCoreOutput(msgspec.Struct):
    seq_id: int
    new_token_ids: list[int] = msgspec.field(default_factory=list)
    is_finished: bool = False
