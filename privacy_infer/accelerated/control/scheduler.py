from collections import deque
from ..data import KVManager, ScheduledBatch, Sequence, SequenceStatus


class Scheduler:
    def __init__(
        self,
        kv: KVManager,
        *,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        eos_token_id: int | frozenset[int],
        mix_prefill_decode: bool = True,
    ) -> None:
        self.kv = kv

        self.max_num_seqs = max_num_seqs

        self.max_num_batched_tokens = max_num_batched_tokens

        self.eos = frozenset([eos_token_id] if isinstance(eos_token_id, int) else eos_token_id)

        self.mix_prefill_decode = mix_prefill_decode

        self.waiting: deque[Sequence] = deque()

        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def schedule(self) -> ScheduledBatch:
        if self.mix_prefill_decode:
            p, d = self._schedule_mixed()
        else:
            p, d = self._schedule_prefill_first()
        return ScheduledBatch(prefills=p, decodes=d)

    def _room(self, prefills: list[Sequence], decodes: list[Sequence]) -> bool:
        return len(prefills) + len(decodes) < self.max_num_seqs

    def _try_alloc(self, seq: Sequence, token_budget: int) -> bool:
        if not seq.block_table:
            return self.kv.allocate(seq, token_budget)
        end = min(len(seq), seq.num_cached_tokens + token_budget)
        return self.kv.reserve([seq], [end])

    def _schedule_decode(
        self, prefills: list[Sequence], decodes: list[Sequence], tokens_used: int
    ) -> int:
        while self.running and self._room(prefills, decodes):
            if tokens_used + 1 > self.max_num_batched_tokens:
                break

            seq = self.running.popleft()

            reserved = self.kv.reserve([seq], [len(seq)])
            while not reserved and self.running:
                self.preempt(self.running.pop())
                reserved = self.kv.reserve([seq], [len(seq)])

            if not reserved:
                self.preempt(seq)
                break

            seq.num_scheduled_tokens = 1
            seq.need_sample = True
            decodes.append(seq)
            tokens_used += 1

        if decodes:
            self.running.extendleft(reversed(decodes))

        return tokens_used

    def _pull_prefill_chunk(
        self,
        prefills: list[Sequence],
        decodes: list[Sequence],
        tokens_used: int,
        *,
        allow_trailing_partial: bool = True,
    ) -> int:
        if not self.waiting or not self._room(prefills, decodes):
            return tokens_used

        for seq in list(self.waiting):
            if not self._room(prefills, decodes):
                break

            remaining = self.max_num_batched_tokens - tokens_used
            if remaining <= 0:
                break

            if not self._try_alloc(seq, remaining):
                break

            need = seq.num_tokens - seq.num_cached_tokens

            if remaining < need and prefills and not allow_trailing_partial:
                break

            chunk = min(need, remaining)
            seq.num_scheduled_tokens = chunk
            prefills.append(seq)
            tokens_used += chunk

            if seq.num_cached_tokens + chunk == seq.num_tokens:
                seq.need_sample = True
                seq.status = SequenceStatus.RUNNING
                self.waiting.remove(seq)
                self.running.append(seq)
            else:
                seq.need_sample = False
                break

        return tokens_used

    def _schedule_prefill_first(self) -> tuple[list[Sequence], list[Sequence]]:
        prefills: list[Sequence] = []
        decodes: list[Sequence] = []

        self._pull_prefill_chunk(prefills, decodes, 0, allow_trailing_partial=False)

        if prefills:
            return prefills, []

        self._schedule_decode(prefills, decodes, 0)
        return [], decodes

    def _schedule_mixed(self) -> tuple[list[Sequence], list[Sequence]]:
        prefills: list[Sequence] = []
        decodes: list[Sequence] = []

        used = self._schedule_decode(prefills, decodes, 0)

        if decodes:
            self._pull_prefill_chunk(prefills, decodes, used)
            return prefills, decodes

        return self._schedule_prefill_first()

    def preempt(self, seq: Sequence) -> None:
        if seq.status == SequenceStatus.FINISHED:
            return

        seq.status = SequenceStatus.WAITING
        self.kv.deallocate(seq, publish=False)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[Sequence]:
        finished: list[Sequence] = []

        for seq, token_id in zip(seqs, token_ids):
            seq.num_cached_tokens += seq.num_scheduled_tokens

            self.kv.sync_prefix(seq)
            seq.num_scheduled_tokens = 0

            if not seq.need_sample:
                continue

            if seq.max_tokens <= 0:
                seq.status = SequenceStatus.FINISHED
                self.kv.deallocate(seq, publish=True)
                finished.append(seq)
                continue

            seq.append_token(token_id)
            hit_eos = (not seq.ignore_eos) and token_id in self.eos
            hit_len = seq.num_completion_tokens >= seq.max_tokens

            if hit_eos or hit_len:
                seq.status = SequenceStatus.FINISHED

                self.kv.deallocate(seq, publish=True)
                finished.append(seq)

        if finished:
            done = {s.seq_id for s in finished}
            self.running = deque(s for s in self.running if s.seq_id not in done)

        return finished
