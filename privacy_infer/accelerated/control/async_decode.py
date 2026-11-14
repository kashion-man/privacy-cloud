from dataclasses import dataclass

import torch


@dataclass
class DecodeTicket:
    sequences: list
    sampled_gpu: torch.Tensor
    sampled_cpu: torch.Tensor
    ready: torch.cuda.Event


class AsyncDecode:
    def __init__(self, engine):
        self.engine = engine
        self.pending: DecodeTicket | None = None
        self._slot = 0
        self._cpu = [
            torch.empty(engine.config.max_num_seqs, dtype=torch.int64, pin_memory=True)
            for _ in range(2)
        ]
        self._ready = [torch.cuda.Event() for _ in range(2)]
        self._refresh_tables = False
        self._tables_dirty = False
        self.stats = dict(
            submitted=0,
            lookahead_submitted=0,
            drains=0,
            eos_fences=0,
            waiting_drains=0,
            batch_drains=0,
            terminal_drains=0,
            memory_drains=0,
            prefix_fences=0,
        )

    def run_decode(self, seqs, input_ids_gpu):
        samples = self.engine.executor.run_decode(
            seqs,
            input_ids_gpu,
            refresh_block_tables=input_ids_gpu is not None and self._refresh_tables,
        )
        self._tables_dirty = False
        return samples

    def advance(self, sequences, tokens):
        finished = self.engine.scheduler.postprocess(sequences, tokens)
        return [(s.seq_id, s.token_ids[s.num_prompt_tokens :]) for s in finished]

    def plan_lookahead(self, ticket):
        scheduler, kv = self.engine.scheduler, self.engine.kv
        seqs = ticket.sequences

        if scheduler.waiting:
            self.stats["waiting_drains"] += 1
            return None
        if len(scheduler.running) != len(seqs) or any(
            current is not previous for current, previous in zip(scheduler.running, seqs)
        ):
            self.stats["batch_drains"] += 1
            return None
        if any(s.is_finished or s.num_completion_tokens + 1 >= s.max_tokens for s in seqs):
            self.stats["terminal_drains"] += 1
            return None
        pages_before = sum(len(seq.block_table) for seq in seqs)
        if not kv.reserve(seqs, [len(seq) + 1 for seq in seqs]):
            self.stats["memory_drains"] += 1
            return None
        self._refresh_tables = (
            sum(len(seq.block_table) for seq in seqs) != pages_before or self._tables_dirty
        )
        # The runner applies a one-token metadata offset; CPU history stays intact.
        return seqs

    @torch.inference_mode()
    def submit(self, seqs, *, input_ids_gpu=None):
        # Copy and consume graph outputs on this stream before the next replay overwrites them.
        samples = self.run_decode(seqs, input_ids_gpu)

        cpu = self._cpu[self._slot][: len(seqs)]
        event = self._ready[self._slot]
        self._slot ^= 1

        cpu.copy_(samples, non_blocking=True)
        event.record()
        self.stats["submitted"] += 1
        return DecodeTicket(list(seqs), samples, cpu, event)

    def consume(self, ticket, ahead=None):
        ticket.ready.synchronize()
        ids = ticket.sampled_cpu.tolist()
        valid = [(s, tid) for s, tid in zip(ticket.sequences, ids) if not s.is_finished]
        will_finish = any(
            (not s.ignore_eos and tid in self.engine.scheduler.eos)
            or s.num_completion_tokens + 1 >= s.max_tokens
            for s, tid in valid
        )
        may_remap = self.engine.kv.prefix_backend == "radix" and any(
            (s.num_cached_tokens + 1) // s.block_size > s.num_cached_tokens // s.block_size
            for s, _ in valid
        )
        if ahead is not None and (will_finish or may_remap):
            ahead.ready.synchronize()
            if will_finish:
                self.stats["eos_fences"] += 1
            if may_remap:
                self.stats["prefix_fences"] += 1
        seqs, tokens = [s for s, _ in valid], [tid for _, tid in valid]
        for seq in seqs:
            seq.num_scheduled_tokens = 1
            seq.need_sample = True
        outputs = self.advance(seqs, tokens)
        if may_remap:
            self._tables_dirty = True
        return outputs

    def step(self):
        engine = self.engine
        if self.pending is not None:
            previous = self.pending
            ahead_seqs = self.plan_lookahead(previous)
            ahead = None
            if ahead_seqs is not None:
                ahead = self.submit(ahead_seqs, input_ids_gpu=previous.sampled_gpu)
                self.stats["lookahead_submitted"] += 1
            else:
                self.stats["drains"] += 1
            outputs = self.consume(previous, ahead)
            self.pending = ahead
            if ahead is not None and all(s.is_finished for s in ahead.sequences):
                ahead.ready.synchronize()
                self.pending = None
            return outputs, len(ahead_seqs) if ahead_seqs is not None else 0
        sched = engine.scheduler.schedule()
        if sched.prefills or not sched.decodes or any(s.temperature > 0 for s in sched.decodes):
            return engine.execute_batch(sched)
        self.pending = self.submit(sched.decodes)
        return [], len(sched.decodes)

    def close(self):
        if self.pending is not None:
            self.pending.ready.synchronize()
            self.pending = None


__all__ = ["AsyncDecode", "DecodeTicket"]
