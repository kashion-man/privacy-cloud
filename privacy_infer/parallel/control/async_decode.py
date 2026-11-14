from dataclasses import dataclass, replace

import torch

from ..data import ScheduledBatch


@dataclass
class DecodeTicket:
    sequences: list
    sampled_gpu: torch.Tensor
    sampled_cpu: torch.Tensor
    ready: torch.cuda.Event


class CoreAsyncDecode:
    def __init__(self, engine):
        self.engine = engine
        self.pending: DecodeTicket | None = None
        self._slot = 0
        self._cpu = [
            torch.empty(engine.config.max_num_seqs, dtype=torch.int64, pin_memory=True)
            for _ in range(2)
        ]
        self.stats = dict(submitted=0, lookahead_submitted=0, drains=0, eos_fences=0)

    def run_decode(self, seqs, input_ids_gpu):
        wire = ScheduledBatch(decodes=seqs).to_wire()
        return self.engine.executor.worker.model_runner._run_decode(wire.decodes, input_ids_gpu)

    def advance(self, sequences, tokens):
        return self.engine.scheduler.update(ScheduledBatch(decodes=sequences), [], tokens)

    def plan_lookahead(self, ticket):
        scheduler, kv = self.engine.scheduler, self.engine.kv
        seqs = ticket.sequences

        if scheduler.waiting or list(scheduler.running) != seqs:
            return None
        if any(s.is_finished or s.num_completion_tokens + 1 >= s.max_tokens for s in seqs):
            return None
        required = [
            max(0, (len(s) + 1 + s.block_size - 1) // s.block_size - len(s.block_table))
            for s in seqs
        ]
        if sum(required) > kv.blocks.available_slots:
            return None
        plans = []
        for seq, pages in zip(seqs, required):
            for _ in range(pages):
                seq.block_table.append(kv.blocks.allocate_fresh())
            plans.append(
                replace(
                    seq,
                    token_ids=[*seq.token_ids, 0],
                    block_table=list(seq.block_table),
                    num_cached_tokens=seq.num_cached_tokens + 1,
                    num_scheduled_tokens=1,
                )
            )
        return plans

    @torch.inference_mode()
    def submit(self, seqs, *, plans=None, input_ids_gpu=None):
        samples = self.run_decode(plans if plans is not None else seqs, input_ids_gpu)

        samples = samples.clone()
        cpu = self._cpu[self._slot][: len(seqs)]
        self._slot ^= 1
        event = torch.cuda.Event()

        cpu.copy_(samples, non_blocking=True)
        event.record()
        self.stats["submitted"] += 1
        return DecodeTicket(list(seqs), samples, cpu, event)

    def consume(self, ticket, ahead=None):
        ticket.ready.synchronize()
        ids = ticket.sampled_cpu.tolist()
        valid = [(s, tid) for s, tid in zip(ticket.sequences, ids) if not s.is_finished]
        will_finish = any(
            (not s.ignore_eos and tid == self.engine.eos_token_id)
            or s.num_completion_tokens + 1 >= s.max_tokens
            for s, tid in valid
        )
        if will_finish and ahead is not None:
            ahead.ready.synchronize()
            self.stats["eos_fences"] += 1
        seqs, tokens = [s for s, _ in valid], [tid for _, tid in valid]
        for seq in seqs:
            seq.num_scheduled_tokens = 1
        return self.advance(seqs, tokens)

    def step(self):
        engine = self.engine
        if self.pending is not None:
            previous = self.pending
            plans = self.plan_lookahead(previous)
            ahead = None
            if plans is not None:
                ahead = self.submit(
                    previous.sequences, plans=plans, input_ids_gpu=previous.sampled_gpu
                )
                self.stats["lookahead_submitted"] += 1
            else:
                self.stats["drains"] += 1
            outputs = self.consume(previous, ahead)
            self.pending = ahead
            if ahead is not None and all(s.is_finished for s in ahead.sequences):
                ahead.ready.synchronize()
                self.pending = None
            return outputs, len(plans) if plans is not None else 0
        if engine.scheduler.waiting or not engine.scheduler.running:
            return engine._step_sync()
        if any(s.temperature > 0 for s in engine.scheduler.running):
            return engine._step_sync()
        sched = engine.scheduler.schedule()
        if sched.prefills or not sched.decodes:
            raise RuntimeError("async decode requires a non-empty decode-only schedule")
        self.pending = self.submit(sched.decodes)
        return [], len(sched.decodes)

    def close(self):
        if self.pending is not None:
            self.pending.ready.synchronize()
            self.pending = None


__all__ = ["CoreAsyncDecode", "DecodeTicket"]
