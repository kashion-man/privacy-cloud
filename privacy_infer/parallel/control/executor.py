from __future__ import annotations

import multiprocessing as mp
from abc import ABC, abstractmethod

import msgspec
import torch
import torch.distributed as dist

from ..compute.worker import Worker
from ..config import EngineConfig
from ..data import KVManager, ScheduledBatch, SchedulerOutput
from ..distributed import ControlChannel, get_world_group


def _sampler_rank(tp_size: int, pp_size: int) -> int:
    return (pp_size - 1) * tp_size


def _terminate_procs(procs: list[mp.Process]) -> None:
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(timeout=3)
    for p in procs:
        if p.is_alive():
            p.kill()
    for p in procs:
        p.join(timeout=5)
    procs.clear()


class ControlCommand(msgspec.Struct):
    op: str
    sched: SchedulerOutput | None = None
    prefix_backend: str | None = None


class Executor(ABC):
    def __init__(self, config: EngineConfig, device: torch.device, dtype: torch.dtype) -> None:
        self.config = config
        self.device = device
        self.dtype = dtype
        self.sampler_rank = _sampler_rank(config.tp_size, config.pp_size)
        self.kv: KVManager | None = None

    def _kv_manager(self, num_blocks: int) -> KVManager:
        return KVManager(
            num_blocks=num_blocks,
            block_size=self.config.block_size,
            prefix_backend=self.config.prefix_backend,
        )

    @abstractmethod
    def init_kv(self) -> KVManager: ...

    @abstractmethod
    def reset_kv(self, prefix_backend: str | None = None) -> KVManager: ...

    @abstractmethod
    def run(self, sched: ScheduledBatch) -> tuple[int, int]: ...

    @abstractmethod
    def read_samples(self, n_p: int, n_d: int) -> tuple[list[int], list[int]]: ...

    @abstractmethod
    def destroy(self) -> None: ...

    def collective_rpc(self, op: str, **payload) -> object:
        raise ValueError(f"unknown control op: {op}")

    def shutdown(self) -> None:
        return None


def _spawn_worker(config: EngineConfig, rank: int, tp_rank: int, pp_rank: int) -> None:
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    worker = Worker(config, rank, tp_rank, pp_rank)
    try:
        worker.init_environment()
        worker.initialize()
        dist.barrier()
        worker_loop(worker)
    finally:
        worker.destroy_environment()


def worker_loop(worker: Worker) -> None:
    channel = ControlChannel(get_world_group().cpu_group)
    enc = msgspec.msgpack.Encoder()
    dec = msgspec.msgpack.Decoder(ControlCommand)
    sampler_rank = _sampler_rank(worker.tp_size, worker.pp_size)

    while True:
        cmd = dec.decode(channel.broadcast(None, src=0))

        if cmd.op == "shutdown":
            break

        if cmd.op == "reset_kv":
            worker.reset_kv(cmd.prefix_backend)
            continue

        if cmd.op == "clear_batch":
            worker.clear_batch()
            dist.barrier(group=get_world_group().cpu_group)
            continue

        if cmd.op != "execute":
            raise ValueError(f"unknown control op: {cmd.op}")

        counts = worker.execute_model(cmd.sched)

        if worker.rank != sampler_rank:
            continue

        n_p, n_d = counts
        channel.send(enc.encode([n_p, n_d]), dst=0)
        p_ids, d_ids = worker.read_samples(n_p, n_d)
        channel.send(enc.encode([p_ids, d_ids]), dst=0)


class UniprocExecutor(Executor):
    def __init__(self, config: EngineConfig, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__(config, device, dtype)
        self.worker = Worker(config, rank=0, tp_rank=0, pp_rank=0)
        self.worker.init_environment()

        self._ready = False

    def init_kv(self) -> KVManager:
        if not self._ready:
            self.worker.initialize()
            self._ready = True
        self.kv = self._kv_manager(self.worker.get_kv_cache_size())
        return self.kv

    def reset_kv(self, prefix_backend: str | None = None) -> KVManager:
        if prefix_backend is not None:
            self.config.prefix_backend = prefix_backend
        self.worker.reset_kv(prefix_backend)
        self.kv = self._kv_manager(self.worker.get_kv_cache_size())
        return self.kv

    def run(self, sched: ScheduledBatch) -> tuple[int, int]:
        counts = self.worker.execute_model(sched.to_wire())
        assert counts is not None
        return counts

    def read_samples(self, n_p: int, n_d: int) -> tuple[list[int], list[int]]:
        return self.worker.read_samples(n_p, n_d)

    def collective_rpc(self, op: str, **payload) -> object:
        if op == "clear_batch":
            self.worker.clear_batch()
            return None
        if op == "reset_kv":
            self.worker.reset_kv(payload.get("prefix_backend"))
            return None
        return super().collective_rpc(op, **payload)

    def destroy(self) -> None:
        self.kv = None
        self.worker.destroy_environment()


class MultiprocExecutor(Executor):
    def __init__(
        self,
        config: EngineConfig,
        device: torch.device,
        dtype: torch.dtype,
        worker: Worker | None = None,
    ) -> None:
        super().__init__(config, device, dtype)

        self._enc = msgspec.msgpack.Encoder()
        self._dec_counts = msgspec.msgpack.Decoder(list[int])
        self._dec_sample = msgspec.msgpack.Decoder(list[list[int]])

        self._procs: list[mp.Process] = []

        self._owns_workers = worker is None

        if worker is None:
            worker = self._spawn_world()

        self.worker = worker
        self.num_blocks = worker.get_kv_cache_size()
        self._channel = ControlChannel(get_world_group().cpu_group)

    def _spawn_world(self) -> Worker:
        driver: Worker | None = None
        try:
            ctx = mp.get_context("spawn")
            tp, pp = self.config.tp_size, self.config.pp_size

            for pp_rank in range(pp):
                for tp_rank in range(tp):
                    rank = pp_rank * tp + tp_rank
                    if rank == 0:
                        continue
                    p = ctx.Process(
                        target=_spawn_worker,
                        args=(self.config, rank, tp_rank, pp_rank),
                        daemon=False,
                    )
                    p.start()
                    self._procs.append(p)

            driver = Worker(self.config, rank=0, tp_rank=0, pp_rank=0)
            driver.init_environment()
            driver.initialize()

            dist.barrier()
            return driver

        except BaseException:
            _terminate_procs(self._procs)
            if driver is not None:
                try:
                    driver.destroy_environment()
                except Exception:
                    pass
            raise

    def init_kv(self) -> KVManager:
        self.kv = self._kv_manager(self.num_blocks)
        return self.kv

    def reset_kv(self, prefix_backend: str | None = None) -> KVManager:
        if prefix_backend is not None:
            self.config.prefix_backend = prefix_backend
        self.collective_rpc("reset_kv", prefix_backend=prefix_backend)
        self.worker.reset_kv(prefix_backend)
        self.num_blocks = self.worker.get_kv_cache_size()
        self.kv = self._kv_manager(self.num_blocks)
        return self.kv

    def collective_rpc(self, op: str, **payload) -> object:
        cmd = ControlCommand(
            op=op, sched=payload.get("sched"), prefix_backend=payload.get("prefix_backend")
        )
        self._channel.broadcast(self._enc.encode(cmd), src=0)
        if op == "clear_batch":
            self.worker.clear_batch()
            dist.barrier(group=get_world_group().cpu_group)
        return None

    def run(self, sched: ScheduledBatch) -> tuple[int, int]:
        wire = sched.to_wire()

        self.collective_rpc("execute", sched=wire)

        counts = self.worker.execute_model(wire)

        if self.sampler_rank == 0:
            assert counts is not None
            return counts

        n_p, n_d = self._dec_counts.decode(self._channel.recv(self.sampler_rank))
        return n_p, n_d

    def read_samples(self, n_p: int, n_d: int) -> tuple[list[int], list[int]]:
        if self.sampler_rank == 0:
            return self.worker.read_samples(n_p, n_d)
        p_ids, d_ids = self._dec_sample.decode(self._channel.recv(self.sampler_rank))
        return p_ids, d_ids

    def shutdown(self) -> None:
        self.collective_rpc("shutdown")

    def destroy(self) -> None:
        if self._owns_workers:
            try:
                if all(p.is_alive() for p in self._procs):
                    self.shutdown()
                    for process in self._procs:
                        process.join(timeout=2)
            except Exception:
                pass
            _terminate_procs(self._procs)
            self.worker.destroy_environment()
        self.kv = None


def create_executor(
    config: EngineConfig,
    device: torch.device,
    dtype: torch.dtype,
    *,
    rank0_worker: Worker | None = None,
) -> Executor:
    if config.tp_size == 1 and config.pp_size == 1:
        return UniprocExecutor(config, device, dtype)
    return MultiprocExecutor(config, device, dtype, worker=rank0_worker)
