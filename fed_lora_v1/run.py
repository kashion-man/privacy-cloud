"""Run the isolated WHU216 FedAvg job when enough GPUs are safely available."""

import argparse
from collections import Counter, defaultdict
import ctypes
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

from worker import atomic_json, prepare, sha


def smi(fields, compute=False):
    mode = "compute-apps" if compute else "gpu"
    data = subprocess.check_output(
        ["nvidia-smi",f"--query-{mode}="+",".join(fields),"--format=csv,noheader,nounits"],
        text=True,timeout=8)
    return [[item.strip() for item in line.split(",")] for line in data.splitlines() if line.strip()]


def gpu_state():
    jobs = defaultdict(set)
    for uuid,pid in smi(["gpu_uuid","pid"],compute=True):
        jobs[uuid].add(int(pid))
    return {uuid:dict(index=int(i),free=int(free),util=int(util),pids=jobs[uuid])
            for i,uuid,free,util in smi(["index","uuid","memory.free","utilization.gpu"])}


def terminate(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def child_setup(parent):
    # Only our workers receive a termination signal if the controller exits.
    if ctypes.CDLL(None).prctl(1,signal.SIGTERM)!=0 or os.getppid()!=parent:
        os._exit(125)


def ensure_disk(root):
    if shutil.disk_usage(root).free < 3*2**30:
        raise RuntimeError("Less than 3 GiB free on the run filesystem")


class Dispatcher:
    def __init__(self,root,config):
        self.root,self.config = root,config
        self.streak = {}

    def status(self,state,**details):
        atomic_json(self.root/"status.json",dict(state=state,pid=os.getpid(),
                    updated_unix=time.time(),**details))

    def task(self,directory,role,incoming=None,**details):
        directory.mkdir(parents=True,exist_ok=True)
        spec = dict(config=self.config,role=role,**details)
        if incoming:
            spec.update(incoming=str(incoming),incoming_sha256=sha(incoming))
        path = directory/"spec.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8"))!=spec:
            raise ValueError(f"Existing job configuration differs: {path}")
        atomic_json(path,spec)
        return path

    def completed(self,path):
        done = path.parent/"done.json"
        if not done.exists():
            return None
        result = json.loads(done.read_text(encoding="utf-8"))
        if result["spec_sha256"]!=sha(path):
            raise ValueError(f"Stale worker result: {done}")
        metrics = result["metrics"]
        if "adapter_sha256" in metrics and sha(path.parent/"adapter.safetensors")!=metrics["adapter_sha256"]:
            raise ValueError(f"Invalid completed adapter: {path.parent}")
        return metrics

    def dispatch(self,tasks,stage,require_two=True):
        cfg = self.config
        pending = [p for p in tasks if self.completed(p) is None]
        active = {}
        retries = Counter()
        # Admission has to be re-established for every group of tasks.
        self.streak.clear()
        probe_failures = 0
        minimum = cfg["min_parallel"] if require_two else 1
        admitted = False
        try:
            while pending or active:
                ensure_disk(self.root)
                try:
                    snapshot = gpu_state()
                    if cfg.get("gpu_allowlist"):
                        snapshot = {uuid:state for uuid,state in snapshot.items()
                                    if uuid in cfg["gpu_allowlist"]}
                        if len(snapshot)!=len(cfg["gpu_allowlist"]):
                            raise ValueError("Configured GPUs are absent")
                    probe_failures = 0
                except (OSError,ValueError,subprocess.SubprocessError):
                    probe_failures += 1
                    self.streak.clear()
                    self.status("gpu_probe_unavailable",stage=stage,failures=probe_failures)
                    if probe_failures >= 3:
                        raise RuntimeError("GPU inventory repeatedly unavailable")
                    time.sleep(cfg["poll_seconds"])
                    continue
                for uuid,(path,process) in list(active.items()):
                    state = snapshot.get(uuid)
                    code = process.poll()
                    if code is not None:
                        del active[uuid]
                        if code or self.completed(path) is None:
                            raise RuntimeError(f"Worker failed: {path.parent}, exit={code}")
                    elif state is None or state["pids"]-{process.pid} or state["free"]<cfg["free_mib_reserve"]:
                        terminate(process)
                        del active[uuid]
                        if self.completed(path) is None:
                            retries[path] += 1
                            pending.insert(0,path)
                        self.streak[uuid] = 0
                        print(json.dumps(dict(event="yielded_gpu",stage=stage,gpu=uuid,
                                              task=path.parent.name,retries=retries[path])),flush=True)
                for uuid,state in snapshot.items():
                    idle = (uuid not in active and not state["pids"] and
                            state["util"]<=cfg["idle_utilization_max"] and
                            state["free"]>=cfg["free_mib_to_start"])
                    self.streak[uuid] = self.streak.get(uuid,0)+1 if idle else 0
                eligible = sorted((u for u in snapshot if self.streak[u]>=cfg["idle_probes"]),
                                  key=lambda u:(-snapshot[u]["free"],snapshot[u]["index"]))
                if len(active)+len(eligible)>=minimum:
                    admitted = True
                # Once a multi-GPU wave starts, finish its pending tail on any
                # safe card. Each new wave establishes the initial capacity.
                if admitted:
                    for uuid in eligible:
                        if not pending or len(active)>=cfg["max_workers"]:
                            break
                        path = pending.pop(0)
                        env = dict(os.environ,CUDA_VISIBLE_DEVICES=uuid)
                        with (path.parent/"worker.log").open("a",encoding="utf-8") as log:
                            log.write(f"\nWorker launch on {uuid}; retry {retries[path]}\n")
                            log.flush()
                            parent = os.getpid()
                            process = subprocess.Popen(
                                [sys.executable,"-u",str(Path(__file__).with_name("worker.py")),
                                 "--root",str(self.root),"--spec",str(path)],
                                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                env=env,preexec_fn=lambda:child_setup(parent))
                        active[uuid]=(path,process)
                        self.streak[uuid]=0
                self.status("running" if active else "waiting_for_gpus",stage=stage,
                            pending=len(pending),active=[
                                dict(gpu=snapshot[u]["index"],pid=p.pid,task=path.parent.name)
                                for u,(path,p) in active.items()],
                            gpu=[dict(index=s["index"],free_mib=s["free"],
                                      utilization=s["util"],compute_processes=len(s["pids"]),
                                      idle_probes=self.streak[u]) for u,s in snapshot.items()])
                if pending or active:
                    time.sleep(cfg["poll_seconds"])
        finally:
            for _,process in active.values():
                terminate(process)
        return [self.completed(p) for p in tasks]


def average(tasks,results,output,incoming):
    import torch
    from safetensors.torch import load_file,save_file

    reference = load_file(str(incoming),device="cpu")
    total = sum(r["examples"] for r in results)
    if total<=0:
        raise ValueError("No examples to aggregate")
    combined = {k:torch.zeros_like(v,dtype=torch.float64) for k,v in reference.items()}
    for task,result in zip(tasks,results):
        state = load_file(str(task.parent/"adapter.safetensors"),device="cpu")
        if state.keys()!=reference.keys() or result["examples"]<=0:
            raise ValueError("Client adapter or sample count mismatch")
        for key,value in state.items():
            if value.shape!=reference[key].shape or not torch.isfinite(value).all():
                raise ValueError("Incompatible or nonfinite client adapter")
            combined[key].add_(value.double(),alpha=result["examples"]/total)
    state = {k:v.float().contiguous() for k,v in combined.items()}
    if not any(not torch.equal(v,reference[k]) for k,v in state.items()):
        raise ValueError("Global adapter did not change")
    save_file(state,str(output)+".pending")
    os.replace(str(output)+".pending",output)
    return dict(examples=total,supervised_tokens=sum(r["supervised_tokens"] for r in results),
                train_loss=sum(r["loss"]*r["supervised_tokens"] for r in results)/
                           sum(r["supervised_tokens"] for r in results))


def f1(tp,fp,fn):
    return 2*tp/max(1,2*tp+fp+fn)


def merge_adapter(tasks,results,output,incoming,config,release_index):
    privacy = config.get("privacy", {})
    if privacy.get("dp") or privacy.get("mpc"):
        from privacy import aggregate
        return aggregate(tasks,results,output,incoming,config,release_index)
    return average(tasks,results,output,incoming)


def validation_summary(results):
    scores = [s for r in results for s in r["scores"]]
    if not scores or len(scores)!=sum(r["examples"] for r in results):
        raise ValueError("Unexpected number of validation results")
    counts,confusion,groups = defaultdict(Counter),Counter(),defaultdict(list)
    for row in scores:
        for key,v in row["counts"].items():
            counts[key].update(dict(zip(("tp","fp","fn"),v)))
        confusion.update(row["confusion"])
        groups[row["group"]].append(row["fact_labels_match"])
    pair = [v for v in groups.values() if len(v)==2]
    classes = ("required","optional","irrelevant")
    label_f1 = {}
    for c in classes:
        label_f1[c] = f1(confusion[c+":"+c],
                        sum(confusion[o+":"+c] for o in classes if o!=c),
                        sum(confusion[c+":"+o] for o in classes if o!=c))
    n = len(scores)
    info = dict(examples=n,valid_json_rate=sum(s["valid"] for s in scores)/n,
                task_normalized_exact_match=sum(s["task_match"] for s in scores)/n,
                necessity_macro_f1_on_matched_facts=sum(label_f1.values())/len(classes),
                necessity_matched_facts=sum(confusion.values()),necessity_by_label=label_f1,
                pair_groups=len(pair),pairwise_fact_label_accuracy=(
                    sum(all(v) for v in pair)/len(pair) if pair else None),
                loss=sum(r["loss_sum"] for r in results)/sum(r["supervised_tokens"] for r in results))
    for key,c in counts.items():
        tp,fp,fn = (c[k] for k in ("tp","fp","fn"))
        info[key]=dict(precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                       f1=f1(tp,fp,fn),tp=tp,fp=fp,fn=fn)
    return info


def run(root, model):
    cfg = json.loads((root/"config.json").read_text(encoding="utf-8"))
    cfg["model_path"] = str(model)
    dispatch = Dispatcher(root,cfg)
    ensure_disk(root)
    config_stamp = dict(config=cfg,dataset_sha256=sha(root/"data"/"manifest.json"),
                        model_config_sha256=sha(model/"config.json"),
                        model_files={p.name:[p.stat().st_size,p.stat().st_mtime_ns]
                                     for p in sorted(model.glob("*.safetensors"))})
    manifest = root/"run.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8"))!=config_stamp:
        raise ValueError("Configuration, dataset or model changed; use a new run directory")
    atomic_json(manifest,config_stamp)
    dispatch.status("preparing")
    # Tokenizing again on an explicit resume also detects a changed or broken
    # local tokenizer before any workers load the model.
    prepare(root,model,cfg)
    prepared = json.loads((root/"prepared.json").read_text(encoding="utf-8"))
    clients = sorted(prepared["clients"])
    validation = sorted(prepared["data"]["validation"])
    initial = dispatch.task(root/"initial","initial")
    dispatch.dispatch([initial],"initial_adapter")
    start_adapter = initial.parent/"adapter.safetensors"
    smoke = root/"smoke"
    if not (smoke/"report.json").exists():
        tasks = [dispatch.task(smoke/client,"train",start_adapter,phase="foundation",
                               seed=cfg["seed"]+i,client=client,limit=8)
                 for i,client in enumerate(clients[:2])]
        metrics = dispatch.dispatch(tasks,"two_gpu_smoke")
        if any(r["examples"]!=8 or r["steps"]!=1 for r in metrics):
            raise ValueError("GPU smoke did not perform one local step per client")
        averaged = smoke/"adapter.safetensors"
        stats = merge_adapter(tasks,metrics,averaged,start_adapter,cfg,release_index=1)
        task = dispatch.task(smoke/"validation","validation",averaged,ids=validation[:8])
        ev = validation_summary(dispatch.dispatch([task],"smoke_validation",require_two=False))
        atomic_json(smoke/"report.json",dict(clients=metrics,aggregate=stats,validation=ev))
        print(json.dumps(dict(event="gpu_smoke_pass",client_count=2,validation=ev)),flush=True)
    latest = root/"latest.json"
    first_round = 1
    if latest.exists():
        state = json.loads(latest.read_text(encoding="utf-8"))
        first_round = state["round"]+1
        start_adapter = root/state["adapter"]
        if sha(start_adapter)!=state["sha256"]:
            raise ValueError("Committed checkpoint hash differs")
    for number in range(first_round,cfg["rounds"]+1):
        phase = "foundation" if number<=cfg["foundation_rounds"] else "joint"
        selected = sorted(random.Random(cfg["seed"]+number).sample(clients,cfg["clients_per_round"]))
        folder = root/"rounds"/f"{number:03}"
        tasks = [dispatch.task(folder/"clients"/client,"train",start_adapter,phase=phase,
                               seed=cfg["seed"]+1000*number+i,client=client)
                 for i,client in enumerate(selected)]
        train = dispatch.dispatch(tasks,f"round_{number:02}_clients")
        if any(r["examples"]!=250 or r["steps"]!=32 for r in train):
            raise ValueError("Round did not complete the configured half local epoch")
        adapter = folder/"adapter.safetensors"
        stats = merge_adapter(tasks,train,adapter,start_adapter,cfg,release_index=number+1)
        shards = [dispatch.task(folder/"validation"/f"{i:02}","validation",adapter,
                                ids=validation[i::cfg["validation_shards"]])
                  for i in range(cfg["validation_shards"])]
        ev = validation_summary(dispatch.dispatch(shards,f"round_{number:02}_validation"))
        if ev["examples"]!=1000 or ev["pair_groups"]!=400:
            raise ValueError("Validation did not cover the full split and all pairs")
        metrics = dict(round=number,phase=phase,selected_clients=selected,clients=train,
                       aggregate=stats,validation=ev,adapter_sha256=sha(adapter),
                       adapter_bytes=adapter.stat().st_size,
                       simulated_adapter_transfer_bytes=adapter.stat().st_size*2*len(selected))
        atomic_json(folder/"metrics.json",metrics)
        atomic_json(latest,dict(round=number,adapter=str(adapter.relative_to(root)),sha256=sha(adapter)))
        start_adapter = adapter
        print(json.dumps(dict(event="round_committed",round=number,phase=phase,
                              train_loss=stats["train_loss"],validation=ev)),flush=True)
    dispatch.status("complete",round=cfg["rounds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    args = parser.parse_args()
    os.umask(0o077)
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="2",
                      MKL_NUM_THREADS="2",CUDA_VISIBLE_DEVICES="")
    root = args.root.resolve()
    def cancel(_sig,_frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,cancel)
    with (root/"controller.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run(root,args.model.resolve())
        except BaseException as exc:
            atomic_json(root/"status.json",dict(state="stopped" if isinstance(exc,KeyboardInterrupt)
                        else "failed",pid=os.getpid(),error=type(exc).__name__,
                        reason=str(exc),updated_unix=time.time()))
            raise


if __name__=="__main__":
    main()
