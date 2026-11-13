"""Optional client-level Gaussian DP and two-party additive-share aggregation."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys

import numpy as np

from worker import atomic_json, sha


SCALE = 1 << 24
MAX_COORD = 1024


def options(config):
    p = config.get("privacy", {})
    flags = p.get("dp", False),p.get("mpc", False)
    if any(type(value) is not bool for value in flags):
        raise ValueError("DP and MPC switches must be booleans")
    return flags


def budget(clip, multiplier, delta, releases, n, dimensions, mpc):
    if (not 0 < clip <= MAX_COORD or not math.isfinite(multiplier) or
            multiplier <= 0 or not 0 < delta < 1 or releases < 1 or n < 2):
        raise ValueError("Invalid DP configuration")
    radius = clip + (math.sqrt(dimensions)/(2*SCALE) if mpc else 0)
    sensitivity = 2*radius/n
    rho = releases/(2*multiplier**2)
    noise_std = multiplier*sensitivity
    if not math.isfinite(rho) or not math.isfinite(noise_std) or noise_std <= 0:
        raise ValueError("Invalid noise scale or privacy budget")
    return dict(adjacency="replace_one_client_fixed_population", ideal_gaussian_only=True,
                releases=releases, delta=delta, rho=rho, sensitivity=sensitivity,
                noise_std=noise_std, epsilon_upper_bound=rho+2*math.sqrt(rho*math.log(1/delta)))


def gaussian(size):
    # Use independent OS random bits, not the deterministic training RNG.
    bits = np.frombuffer(secrets.token_bytes(16*size), dtype="<u8").reshape(2,size)
    u,v = ((bits >> 12).astype(np.float64)+0.5)*2**-52
    return np.sqrt(-2*np.log(u))*np.cos(2*np.pi*v)


def flatten_delta(state, reference):
    if state.keys()!=reference.keys() or any(state[k].shape!=v.shape for k,v in reference.items()):
        raise ValueError("Adapter schema mismatch")
    vec = np.concatenate([
        state[k].detach().cpu().double().numpy().reshape(-1) -
        reference[k].double().numpy().reshape(-1) for k in sorted(reference)])
    if not np.isfinite(vec).all():
        raise ValueError("Nonfinite client delta")
    return vec


def clipped(vector, clip):
    if not 0 < clip <= MAX_COORD:
        raise ValueError("Invalid clipping radius")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm):
        raise ValueError("Nonfinite client delta norm")
    return vector * min(1.0,clip/max(norm,1e-300))


def write_vector(path, vector):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_suffix(".pending")
    with temp.open("wb") as stream:
        np.save(stream,vector,allow_pickle=False)
    temp.replace(path)


def read_vector(path, size, dtype, digest):
    if sha(path)!=digest:
        raise ValueError(f"Vector checksum differs: {path.name}")
    vector = np.load(path,allow_pickle=False)
    if vector.shape!=(size,) or vector.dtype!=np.dtype(dtype):
        raise ValueError("Vector shape or dtype mismatch")
    return vector


def party_root(task_dir, party):
    return task_dir.parent/"shares"/f"P{party}"


def emit_update(state, incoming, task_dir, client, config):
    from safetensors.torch import load_file

    dp,mpc = options(config)
    if not (dp or mpc) or incoming is None or not re.fullmatch(r"C\d{2}",client):
        raise ValueError("Invalid private client task")
    reference = load_file(incoming,device="cpu")
    vector = flatten_delta(state,reference)
    if dp:
        vector = clipped(vector,config["privacy"]["clip_norm"])
    if not mpc:
        path = task_dir/"clipped_delta.npy"
        write_vector(path,vector)
        return {"delta_sha256":sha(path)}
    if np.max(np.abs(vector))>MAX_COORD:
        raise ValueError("Coordinate outside the fixed-point range")
    encoded = np.rint(vector*SCALE).astype("<i8").view("<u8")
    mask = np.frombuffer(secrets.token_bytes(encoded.nbytes),dtype="<u8")
    digests = []
    for i,share in enumerate((mask,encoded-mask)):
        path = party_root(task_dir,i)/(client+".npy")
        write_vector(path,share)
        digests.append(sha(path))
    return {"share_sha256":digests}


def party_sum(inbox, manifest):
    if manifest["dimension"]<1 or not 2<=len(manifest["clients"])<=10:
        raise ValueError("Invalid party manifest")
    total = np.zeros(manifest["dimension"],dtype="<u8")
    for client,digest in manifest["clients"].items():
        if not re.fullmatch(r"C\d{2}",client):
            raise ValueError("Invalid client ID")
        total += read_vector(inbox/(client+".npy"),len(total),"<u8",digest)
    output = inbox/"sum.npy"
    write_vector(output,total)
    print(sha(output),flush=True)


def reconstruct(tasks, results, dimensions):
    outputs = []
    for party in range(2):
        inbox = party_root(tasks[0].parent,party)
        if any(party_root(t.parent,party)!=inbox for t in tasks):
            raise ValueError("Mixed aggregation rounds")
        manifest = inbox/"manifest.json"
        atomic_json(manifest,dict(dimension=dimensions,clients={
            t.parent.name:r["share_sha256"][party] for t,r in zip(tasks,results)}))
        result = subprocess.check_output([sys.executable,str(Path(__file__).resolve()),
                                          "party",str(inbox),str(manifest)],
                                         text=True,stderr=subprocess.PIPE,timeout=180).strip()
        outputs.append(read_vector(inbox/"sum.npy",dimensions,"<u8",result))
    return (outputs[0]+outputs[1]).view("<i8").astype(np.float64)/(SCALE*len(tasks))


def aggregate(tasks, results, output, incoming, config, release_index):
    import torch
    from safetensors.torch import load_file,save_file

    dp,mpc = options(config)
    n = len(tasks)
    if (not (dp or mpc) or not 2<=n<=10 or len(results)!=n or
            len({t.parent.name for t in tasks})!=n or
            len({r["examples"] for r in results})!=1 or results[0]["examples"]<=0):
        raise ValueError("Private aggregation requires distinct, equally weighted client updates")
    reference = load_file(incoming,device="cpu")
    dimensions = sum(v.numel() for v in reference.values())
    accounting = None
    if dp:
        p = config["privacy"]
        accounting = budget(p["clip_norm"],p["noise_multiplier"],p["delta"],
                            release_index,n,dimensions,mpc)
    receipt = output.with_suffix(".privacy.json")
    # The checkpoint is published once. Reuse its sampled noise on retries.
    if output.exists():
        if not receipt.exists() or json.loads(receipt.read_text()) != dict(
                sha256=sha(output),release_index=release_index):
            raise RuntimeError("Ambiguous private release; do not resample noise")
    else:
        if mpc:
            mean = reconstruct(tasks,results,dimensions)
        else:
            mean = np.zeros(dimensions)
            for task,r in zip(tasks,results):
                mean += read_vector(task.parent/"clipped_delta.npy",dimensions,"<f8",r["delta_sha256"])
            mean /= n
        if dp:
            # Central release model: this coordinator is trusted with the sum.
            mean += gaussian(dimensions)*accounting["noise_std"]
        state,offset = {},0
        for k in sorted(reference):
            v = reference[k]
            stop = offset+v.numel()
            state[k] = (v.double()+torch.from_numpy(mean[offset:stop]).reshape(v.shape)).float().contiguous()
            offset = stop
        if not all(torch.isfinite(v).all() for v in state.values()):
            raise ValueError("Nonfinite aggregate")
        pending = output.with_suffix(".safetensors.pending")
        save_file(state,str(pending))
        # If interrupted after writing this receipt but before the atomic
        # publication, the unpublished pending file is replaced on retry.
        atomic_json(receipt,dict(sha256=sha(pending),release_index=release_index))
        pending.replace(output)
    return dict(examples=sum(r["examples"] for r in results),
                supervised_tokens=sum(r["supervised_tokens"] for r in results),train_loss=None,
                mpc=mpc,dp=dp,privacy_accounting=accounting)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=["party"])
    parser.add_argument("inbox",type=Path)
    parser.add_argument("manifest",type=Path)
    args = parser.parse_args()
    party_sum(args.inbox,json.loads(args.manifest.read_text(encoding="utf-8")))


if __name__=="__main__":
    main()
