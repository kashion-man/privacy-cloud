"""Qwen3 tokenizer preparation and one isolated LoRA client/evaluation task."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import time
import unicodedata


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".pending")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    tmp.replace(path)


def prepare(root, model, config):
    from transformers import AutoConfig, AutoTokenizer
    from peft import LoraConfig

    model_config = AutoConfig.from_pretrained(model, local_files_only=True)
    if (model_config.model_type, model_config.num_hidden_layers, model_config.hidden_size) != (
            "qwen3", 28, 2048):
        raise ValueError("Expected the local Qwen3-1.7B checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if tokenizer.convert_ids_to_tokens(eos) != "<|im_end|>":
        raise ValueError("Missing Qwen3 chat stop token")
    source = root/"data"
    manifest = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["sha256"].items():
        if sha(source/name) != digest:
            raise ValueError(f"Packaged data digest mismatch: {name}")
    packed = {}
    summary = {}
    for split, count in (("train",10000),("validation",1000)):
        rows = read_rows(source/(split+".jsonl"))
        if len(rows) != count or len({r["id"] for r in rows}) != count:
            raise ValueError(f"Invalid {split} count")
        result = {}
        for row in rows:
            msgs = row.pop("messages")
            if [m["role"] for m in msgs] != ["system","user","assistant"]:
                raise ValueError(f"Unexpected chat roles: {row['id']}")
            prompt = tokenizer.apply_chat_template(msgs[:2], tokenize=False,
                        add_generation_prompt=True, enable_thinking=False)
            full = tokenizer.apply_chat_template(msgs, tokenize=False,
                        add_generation_prompt=False, enable_thinking=False)
            if not full.startswith(prompt + msgs[-1]["content"] + "<|im_end|>"):
                raise ValueError(f"Assistant boundary mismatch: {row['id']}")
            prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
            ids = tokenizer.encode(full, add_special_tokens=False)
            if (ids[:len(prefix_ids)] != prefix_ids or len(ids) != row.pop("token_count")
                    or len(ids)>config["max_length"] or eos not in ids[len(prefix_ids):]):
                raise ValueError(f"Token boundary or length mismatch: {row['id']}")
            stop = ids.index(eos,len(prefix_ids))+1
            row.update(tokens=ids[:stop], label_start=len(prefix_ids))
            if split=="validation" and json.loads(msgs[-1]["content"]) != row["target"]:
                raise ValueError(f"Validation target mismatch: {row['id']}")
            result[row["id"]] = row
        packed[split] = result
        summary[split] = {"rows":len(result),"max_tokens":max(len(r["tokens"]) for r in result.values()),
                          "min_target_tokens":min(len(r["tokens"])-r["label_start"] for r in result.values()),
                          "max_target_tokens":max(len(r["tokens"])-r["label_start"] for r in result.values())}
    clients, owners = {}, {}
    all_ids = Counter()
    for index in range(1,11):
        client = f"C{index:02}"
        ids = [r["id"] for r in read_rows(source/"clients"/(client+".jsonl"))]
        rows = [packed["train"][k] for k in ids]
        if len(ids)!=1000 or Counter(r["mode"] for r in rows) != {
                "spans":300,"candidate_necessity":200,"joint":500}:
            raise ValueError(f"Invalid client partition: {client}")
        for row in rows:
            if owners.setdefault(row["group_id"],client) != client:
                raise ValueError("A training group crosses clients")
        all_ids.update(ids)
        clients[client] = ids
    if all_ids != dict.fromkeys(packed["train"],1):
        raise ValueError("Invalid partition coverage")
    if {r["group_id"] for r in packed["validation"].values()} & set(owners):
        raise ValueError("Validation overlaps training groups")
    if summary["validation"]["max_target_tokens"] >= config["max_new_tokens"]:
        raise ValueError("Validation generation cap is shorter than a reference")
    atomic_json(root/"prepared.json",dict(data=packed, clients=clients, eos=eos,
                                          summary=summary, source_digest=sha(source/"manifest.json")))
    LoraConfig(r=config["lora_rank"], lora_alpha=config["lora_alpha"],
               target_modules=config["target_modules"], bias="none",
               task_type="CAUSAL_LM", lora_dropout=0.0).save_pretrained(root/"adapter_config")
    print(json.dumps({"prepared":summary, "clients":len(clients)}, ensure_ascii=False),flush=True)


def sample_client(rows, ids, phase, seed, limit=None):
    rng = random.Random(seed)
    modes = ("spans","candidate_necessity") if phase=="foundation" else ("joint",)
    result = []
    for mode in modes:
        groups = defaultdict(list)
        for rid in ids:
            if rows[rid]["mode"]==mode:
                groups[rows[rid]["group_id"]].append(rid)
        values = list(groups.values())
        quota = sum(map(len,values))//2
        if limit is not None:
            quota = min(quota, limit//len(modes))
        rng.shuffle(values)
        selected = []
        for group in values:
            if len(selected)+len(group)<=quota:
                selected.extend(group)
            if len(selected)==quota:
                break
        if len(selected)!=quota:
            raise ValueError("Cannot sample half of the selected stage without splitting a group")
        result.extend(selected)
    rng.shuffle(result)
    return result


def loss_sum(model, row, device, chunk_size=64):
    import torch
    from torch.nn import functional as F

    ids = torch.tensor(row["tokens"], dtype=torch.long, device=device).unsqueeze(0)
    base = model.get_base_model()
    states = base.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
    start = row["label_start"]
    features = states[0,start-1:-1]
    labels = ids[0,start:]
    parts = []
    for h,y in zip(features.split(chunk_size),labels.split(chunk_size)):
        parts.append(F.cross_entropy(base.lm_head(h).float(),y,reduction="sum"))
    return sum(parts)


def get_adapter(model):
    from peft import get_peft_model_state_dict
    return {k:v.detach().cpu().float().contiguous() for k,v in
            get_peft_model_state_dict(model).items()}


def load_model(path, adapter_dir, incoming, config, training):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file

    if torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Each worker requires exactly one visible BF16-capable GPU")
    torch.cuda.set_per_process_memory_fraction(config["torch_memory_fraction"], 0)
    torch.manual_seed(config["seed"])
    base = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
             use_safetensors=True, torch_dtype=torch.bfloat16, device_map={"":0},
             attn_implementation="sdpa")
    model = get_peft_model(base,LoraConfig.from_pretrained(adapter_dir))
    if incoming is not None:
        state = load_file(incoming,device="cpu")
        expected = get_adapter(model)
        if state.keys()!=expected.keys() or any(state[k].shape!=v.shape for k,v in expected.items()):
            raise ValueError("Global adapter keys/shapes differ from worker")
        set_peft_model_state_dict(model,state)
        if any(not torch.equal(v,state[k]) for k,v in get_adapter(model).items()):
            raise ValueError("Global adapter did not load exactly")
    if training:
        base.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        model.enable_input_require_grads()
        model.train()
    else:
        model.eval()
    return model


def train_client(model, data, ids, config, report_loss=True):
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    if not params or any(p.dtype!=torch.float32 for p in params):
        raise ValueError("Expected trainable FP32 LoRA weights only")
    if not all("lora_" in name for name,p in model.named_parameters() if p.requires_grad):
        raise ValueError("Base weights must remain frozen")
    optimizer = torch.optim.AdamW(params,lr=config["learning_rate"],weight_decay=0.0)
    device = next(model.parameters()).device
    batch = config["batch_size"]
    tokens,loss_total,steps = 0,0.0,0
    for start in range(0,len(ids),batch):
        rows = [data[rid] for rid in ids[start:start+batch]]
        denominator = sum(len(r["tokens"])-r["label_start"] for r in rows)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((),device=device)
        for row in rows:
            term = loss_sum(model,row,device)
            (term/denominator).backward()
            loss += term.detach()
        torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        optimizer.step()
        tokens += denominator
        loss_total += loss.item()
        steps += 1
        if report_loss and (steps%10==0 or start+batch>=len(ids)):
            print(json.dumps({"step":steps,"examples":min(start+batch,len(ids)),
                              "loss":loss_total/tokens},allow_nan=False),flush=True)
    return dict(examples=len(ids),supervised_tokens=tokens,steps=steps,loss=loss_total/tokens,
                mode_counts=dict(Counter(data[k]["mode"] for k in ids)))


def parse_prediction(text,prompt):
    try:
        result = json.loads(text.strip())
        facts = result["private_facts"]
        task = result["task"]
        if not isinstance(facts,list) or not isinstance(task,str):
            return None
        for f in facts:
            a,b = f["start"],f["end"]
            if (type(a) is not int or type(b) is not int or not 0<=a<b<=len(prompt)
                    or f["span"]!=prompt[a:b] or f["subject"] not in ("self","third_party")
                    or f["necessity"] not in ("required","optional","irrelevant")
                    or not isinstance(f["type"],str)):
                return None
        return result
    except (ValueError,KeyError,TypeError,IndexError):
        return None


def score_prediction(row, text):
    truth = row["target"]
    pred = parse_prediction(text,row["prompt"])
    normalize = lambda x: " ".join(unicodedata.normalize("NFKC",x).split()).casefold()
    gold = truth["private_facts"]
    facts = pred["private_facts"] if pred else []
    span = lambda f: (f["start"],f["end"])
    fact = lambda f: (*span(f),f["type"],f["subject"])
    predicted = {fact(f):f["necessity"] for f in facts}
    expected = {fact(f):f["necessity"] for f in gold}
    counts = {}
    for name,key in (("span",span),("fact",fact)):
        a,b = {key(f) for f in facts},{key(f) for f in gold}
        counts[name] = [len(a&b),len(a-b),len(b-a)]
    confusion = Counter(expected[k]+":"+predicted[k] for k in expected.keys() & predicted.keys())
    return dict(counts=counts, confusion=dict(confusion), valid=pred is not None,
                task_match=pred is not None and normalize(pred["task"])==normalize(truth["task"]),
                fact_labels_match=pred is not None and expected==predicted,
                group=row["group_id"])


def validate(model, data, ids, eos, config):
    import torch
    from transformers import GenerationConfig
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"],local_files_only=True)
    results, total, ntok = [],0.0,0
    batch = config["validation_batch"]
    generation = GenerationConfig(do_sample=False,max_new_tokens=config["max_new_tokens"],
                                  pad_token_id=eos,eos_token_id=eos,use_cache=True)
    with torch.inference_mode():
        for start in range(0,len(ids),batch):
            rows = [data[rid] for rid in ids[start:start+batch]]
            seqs = [r["tokens"][:r["label_start"]] for r in rows]
            width = max(map(len,seqs))
            padded = [[eos]*(width-len(s))+s for s in seqs]
            mask = [[0]*(width-len(s))+[1]*len(s) for s in seqs]
            outputs = model.generate(input_ids=torch.tensor(padded,device="cuda"),
                        attention_mask=torch.tensor(mask,device="cuda"),
                        generation_config=generation)
            decoded = tokenizer.batch_decode(outputs[:,width:],skip_special_tokens=True)
            for row,text in zip(rows,decoded):
                results.append(score_prediction(row,text))
                total += loss_sum(model,row,"cuda:0").item()
                ntok += len(row["tokens"])-row["label_start"]
    if not torch.isfinite(torch.tensor(total)):
        raise ValueError("Nonfinite validation loss")
    return dict(examples=len(ids), loss_sum=total, supervised_tokens=ntok, scores=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    import torch
    from safetensors.torch import save_file

    torch.set_num_threads(2)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    config = spec["config"]
    directory = args.spec.parent
    prepared = json.loads((args.root/"prepared.json").read_text(encoding="utf-8"))
    role = spec["role"]
    privacy = config.get("privacy", {})
    private = role=="train" and (privacy.get("dp") or privacy.get("mpc"))
    incoming = spec.get("incoming")
    if incoming and sha(Path(incoming))!=spec["incoming_sha256"]:
        raise ValueError("Incoming adapter hash mismatch")
    started = time.monotonic()
    model = load_model(config["model_path"],args.root/"adapter_config",incoming,config,role=="train")
    if role=="train":
        ids = sample_client(prepared["data"]["train"],prepared["clients"][spec["client"]],
                            spec["phase"],spec["seed"],spec.get("limit"))
        metrics = train_client(model,prepared["data"]["train"],ids,config,report_loss=not private)
        if private:
            metrics.pop("loss")
    elif role=="validation":
        metrics = validate(model,prepared["data"]["validation"],spec["ids"],prepared["eos"],config)
    elif role=="initial":
        metrics = {}
    else:
        raise ValueError("Unknown worker role")
    if role in ("train","initial"):
        state = get_adapter(model)
        if not state or not all(torch.isfinite(v).all() for v in state.values()):
            raise ValueError("Missing or nonfinite adapter")
        if private:
            from privacy import emit_update
            metrics.update(emit_update(state,incoming,directory,spec["client"],config))
        else:
            path = directory/"adapter.safetensors"
            save_file(state,str(path)+".pending")
            os.replace(str(path)+".pending",path)
            metrics["adapter_sha256"] = sha(path)
    metrics.update(role=role,elapsed=time.monotonic()-started,
                   peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20,2),
                   peak_reserved_mib=round(torch.cuda.max_memory_reserved()/2**20,2))
    atomic_json(directory/"done.json",dict(spec_sha256=sha(args.spec),metrics=metrics))
    print(json.dumps({k:v for k,v in metrics.items() if k!="scores"},allow_nan=False),flush=True)


if __name__=="__main__":
    main()
