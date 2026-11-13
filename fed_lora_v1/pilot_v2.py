"""Offline checkpoint diagnosis and a small centralized privacy LoRA pilot."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random

from worker import atomic_json, get_adapter, parse_prediction, train_client


def prompt(system, user):
    return [{"role":"system","content":system},{"role":"user","content":user}]


class Runtime:
    def __init__(self, path, device):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        if device == "cuda":
            if torch.cuda.device_count()!=1 or torch.cuda.mem_get_info()[0] < 5500*2**20:
                raise RuntimeError("Pilot requires one GPU with 5500 MiB free")
            torch.cuda.set_per_process_memory_fraction(0.70,0)
        self.device = device
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.eos = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        base = AutoModelForCausalLM.from_pretrained(path,local_files_only=True,
                  use_safetensors=True,torch_dtype=torch.bfloat16,device_map={"":device},
                  attn_implementation="sdpa")
        self.model = get_peft_model(base,LoraConfig(r=8,lora_alpha=16,bias="none",
                         target_modules=["q_proj","v_proj"],task_type="CAUSAL_LM"))
        self.model.eval()

    def tensors(self, rows):
        n=max(map(len,rows))
        ids=[[self.eos]*(n-len(r))+r for r in rows]
        mask=[[0]*(n-len(r))+[1]*len(r) for r in rows]
        return self.torch.tensor(ids,device=self.device),self.torch.tensor(mask,device=self.device)

    def generate(self, prompts, limit):
        for start in range(0,len(prompts),4):
            rows=prompts[start:start+4]
            ids, mask=self.tensors(rows)
            with self.torch.inference_mode():
                out=self.model.generate(input_ids=ids,attention_mask=mask,do_sample=False,
                    max_new_tokens=limit,eos_token_id=self.eos,pad_token_id=self.eos,use_cache=True)
            for seq in out[:,ids.shape[1]:]:
                yield self.tokenizer.decode(seq,skip_special_tokens=True), not bool((seq==self.eos).any())

    def choose(self, messages):
        from privacy_infer.privacy_judge import choose_label

        return choose_label(self.model,self.tokenizer,messages,self.device)


def sample_validation(rows, seed):
    groups=defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    randomizer=random.Random(seed)
    pairs=defaultdict(list)
    singles=defaultdict(list)
    for group in groups.values():
        r=group[0]
        key=(r["domain"],r["metadata"].get("pair_type",""))
        (pairs if len(group)==2 else singles)[key].append(group)
    selected=[]
    for k in sorted(pairs):
        selected.extend(randomizer.choice(pairs[k]))
    for k in sorted(singles):
        selected.extend(randomizer.choice(singles[k]))
    return selected


def summarize_old(rows, results):
    counts=Counter(rows=len(rows))
    for row,(text,capped) in zip(rows,results,strict=True):
        gold=row["target"]["private_facts"]
        expected={f["span"] for f in gold}
        counts["capped"]+=capped
        counts["legacy_strict"]+=parse_prediction(text,row["prompt"]) is not None
        try: obj=json.loads(text.strip())
        except ValueError: obj=None
        counts["json_syntax"]+=obj is not None
        if not isinstance(obj,dict) or not isinstance(obj.get("task"),str) or not isinstance(obj.get("private_facts"),list):
            counts["no_joint_schema"]+=1
            counts["missed_unique_gold_spans"]+=len(expected)
            continue
        counts["joint_schema"]+=1
        found=set()
        for f in obj["private_facts"]:
            if not isinstance(f,dict) or not isinstance(f.get("span"),str) or not f["span"]:
                counts["bad_fact_schema"]+=1
                continue
            s=f["span"]
            if s not in row["prompt"]:
                counts["not_verbatim"]+=1
                continue
            found.add(s)
            counts["groundable_facts"]+=1
            a,b=f.get("start"),f.get("end")
            counts["bad_offsets_on_groundable_fact"]+=not (
                type(a) is int and type(b) is int and 0<=a<b<=len(row["prompt"]) and row["prompt"][a:b]==s)
            matching=[g for g in gold if g["span"]==s]
            if len(matching)==1:
                counts["necessity_comparable"]+=1
                counts["necessity_correct"]+=f.get("necessity")==matching[0]["necessity"]
        counts["matched_unique_gold_spans"]+=len(found & expected)
        counts["missed_unique_gold_spans"]+=len(expected-found)
        counts["extra_grounded_unique_spans"]+=len(found-expected)
    return dict(counts)


def evaluate(runtime, rows):
    from privacy_infer.privacy_judge import (
        EXTRACT, JUDGE, Candidate, encode, evidence_input, ground, judge_input, parse_evidence,
    )

    counts=Counter(rows=len(rows))
    pairs=defaultdict(list)
    confusion=Counter()
    prompts=[encode(runtime.tokenizer,prompt(EXTRACT,evidence_input(r["prompt"])),True) for r in rows]
    for row,(text,capped) in zip(rows,runtime.generate(prompts,128),strict=True):
        sentence=row["prompt"]
        gold={(f["start"],f["end"]):f for f in row["target"]["private_facts"]}
        counts["gold_facts"]+=len(gold)
        counts["extract_capped"]+=capped
        try:
            spans=parse_evidence(text)
            counts["extract_schema_pass"]+=1
        except (ValueError,TypeError):
            spans=[]
        candidates, stats=ground(sentence,spans)
        counts.update(stats)
        predicted={(c.start,c.end):c for c in candidates}
        counts["predicted_facts"]+=len(predicted)
        counts["matched_facts"]+=len(gold.keys()&predicted.keys())
        counts["overlap_gold_facts"]+=sum(any(a<d and c<b for c,d in predicted) for a,b in gold)
        oracle={}
        for n,((a,b),f) in enumerate(sorted(gold.items()),1):
            label=runtime.choose(prompt(JUDGE,judge_input(sentence,Candidate(f"F{n}",a,b,f["span"]))))
            oracle[a,b]=label
            confusion[f["necessity"]+":"+label]+=1
            counts["oracle_correct"]+=label==f["necessity"]
        pipeline={}
        for k,c in predicted.items():
            if k in gold:
                pipeline[k]=runtime.choose(prompt(JUDGE,judge_input(sentence,c)))
                counts["pipeline_correct"]+=pipeline[k]==gold[k]["necessity"]
        counts["oracle_uncertain"]+=sum(x=="uncertain" for x in oracle.values())
        pairs[row["group_id"]].append((all(oracle[k]==f["necessity"] for k,f in gold.items()),
                 predicted.keys()==gold.keys() and all(pipeline[k]==f["necessity"] for k,f in gold.items())))
        # A simulation, not a release decision: how many gold spans are fully
        # covered if all proposed candidates are masked?
        counts["simulated_unmasked_gold_facts"]+=sum(
            not all(any(c<=i<d for c,d in predicted) for i in range(a,b)) for a,b in gold)
    complete_pairs=[p for p in pairs.values() if len(p)==2]
    counts["pairs"]=len(complete_pairs)
    counts["oracle_both_correct"]=sum(all(a for a,_ in p) for p in complete_pairs)
    counts["pipeline_both_correct"]=sum(all(b for _,b in p) for p in complete_pairs)
    by_label={}
    for label in ("required","optional","irrelevant"):
        correct=confusion[label+":"+label]
        gold=sum(n for pair,n in confusion.items() if pair.split(":")[0]==label)
        predicted=sum(n for pair,n in confusion.items() if pair.split(":")[1]==label)
        by_label[label]=dict(correct=correct,gold=gold,predicted=predicted,
                             f1=2*correct/(gold+predicted) if gold+predicted else 0.0)
    return dict(counts=dict(counts),oracle_confusion=dict(confusion),
                oracle_metrics=dict(by_label=by_label,macro_f1=sum(x["f1"] for x in by_label.values())/3))


def train_rows(rows, tokenizer, eos, seed):
    from privacy_infer.privacy_judge import (
        BINARY, EXTRACT, JUDGE, LABELS, Candidate, compact, encode, evidence_input, judge_input,
    )

    rng=random.Random(seed)
    grouped=defaultdict(list)
    for row in rows:
        grouped[row["mode"],row["group_id"]].append(row)
    selected=[]
    for mode,limit in (("spans",120),("candidate_necessity",40),("joint",100)):
        groups=[v for (m,_),v in grouped.items() if m==mode]
        rng.shuffle(groups)
        selected.extend(r for g in groups[:limit] for r in g)
    result={}
    for row in selected:
        source=row["prompt"]
        facts=row["target"].get("private_facts",[])
        tasks=[]
        if row["mode"]!="candidate_necessity":
            tasks.append((EXTRACT,evidence_input(source),compact({"spans":list(dict.fromkeys(
                             f["span"] for f in facts))}),"extract"))
        if row["mode"]=="joint":
            for n,f in enumerate(facts,1):
                c=Candidate(f"F{n}",f["start"],f["end"],f["span"])
                tasks.append((JUDGE,judge_input(source,c),"ABCD"[LABELS.index(f["necessity"])],"judge"))
        elif row["mode"]=="candidate_necessity":
            tasks.append((BINARY,row["messages"][1]["content"],
                          "A" if row["target"]["necessity"]=="required" else "B","binary"))
        for i,(system,user,answer,mode) in enumerate(tasks):
            m=prompt(system,user)
            pre=encode(tokenizer,m,True)
            full=encode(tokenizer,m+[{"role":"assistant","content":answer}])
            if full[:len(pre)]!=pre or eos not in full[len(pre):]:
                raise ValueError("Training target boundary mismatch")
            if len(full)>512:
                continue
            result[f"{row['id']}:{i}"]=dict(tokens=full[:full.index(eos,len(pre))+1],
                                            label_start=len(pre),mode=mode)
    return result


def main():
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file,save_file
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset",type=Path,required=True)
    ap.add_argument("--model",required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--checkpoints",type=Path)
    ap.add_argument("--diagnosis-only",action="store_true")
    ap.add_argument("--validation-ids",type=Path)
    ap.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    args=ap.parse_args()
    if args.device=="cpu" and not args.diagnosis_only:
        ap.error("CPU is supported for checkpoint diagnosis")
    os.environ["HF_HUB_OFFLINE"]="1"
    os.environ["TRANSFORMERS_OFFLINE"]="1"
    os.umask(0o077)
    args.out.mkdir(exist_ok=False,parents=True)
    rows=[json.loads(x) for x in (args.dataset/"validation.jsonl").read_text("utf-8").splitlines()]
    if args.validation_ids:
        by_id={r["id"]:r for r in rows}
        val=[by_id[k] for k in json.loads(args.validation_ids.read_text("utf-8"))]
    else:
        val=sample_validation(rows,31)
    torch.manual_seed(20261113)
    runtime=Runtime(args.model,args.device)
    initial=get_adapter(runtime.model)
    report=dict(validation_rows=len(val),validation_ids=[r["id"] for r in val],held_out_test_used=False)
    if args.diagnosis_only or args.checkpoints:
        inputs=[runtime.tokenizer.encode(runtime.tokenizer.apply_chat_template(
            r["messages"][:2],tokenize=False,enable_thinking=False,add_generation_prompt=True),
            add_special_tokens=False) for r in val]
        report["legacy"]={}
        rounds=("base","011","020") if args.checkpoints else ("base",)
        for round_name in rounds:
            if round_name!="base":
                p=args.checkpoints/round_name
                adapter=p/"adapter.safetensors"
                expected=json.loads((p/"metrics.json").read_text("utf-8"))["adapter_sha256"]
                if hashlib.sha256(adapter.read_bytes()).hexdigest()!=expected:
                    raise ValueError("Downloaded adapter checksum mismatch")
                state=load_file(adapter,device="cpu")
                if state.keys()!=initial.keys():
                    raise ValueError("Unexpected checkpoint keys")
                set_peft_model_state_dict(runtime.model,state)
                if any(not torch.equal(v,state[k]) for k,v in get_adapter(runtime.model).items()):
                    raise ValueError("Checkpoint load mismatch")
            report["legacy"][round_name]=summarize_old(val,runtime.generate(inputs,192))
            atomic_json(args.out/"report.json",report)
            print("LEGACY",round_name,report["legacy"][round_name],flush=True)
        set_peft_model_state_dict(runtime.model,initial)
    if args.diagnosis_only:
        return
    report["v2_before"]=evaluate(runtime,val)
    atomic_json(args.out/"report.json",report)
    print("V2_BEFORE",report["v2_before"]["counts"],flush=True)
    rows=[json.loads(x) for x in (args.dataset/"train.jsonl").read_text("utf-8").splitlines()]
    data=train_rows(rows,runtime.tokenizer,runtime.eos,31)
    ids=list(data)
    random.Random(32).shuffle(ids)
    runtime.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    runtime.model.enable_input_require_grads()
    runtime.model.train()
    report["training"]=train_client(runtime.model,data,ids,
                                  {"learning_rate":1e-4,"batch_size":8})
    runtime.model.eval()
    save_file(get_adapter(runtime.model),args.out/"adapter_model.safetensors")
    runtime.model.peft_config["default"].save_pretrained(args.out)
    atomic_json(args.out/"report.json",report)
    report["v2_after"]=evaluate(runtime,val)
    report["peak_reserved_mib"]=torch.cuda.max_memory_reserved()/2**20
    atomic_json(args.out/"report.json",report)
    print("V2_AFTER",report["v2_after"]["counts"],flush=True)


if __name__=="__main__":
    main()
