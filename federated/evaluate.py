"""Compare local privacy models on the same held-out counterfactual groups."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from local_lora import Runtime, evaluate, sample_validation, summarize_old
from worker import atomic_json, get_adapter, sha


def load_adapter(runtime, path, expected_schema):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    import torch

    state = load_file(path, device="cpu")
    if state.keys() != expected_schema.keys() or any(state[k].shape != v.shape for k, v in expected_schema.items()):
        raise ValueError("Adapter schema differs from the loaded Qwen3 LoRA")
    set_peft_model_state_dict(runtime.model, state)
    if any(not torch.equal(t, state[k]) for k, t in get_adapter(runtime.model).items()):
        raise ValueError("Checkpoint load mismatch")


def rule_metrics(rows):
    from privacy_infer.agent_policy import mechanical_spans, private_records
    from privacy_infer.detection import merge_spans, normalize

    predicted = matched = gold_count = covered = comparable = 0
    for row in rows:
        original = row["prompt"]
        text = normalize(original)
        spans = merge_spans(mechanical_spans(text) + [s for _, s in private_records(text)], text)
        gold = row["target"]["private_facts"]
        gold_count += len(gold)
        predicted += len(spans)
        positions = {(s.start, s.end) for s in spans}
        for fact in gold:
            # Align source offsets after Unicode normalization on both sides.
            a, b = fact["start"], fact["end"]
            start = len(normalize(original[:a]))
            end = len(normalize(original[:b]))
            if text[start:end] != normalize(fact["span"]) or end <= start:
                continue
            comparable += 1
            matched += (start, end) in positions
            covered += all(any(s.start <= i < s.end for s in spans) for i in range(start, end))
    return dict(predicted_facts=predicted, matched_facts=matched, gold_facts=gold_count,
                comparable_gold=comparable, simulated_unmasked_gold_facts=gold_count-covered)


def summary(counts):
    gold, matched, predicted = (counts[k] for k in ("gold_facts", "matched_facts", "predicted_facts"))
    return dict(fact_f1=2*matched/(gold+predicted) if gold+predicted else None,
                complete_coverage=1-counts["simulated_unmasked_gold_facts"]/gold if gold else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--central", type=Path, required=True)
    parser.add_argument("--federated", type=Path, required=True)
    parser.add_argument("--federated-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--all-test", action="store_true", help="Evaluate every test row")
    args = parser.parse_args()
    expected = json.loads(args.federated_metrics.read_text(encoding="utf-8"))["adapter_sha256"]
    if sha(args.federated) != expected:
        raise ValueError("FedLoRA adapter checksum differs from the training receipt")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch

    torch.manual_seed(20261115)
    test = args.dataset/"test.jsonl"
    all_rows = [json.loads(line) for line in test.read_text(encoding="utf-8").splitlines()]
    if not all_rows or any(row["split"] != "test" or row["mode"] != "joint" for row in all_rows):
        raise ValueError("Expected a held-out joint privacy test split")
    rows = all_rows if args.all_test else sample_validation(all_rows, 73)
    report = dict(device=args.device, test_sha256=sha(test), test_rows=len(all_rows),
                  sampled_rows=len(rows), sample_seed=73,
                  sampled_ids_sha256=hashlib.sha256("\n".join(r["id"] for r in rows).encode()).hexdigest(),
                  central_adapter_sha256=sha(args.central), federated_adapter_sha256=expected, results={})
    rules = rule_metrics(rows)
    report["results"]["rules"] = {"counts": rules, "summary": summary(rules)}
    runtime = Runtime(args.model, args.device)
    initial = get_adapter(runtime.model)
    for name, path in (("base", None), ("central_lora", args.central), ("fed_lora", args.federated)):
        if path is not None:
            load_adapter(runtime, path, initial)
        result = evaluate(runtime, rows)
        result["summary"] = summary(result["counts"])
        report["results"][name] = result
        atomic_json(args.output, report)
        print(json.dumps({"model": name, **result["summary"],
                          "macro_f1": result["oracle_metrics"]["macro_f1"]}), flush=True)
    inputs = [runtime.tokenizer.encode(runtime.tokenizer.apply_chat_template(
        row["messages"][:2], tokenize=False, enable_thinking=False, add_generation_prompt=True),
        add_special_tokens=False) for row in rows]
    report["fed_lora_native_joint_prompt"] = summarize_old(rows, runtime.generate(inputs, 192))
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
