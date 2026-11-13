"""Read-only audit of shipped samples, provenance, client maps and template cues."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import hashlib
import json
from pathlib import Path
import re

from build_privacy_fed_v0 import DOMAINS, PAIR_COUNTS, normal, read_jsonl


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(root, mapping):
    return sorted(name for name, expected in mapping.items() if sha(root/name) != expected)


def answer_spans(row):
    return row["target"].get("private_facts", [])


def input_view(row):
    content = row["messages"][1]["content"]
    if row["mode"] == "candidate_necessity":
        return json.loads(content) == {
            "prompt": row["prompt"], "candidate": row["metadata"]["candidate"]}
    return content == row["prompt"]


def with_evidence_removed(row):
    text = row["prompt"]
    for f in sorted(answer_spans(row), key=lambda v: v["start"], reverse=True):
        text = text[:f["start"]] + "[MASK]" + text[f["end"]:]
    return text


PUBLIC = re.compile(r"公开教材|公开练习册|没有填写实际个人资料")
REQUIRED = re.compile(r"计算|算出|估算|倒推|顺推|至少|最多|必须|排除|"
                      r"根据(?:实际|已知|描述)|按(?:实际|已知)|依据|哪一类|应在哪")
OPTIONAL = re.compile(r"通用|常见|通常|介绍|解释|说明|一般|示例|例子")


def cue_guess(row):
    """Diagnostic on evidence-masked text; does not fit a model or use gold metadata."""
    text = with_evidence_removed(row)
    if PUBLIC.search(text):
        return "no_fact"
    gate = re.search(r"(?:用途为验算且|仅在)表单选([甲乙])", text)
    actual = re.search(r"。(?:用途是验算，)?表单选([甲乙])", text)
    if gate and actual:
        return "required" if gate[1] == actual[1] else "irrelevant"
    if REQUIRED.search(text):
        return "required"
    if OPTIONAL.search(text):
        return "optional"
    return "irrelevant"


def cue_diagnostics(rows):
    joint = [row for row in rows if row["mode"] == "joint"]
    primary = lambda r: answer_spans(r)[0]["necessity"] if answer_spans(r) else "no_fact"
    confusion = Counter((primary(r), cue_guess(r)) for r in joint)
    grouping = defaultdict(list)
    for row in joint:
        if row["source"] == "counterfactual":
            grouping[row["group_id"]].append(row)
    public = Counter((bool(answer_spans(r)), bool(PUBLIC.search(r["prompt"]))) for r in joint)
    values = defaultdict(set)
    for row in joint:
        if row["source"] == "counterfactual":
            m = row["metadata"]
            values[(m["family"], m["pair_type"], m["expected"])].add(m["anchor"])
    return {
        "joint_rows": len(joint),
        "no_fact_expected": sum(not answer_spans(r) for r in joint),
        "explicit_public_or_placeholder_cue_true_negative_rows": public[False, True],
        "explicit_public_or_placeholder_cue_positive_rows": public[True, True],
        "evidence_masked_hand_rules_correct": sum(n for (a,b), n in confusion.items() if a == b),
        "evidence_masked_hand_rules_both_pair_members_correct": sum(
            all(primary(r) == cue_guess(r) for r in pair) for pair in grouping.values()),
        "pair_groups": len(grouping),
        "pair_groups_same_request_text": sum(
            bool(request := pair[0]["metadata"].get("request")) and
            request == pair[1]["metadata"].get("request") and
            all(request in row["prompt"] for row in pair)
            for pair in grouping.values()),
        "pair_groups_same_prompt_character_multiset": sum(
            Counter(pair[0]["prompt"]) == Counter(pair[1]["prompt"])
            for pair in grouping.values()),
        "confusion_true_predicted": {f"{a}:{b}": n for (a,b), n in sorted(confusion.items())},
        "scenario_families": len({r["metadata"]["family"] for r in joint}),
        "distinct_task_targets": len({r["target"]["task"] for r in joint}),
        "unique_pair_anchors": len({r["metadata"]["anchor"] for r in joint
                                    if r["source"] == "counterfactual"}),
        "counterfactual_family_intervention_label_cells": len(values),
        "pair_groups_that_intervene_on_fact_value": sum(
            len({r["metadata"]["anchor"] for r in pair}) > 1 for pair in grouping.values()),
    }


def audit(data, raw):
    quality = json.loads((data/"quality_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((data/"source_manifest.json").read_text(encoding="utf-8"))
    failures = []
    def expect(condition, name):
        if not condition:
            failures.append(name)

    expect(not check_hashes(data, quality["sha256"]), "shipped JSONL hashes")
    expect(not check_hashes(raw, manifest["files_sha256"]), "raw provenance hashes")
    is_v1 = quality["version"] == "v1-candidate"
    builder = "repair_privacy_fed_v1.py" if is_v1 else "build_privacy_fed_v0.py"
    scenarios = "privacy_fed_v1_eval.txt" if is_v1 else "privacy_fed_scenarios.json"
    expect(sha(Path(__file__).with_name(builder)) == quality["builder_sha256"], "builder hash")
    expect(sha(Path(__file__).with_name(scenarios)) == quality["scenario_sha256"], "scenario hash")
    if is_v1:
        expect(sha(Path(__file__).with_name("build_privacy_fed_v0.py")) ==
               quality["base_builder_sha256"], "base builder hash")
    splits = {s: read_jsonl(data/(s+".jsonl")) for s in ("train", "validation", "test")}
    upstream_open = {r["uid"]: r for r in read_jsonl(raw/"open_zh_candidates.jsonl")}
    upstream_delegate = read_jsonl(raw/"delegate_train.jsonl")
    excluded = set(json.loads((raw/"delegate_test_fingerprints.json").read_text(encoding="utf-8"))
                   ["normalized_question_sha256"])
    issues, examples, split_stats = Counter(), defaultdict(list), {}
    def note(name, row):
        issues[name] += 1
        if len(examples[name]) < 8:
            examples[name].append({"id":row["id"], "split":row["split"],
                                   "source":row["source"], "group":row["group_id"]})

    for split, rows in splits.items():
        expect(len(rows) == (10000 if split == "train" else 1000), f"{split} count")
        expect(len({r["id"] for r in rows}) == len(rows), f"{split} unique IDs")
        pairs = defaultdict(list)
        delegate_groups = defaultdict(list)
        for row in rows:
            rid, prompt, target, m = row["id"], row["prompt"], row["target"], row["metadata"]
            expect(row["split"] == split, f"{rid}: split")
            expect(0 < row["token_count"] <= 512, f"{rid}: recorded length")
            expect([msg["role"] for msg in row["messages"]] == ["system", "user", "assistant"],
                   f"{rid}: chat roles")
            expect(input_view(row), f"{rid}: input contains unintended fields")
            expect(json.loads(row["messages"][-1]["content"]) == target, f"{rid}: assistant JSON")
            spans = answer_spans(row) + ([m["candidate"]] if row["mode"] == "candidate_necessity" else [])
            for f in spans:
                a,b = f["start"],f["end"]
                expect(0 <= a < b <= len(prompt) and prompt[a:b] == f["span"], f"{rid}: offsets")
            if row["source"] == "openpii":
                src = upstream_open[m["uid"]]
                expect(src["source_text"] == prompt and src["language"] == "zh" and
                       src["region"] == "CN" and src["split"] == "train", f"{rid}: OpenPII source")
                expect({(f["start"],f["end"],f["span"]) for f in spans} ==
                       {(f["start"],f["end"],f["value"]) for f in src["privacy_mask"]},
                       f"{rid}: OpenPII mask")
                if not spans and re.search(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", prompt):
                    note("openpii_empty_mask_contains_email_like_text", row)
            elif row["source"] == "delegateci":
                src = upstream_delegate[m["row_number"]-1]
                label = target["necessity"]
                field = {"required":"essential_spans", "not_required":"non_essential_spans"}[label]
                expect(src["question"] == prompt and src["split"] == "train" and
                       m["candidate"]["span"] in src[field], f"{rid}: DelegateCI source")
                expect(hashlib.sha256(normal(prompt).encode()).hexdigest() not in excluded,
                       f"{rid}: official test overlap")
                delegate_groups[row["group_id"]].append(label)
                a = m["candidate"]["start"]
                line = prompt[prompt.rfind("\n", 0, a)+1:]
                if m["origin"] == "Med" and re.match(r"\s*[A-E][.):]\s", line):
                    note("delegateci_candidate_inside_medical_answer_choice", row)
            if row["source"] == "counterfactual":
                pairs[row["group_id"]].append(row)
            if row["mode"] == "joint":
                if (m["family"] == "vision_access" and target["task"] == "设计字号测试" and
                        "按我的实际阅读限制" in prompt and spans and spans[0]["subject"] == "third_party"):
                    note("vision_task_first_person_vs_third_party_evidence", row)
                if (m["family"] == "missed_lessons" and target["task"] == "计算补课天数" and
                        spans and spans[0]["necessity"] == "required"):
                    n = re.search(r"缺了(\d+)节", spans[0]["span"])
                    if n and int(n[1]) > 30:
                        note("education_absence_value_over_30_review_realism", row)
        for group, members in pairs.items():
            expect(len(members) == 2 and
                   {r["metadata"]["pair_variant"] for r in members} == {0,1} and
                   len({r["metadata"]["anchor"] for r in members}) == 1,
                   f"{group}: pair structure")
            kinds = {r["metadata"]["pair_type"] for r in members}
            expect(len(kinds)==1, f"{group}: pair type")
            if is_v1 and split != "train":
                expected = {r["metadata"]["expected"] for r in members}
                expect(len(expected) == 2 and all(
                    r["metadata"]["expected"] ==
                    (answer_spans(r)[0]["necessity"] if answer_spans(r) else "public")
                    for r in members), f"{group}: intervention flips anchor label")
        factor = 1 if split=="train" else 5
        expect(Counter(v[0]["metadata"]["pair_type"] for v in pairs.values()) ==
               {k:n//factor for k,n in PAIR_COUNTS.items()}, f"{split}: pair quota")
        expect(all(sorted(v)==["not_required","required"] for v in delegate_groups.values()),
               f"{split}: source-candidate group labels")
        split_stats[split] = {
            "source":dict(Counter(r["source"] for r in rows)), "delegateci_groups":len(delegate_groups),
            "normalized_prompt_count":len({normal(r["prompt"]) for r in rows}),
            "cue_diagnostics":cue_diagnostics(rows),
        }
        if is_v1 and split != "train":
            diagnostic = split_stats[split]["cue_diagnostics"]
            expect(diagnostic["pair_groups_same_request_text"] == 400,
                   f"{split}: request preserved across each evaluation pair")
            expect(diagnostic["pair_groups_same_prompt_character_multiset"] == 400,
                   f"{split}: both arms use identical character counts")
    for a,b in combinations(splits,2):
        for label,key in (
            ("id", lambda r:r["id"]), ("group", lambda r:r["group_id"]),
            ("normalized prompt",lambda r:normal(r["prompt"])),
            ("digit-normalized prompt",lambda r:re.sub(r"\d+","#",normal(r["prompt"]))),
            ("semantic family",lambda r:r["metadata"].get("family")),
        ):
            expect(not (({key(r) for r in splits[a]} & {key(r) for r in splits[b]}) - {None}),
                   f"{a}/{b} overlap: {label}")

    train = {r["id"]:r for r in splits["train"]}
    client_stats = {}
    modes = ("non_iid",) if is_v1 else ("iid", "non_iid")
    if is_v1:
        expect(quality["client_partition"] == "non_iid" and quality["client_count"] == 10,
               "v1 Non-IID partition metadata")
        expect({p.name for p in (data/"clients").iterdir()} == set(modes),
               "v1 ships only the Non-IID partition")
    for mode in modes:
        paths = sorted((data/"clients"/mode).glob("C*.jsonl"))
        expect(len(paths) == 10, f"{mode}: client count")
        all_ids, owner = Counter(), {}
        client_stats[mode] = {}
        for i,path in enumerate(paths):
            ids = [r["id"] for r in read_jsonl(path)]
            all_ids.update(ids)
            rows = [train[rid] for rid in ids]
            focus = DOMAINS[i//2]
            expect(len(ids)==1000 and Counter(r["source"] for r in rows) ==
                   dict(openpii=300,delegateci=200,counterfactual=400,complex=100),
                   f"{mode}/{path.stem}: client quota")
            for row in rows:
                expect(owner.setdefault(row["group_id"],path.stem)==path.stem,
                       f"{mode}: group {row['group_id']} split across clients")
            labels = Counter(f["necessity"] for r in rows if r["mode"]=="joint"
                             for f in answer_spans(r))
            expect(set(labels)=={"required","optional","irrelevant"},f"{mode}/{path.stem}: labels")
            domains = Counter(r["domain"] for r in rows)
            synthetic_focus = sum(r["mode"]=="joint" and r["domain"]==focus for r in rows)
            if mode=="non_iid":
                expect(synthetic_focus==400, f"{mode}/{path.stem}: synthetic domain quota")
            client_stats[mode][path.stem] = dict(
                focus_domain=focus if mode=="non_iid" else None,
                focus_rows=domains[focus] if mode=="non_iid" else None,
                synthetic_focus_rows=synthetic_focus if mode=="non_iid" else None,
                mixed_rows=domains["mixed"], domains=dict(domains), necessity=dict(labels))
        expect(all_ids == dict.fromkeys(train,1), f"{mode}: training ID coverage")

    review = read_jsonl(data/"review_queue.jsonl")
    expected_groups = defaultdict(set)
    for split,rows in splits.items():
        for row in rows:
            if row["mode"]=="joint":
                expected_groups[split,row["group_id"]].add(row["id"])
    expect(len(review)==len(expected_groups) and
           {(r["split"],r["group_id"]):set(r["row_ids"]) for r in review}==expected_groups,
           "review queue group coverage")
    return dict(structural_pass=not failures, failures=failures, split=split_stats,
                findings=dict(issues), example_ids=dict(examples), clients=client_stats,
                review_status=dict(Counter(r["status"] for r in review)))


def tokenizer_audit(data, tokenizer_path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    results = {}
    controls = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>")
    for split in ("train", "validation", "test"):
        counts = Counter(dict(rows=0, max_tokens=0, exactly_512=0, length_mismatch=0,
                              over_512=0, generation_prefix_text_mismatch=0,
                              generation_prefix_token_mismatch=0,
                              assistant_json_after_prefix_mismatch=0, raw_chat_controls=0))
        for row in read_jsonl(data/(split+".jsonl")):
            msgs = row["messages"]
            full = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
            prefix = tokenizer.apply_chat_template(
                msgs[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer.encode(full, add_special_tokens=False)
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            n = len(ids)
            counts["rows"] += 1
            counts["max_tokens"] = max(counts["max_tokens"], n)
            counts["exactly_512"] += n == 512
            counts["length_mismatch"] += n != row["token_count"]
            counts["over_512"] += n > 512
            counts["generation_prefix_text_mismatch"] += not full.startswith(prefix)
            counts["generation_prefix_token_mismatch"] += ids[:len(prefix_ids)] != prefix_ids
            counts["assistant_json_after_prefix_mismatch"] += (
                not full[len(prefix):].startswith(msgs[-1]["content"] + "<|im_end|>"))
            counts["raw_chat_controls"] += any(marker in msgs[1]["content"] for marker in controls)
        results[split] = dict(counts)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/privacy_fed_v0"))
    parser.add_argument("--raw", type=Path, default=Path("data/staging/privacy_fed_v0"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tokenizer", help="Local Qwen3 tokenizer directory for full retokenization")
    args = parser.parse_args()
    report = audit(args.data,args.raw)
    if args.tokenizer:
        report["tokenizer_recheck"] = tokenizer_audit(args.data, args.tokenizer)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="clients"}, ensure_ascii=False, indent=2))
    if not report["structural_pass"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
