"""Build the audited Non-IID dataset from the seed corpus."""

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import random
import re
import tempfile

import build_seed_dataset as base
from audit_dataset import audit, cue_diagnostics


SEED = 20261113
OPTION_LINE = re.compile(r"\s*[A-E][.):]\s")
EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}")
NAMES = (("林岑", "许禾"), ("陈远", "周宁"), ("乔溪", "程安"), ("叶舟", "唐芮"),
         ("顾南", "沈栩"), ("何新", "方迟"), ("简原", "白珂"), ("孟秋", "陆言"))
DISCRETION = ("有匹配背景时可附一个贴合本对象的例子。",
              "如果手头有与这次对象直接相关的材料，可以顺带举例。",
              "只在能结合该对象的情况时补一个示例。")
REVIEW = "rule_generated_unreviewed"
FALLBACK_TASKS = (
    "照项目名称排一下目录顺序", "把现有表单的栏名改成问句",
    "将记录页标题改成短标题", "用一句话描述文件的编排方式",
    "按附件标题的首字排列附件目录",
)


@dataclass(frozen=True)
class Scene:
    domain: str
    family: str
    kind: str
    fact: str
    compute: str
    generic: str
    multiplier: int
    unit: str


def load_scenes(path):
    scenes = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        split, domain, family, kind, fact, compute, generic, multiplier, unit = line.split("|")
        scenes[split, domain].append(Scene(domain, family, kind, fact, compute, generic,
                                          int(multiplier), unit))
    return scenes


def has_medical_option(prompt, span, origin):
    start = base.exact_span(prompt, span)
    if origin != "Med" or start is None:
        return False
    line = prompt[prompt.rfind("\n", 0, start[0])+1:]
    return bool(OPTION_LINE.match(line))


def repair_sources(old, raw, length, rng, maximum, held_out):
    opened = [r for r in old if r["source"] == "openpii"]
    good_open = [r for r in opened if r["target"]["private_facts"] or not EMAIL.search(r["prompt"])]
    if len(good_open) < len(opened):
        eligible = [r for r in base.read_jsonl(raw/"open_zh_candidates.jsonl")
                    if r["privacy_mask"] or not EMAIL.search(r["source_text"])]
        occupied = {base.normal(r["prompt"]) for r in good_open}
        occupied_uids = {r["metadata"]["uid"] for r in good_open}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"open.jsonl"
            base.write_jsonl(path, eligible)
            replacements = base.openpii(path, rng, length, maximum, defaultdict(int))
        for row in replacements:
            if base.normal(row["prompt"]) in occupied or row["metadata"]["uid"] in occupied_uids:
                continue
            good_open.append(row)
            if len(good_open) == 3000:
                break
    originals = [r for r in old if r["source"] == "delegateci"]
    flagged = [r for r in originals if has_medical_option(
        r["prompt"], r["metadata"]["candidate"]["span"], r["metadata"]["origin"])]
    flagged_groups = {r["group_id"] for r in flagged}
    retained = [r for r in originals if r["group_id"] not in flagged_groups]
    occupied = {r["group_id"] for r in retained}
    prepared = base.read_jsonl(raw/"delegate_train.jsonl")
    for row in prepared:
        for field in ("essential_spans", "non_essential_spans"):
            row[field] = [s for s in row[field] if not has_medical_option(
                row["question"], s, row["data_source"])]
    candidates = base.delegate(prepared, held_out, rng, length, maximum, defaultdict(int))
    candidate_groups = defaultdict(list)
    for row in candidates:
        if row["group_id"] not in occupied:
            candidate_groups[row["group_id"]].append(row)
    ordered = sorted(candidate_groups, key=lambda group: group not in flagged_groups)
    revised = retained + [r for group in ordered[:len(flagged_groups)] for r in candidate_groups[group]]
    assert len(good_open) == 3000 and len(revised) == 2000
    return good_open + revised, dict(openpii_rows_replaced=len(opened) -
                                    len({r["id"] for r in good_open} & {r["id"] for r in opened}),
                                    delegate_flagged_rows=len(flagged),
                                    delegate_groups_reprocessed=len(flagged_groups),
                                    delegate_candidate_rows_replaced=len(originals) -
                                    len({r["id"] for r in revised} & {r["id"] for r in originals}))


def clone_joint(row, prompt, facts, metadata, length, maximum):
    spans, rationales = [], []
    for span, kind, subject, label, reason in facts:
        item, rationale = base.fact_annotation(prompt, span, kind, subject, label, reason)
        spans.append(item)
        rationales.append(rationale)
    metadata["rationale"] = rationales
    revised = base.make_row(row["source"], row["split"], row["group_id"], row["domain"],
                            prompt, "joint", {"task": row["target"]["task"], "private_facts": spans},
                            metadata)
    if not base.check_length(revised, length, maximum):
        raise ValueError(f"Repaired sequence exceeds token limit: {row['group_id']}")
    return revised


def repair_training(old, length, maximum):
    patterns = json.loads(Path(base.__file__).with_name("privacy_fed_scenarios.json")
                          .read_text(encoding="utf-8"))
    scenarios = {s[0]: s for p in patterns.values() for s in p["train"]}
    joint = [r for r in old if r["mode"] == "joint"]
    grouped = defaultdict(list)
    for row in joint:
        grouped[row["group_id"]].append(row)
    out, seen = [], set()
    for group in grouped.values():
        edited = []
        for row in sorted(group, key=lambda r:r["metadata"].get("pair_variant",0)):
            meta, prompt = deepcopy(row["metadata"]), row["prompt"]
            old_facts = row["target"]["private_facts"]
            anchor = meta.get("anchor") or (old_facts[0]["span"] if old_facts else "")
            sc = scenarios[meta["family"]]
            changed_span = anchor
            if sc[0] == "missed_lessons":
                num = re.search(r"缺了(\d+)节", anchor)
                if num and int(num[1]) > 30:
                    changed_span = anchor.replace(num[1], str(5 + int(num[1]) % 24), 1)
                    prompt = prompt.replace(anchor, changed_span, 1)
                    if "anchor" in meta:
                        meta["anchor"] = changed_span
            if sc[0] == "vision_access":
                # The request refers to the cited record, including on the
                # public side of a provenance intervention.
                prompt = prompt.replace("按我的实际阅读限制", "按记录中描述的阅读限制")
            if meta.get("difficulty") != "implicit":
                # Rephrase the original generator's optional clauses as a
                # discretionary example tied to the cited, relevant context.
                stem = sc[5].split("，")[0]
                if sc[5] in prompt:
                    prompt = prompt.replace(sc[5], stem, 1)
                if row["target"]["task"] == sc[8] and stem in prompt:
                    prompt = prompt.replace(stem, stem + "，如有与该对象直接相关的背景可举一例", 1)
                if (row["target"]["task"] == sc[9] and sc[6] in prompt and
                        meta.get("pair_type") != "conditional"):
                    prompt = prompt.replace(sc[6], sc[6] + "，如有与该对象直接相关的背景可举一例", 1)
            if meta.get("difficulty") == "public_only":
                prompt = re.sub(r"(缺了)(\d+)(节)", lambda m: m[1]+str(
                    int(m[2]) if int(m[2]) <= 30 else 5 + int(m[2]) % 24)+m[3], prompt)
            facts = []
            for f in old_facts:
                span = changed_span if f["span"] == anchor else f["span"]
                reason = base.task_reason(f["necessity"])
                if f["necessity"] == "optional" and "如有与该对象直接相关的背景可举一例" in prompt:
                    reason = "该事实能支持一条与此对象相符的可选示例。说明或清单本身无需该事实。"
                facts.append((span, f["type"], f["subject"], f["necessity"], reason))
            edited.append((row, prompt, facts, meta))
        texts = {base.normal(prompt) for _,prompt,_,_ in edited}
        if texts & seen:
            # Bounded values can collide when the old prompts differ only in
            # their large numbers. Keep the same neutral qualifier in both arms.
            suffix = 1
            while {base.normal(p + f" 请按{suffix}段给出。") for _,p,_,_ in edited} & seen:
                suffix += 1
            edited = [(r,p+f" 请按{suffix}段给出。",f,m) for r,p,f,m in edited]
            texts = {base.normal(prompt) for _,prompt,_,_ in edited}
        assert len(texts) == len(edited)
        seen.update(texts)
        out.extend(clone_joint(r,p,f,m,length,maximum) for r,p,f,m in edited)
    return out


def format_request(request, context, style):
    if style == 0:
        return f"{context}\n麻烦按下面这项处理：{request}"
    if style == 1:
        return f"{request}\n供你判断的记录如下。{context}"
    if style == 2:
        return f"需要交付：{request}\n材料是这样的：{context}"
    return f"{context}\n{request}"


def eval_pair(scene, kind, a, b, names, style, index):
    first, second = names
    evidence = (scene.fact.format(n=a), scene.fact.format(n=b))
    flip = index % 2
    target_task = scene.compute
    if kind in ("required_irrelevant", "optional_irrelevant"):
        target_task = scene.compute if kind == "required_irrelevant" else scene.generic
        positive = "required" if kind == "required_irrelevant" else "optional"
        request = (f"仅为本次排入名单的对象办理。{target_task}。"
                   + (DISCRETION[index%len(DISCRETION)] if positive=="optional" else ""))
        output = []
        for j in range(2):
            selected = (first,second)[j ^ flip]
            comparison = (first,second)[1 ^ j ^ flip]
            context = (f"{first}说「{evidence[0]}」。{second}说「{evidence[1]}」。"
                       f"名单上是{selected}，用于核对格式的另一份记录来自{comparison}。")
            labels = (positive,"irrelevant") if selected==first else ("irrelevant",positive)
            reasons = tuple("这是本次办理对象的事实。" if y != "irrelevant"
                            else "这是对照对象的情况，不能用来替代本次对象的条件。"
                            for y in labels)
            output.append((format_request(request,context,style), target_task,
                           [(span,scene.kind,"third_party",label,reason)
                            for span,label,reason in zip(evidence,labels,reasons)], labels[0]))
        return request, evidence[0], "subject_binding", output
    if kind == "required_optional":
        request = (f"{scene.compute}。当前批次若有已核定结果，直接登记；"
                   "如果没有结果，依据当前原始记录推算。已有结果时，计算过程可以作为复核附注。")
        output = []
        value = f"{a*scene.multiplier}{scene.unit}"
        other_value = f"{b*scene.multiplier}{scene.unit}"
        for j in range(2):
            current, old = ("乙","甲") if j ^ flip else ("甲","乙")
            label = "optional" if current=="甲" else "required"
            context = (f"本次对象的情况写在补录里：「{evidence[0]}」。"
                       f"甲表已经核定结果为{value}，乙表没有核定结果，只有草算{other_value}。"
                       f"这次引用的是{current}表，另一张{old}表属于上期。")
            reason = ("甲表给出本次可直接登记的结果，原始事实只用于可选复核。"
                      if label=="optional" else "本次乙表的草算不是已核定结果，原始事实用于独立推算。")
            output.append((format_request(request,context,style), target_task,
                           [(evidence[0],scene.kind,"third_party",label,reason)],label))
        return request, evidence[0], "substitute_validity", output
    if kind == "private_public":
        request = f"用下面引文中的参数完成这项工作：{scene.compute}。"
        output = []
        for j in range(2):
            original, demo = ("乙","甲") if j ^ flip else ("甲","乙")
            private = original == "甲"
            context = (f"档案{original}为当事人亲自提供的实际情况，档案{demo}里的参数是演示人员编的。"
                       f"现在打开的是甲号档案，其中写着「{evidence[0]}」。")
            facts = ([(evidence[0],scene.kind,"third_party","required",
                       "引文来自实际当事人，且指定结果需要此参数。")] if private else [])
            output.append((format_request(request,context,style),target_task,facts,
                           "required" if private else "public"))
        return request, evidence[0], "provenance_binding", output
    if kind != "conditional":
        raise ValueError(kind)
    fallback = FALLBACK_TASKS[index%len(FALLBACK_TASKS)]
    request = (f"同时满足‘走复核流程’和‘已接收数值材料’时，{scene.compute}。"
               f"其余情况只{fallback}。以本次卡片为准。")
    output = []
    for j in range(2):
        received, pending = ("乙","甲") if j ^ flip else ("甲","乙")
        selected = received == "甲"
        label = "required" if selected else "irrelevant"
        context = (f"这次走复核流程，使用甲号卡片。{received}号的数值材料已接收，"
                   f"{pending}号还在等待。甲号附录写着「{evidence[0]}」。")
        reason = ("甲号同时满足两项前置条件，数值交付需要该事实。"
                  if selected else "当前卡片的材料状态未通过，实际交付只涉及文档格式。")
        output.append((format_request(request,context,style),target_task if selected else fallback,
                       [(evidence[0],scene.kind,"third_party",label,reason)],label))
    return request, evidence[0], "joint_condition_binding", output


def eval_row(source, split, scene, group, prompt, task, facts, meta, length, maximum):
    original = dict(source=source, split=split, group_id=group, domain=scene.domain,
                    target={"task":task})
    return clone_joint(original,prompt,facts,dict(meta,family=scene.family,lang="zh-CN",
                                                 review=REVIEW),length,maximum)


def eval_set(split, scenes, rng, length, maximum):
    rows, seen = [], set()
    for domain in base.DOMAINS:
        catalog = scenes[split,domain]
        for pair_kind,total in base.PAIR_COUNTS.items():
            for i in range(total//25):
                scene = catalog[i%len(catalog)]
                group = f"v1:{split}:{domain}:{pair_kind}:{i:03d}"
                for _ in range(100):
                    a,b = rng.sample(range(2,29),2)
                    names,style = rng.choice(NAMES),rng.randrange(4)
                    request,anchor,intervention,pair = eval_pair(
                        scene,pair_kind,a,b,names,style,i)
                    if {base.normal(p[0]) for p in pair} & seen:
                        continue
                    assert sorted(pair[0][0]) == sorted(pair[1][0]), group
                    for j,(prompt,task,facts,label) in enumerate(pair):
                        rows.append(eval_row("counterfactual",split,scene,group,prompt,task,facts,
                                    dict(pair_type=pair_kind,pair_variant=j,anchor=anchor,
                                         request=request,expected=label,intervention=intervention),
                                    length,maximum))
                        seen.add(base.normal(prompt))
                    break
                else:
                    raise ValueError(f"Could not render a distinct pair: {group}")
        for i in range(40):
            scene = catalog[(i//5)%len(catalog)]
            kind = ("multi_fact","subject","implicit","public_only","format_negative")[i%5]
            group = f"v1:{split}:{domain}:complex:{i:03d}"
            for _ in range(100):
                first,second = rng.choice(NAMES)
                a,b = rng.sample(range(2,29),2)
                fact,other = scene.fact.format(n=a),scene.fact.format(n=b)
                request,task,facts = scene.compute,scene.compute,[]
                if kind == "multi_fact":
                    context = (f"我在替{first}核算，{first}说「{fact}」。"
                               f"{second}还说过「{other}」，我也记下了。")
                    facts = [(fact,scene.kind,"third_party","required","本次对象的原始参数。"),
                             (other,scene.kind,"third_party","irrelevant","另一对象的记录。")]
                elif kind == "subject":
                    context = (f"{first}帮我转述了我自己的情况：「{fact}」。"
                               f"这次申请的是我，{first}只负责送材料。")
                    facts = [(fact,scene.kind,"self","required","发话者是代传者，事实主体是申请人本人。")]
                elif kind == "implicit":
                    # These are observations rather than inferred diagnoses
                    # or sensitive categories. The optional use is explicit.
                    idx = ("train","validation","test").index(split)
                    observed = base.IMPLICIT[domain][idx]
                    slot = (i//5)%3
                    label = ("required","optional","irrelevant")[slot]
                    task = request = (base.IMPLICIT_GOALS[domain][idx][slot] if slot<2
                                      else FALLBACK_TASKS[i%len(FALLBACK_TASKS)])
                    if label == "optional":
                        request += "，有相关的实际情形可补一条例子"
                    context = f"{first}没有给总结，只留了这句记录：「{observed}」。交给我整理。"
                    facts = [(observed,scene.kind,"third_party",label,base.task_reason(label))]
                elif kind == "public_only":
                    context = f"想跑一下界面演示，演示员把参数随手设成「{fact}」，用于验证计算按钮。"
                else:
                    task = request = FALLBACK_TASKS[(i//5)%len(FALLBACK_TASKS)]
                    serial = rng.randrange(100,9999)
                    context = (f"界面上的联系人栏显示‘等待录入’，测试任务编码是 REF-{serial}。"
                               "我只需要把表的样式定下来。" if split=="validation" else
                               f"请看这个空的交接界面，联系人只显示空白下划线。"
                               f"右上角的 DEMO-{serial} 是表单版本号。")
                prompt = format_request(request,context,rng.randrange(4))
                if base.normal(prompt) in seen:
                    continue
                rows.append(eval_row("complex",split,scene,group,prompt,task,facts,
                            {"difficulty":kind},length,maximum))
                seen.add(base.normal(prompt))
                break
            else:
                raise ValueError(f"Could not render a distinct case: {group}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=Path("data/seed"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a fresh output directory")
    maximum, rng = 512, random.Random(SEED)
    length = base.fit_tokenizer(args.tokenizer)
    manifest = json.loads((args.old/"source_manifest.json").read_text(encoding="utf-8"))
    for name,sha in manifest["files_sha256"].items():
        if hashlib.sha256((args.raw/name).read_bytes()).hexdigest()!=sha:
            raise ValueError(f"Provenance mismatch: {name}")
    held_out = set(json.loads((args.raw/"delegate_test_fingerprints.json")
                             .read_text(encoding="utf-8"))["normalized_question_sha256"])
    old = base.read_jsonl(args.old/"train.jsonl")
    external,changes = repair_sources(old,args.raw,length,rng,maximum,held_out)
    train = repair_training(old,length,maximum)+external
    scene_path = Path(__file__).with_name("evaluation_scenarios.txt")
    scenes = load_scenes(scene_path)
    splits = {"train":train}
    for split in ("validation","test"):
        splits[split] = eval_set(split,scenes,rng,length,maximum)
    buckets = base.clients(train, random.Random(SEED+1), non_iid=True)
    base.audit(splits, {"non_iid": buckets}, maximum)
    args.out.mkdir(parents=True)
    for split,rows in splits.items():
        rng.shuffle(rows)
        base.write_jsonl(args.out/(split+".jsonl"),rows)
    directory = args.out/"clients"/"non_iid"
    directory.mkdir(parents=True)
    for i, rows in enumerate(buckets, 1):
        rng.shuffle(rows)
        base.write_jsonl(directory/f"C{i:02}.jsonl", [{"id":r["id"]} for r in rows])
    queue = []
    for split in ("test","validation","train"):
        groups = defaultdict(list)
        for row in splits[split]:
            if row["mode"]=="joint":
                groups[row["group_id"]].append(row["id"])
        queue.extend(dict(split=split,group_id=g,row_ids=ids,status="pending")
                     for g,ids in sorted(groups.items()))
    base.write_jsonl(args.out/"review_queue.jsonl",queue)
    (args.out/"source_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    root = Path(__file__).parent
    report = dict(version="v1-candidate",seed=SEED,max_length=maximum,
                  client_partition="non_iid",client_count=len(buckets),
                  builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scenario_sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest(),
                  base_builder_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
                  split={s:base.summarize(rows) for s,rows in splits.items()},
                  source_changes=changes,
                  train_joint_rows_rewritten=len(
                      {r["id"] for r in train if r["mode"]=="joint"} -
                      {r["id"] for r in old if r["mode"]=="joint"}),
                  synthetic_labels_human_reviewed=False,training_started=False,
                  sha256={str(p.relative_to(args.out).as_posix()):
                          hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(args.out.rglob("*.jsonl"))})
    (args.out/"quality_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",
                                               encoding="utf-8")
    checked = audit(args.out,args.raw)
    if not checked["structural_pass"]:
        raise ValueError(checked["failures"][:20])
    if checked["findings"]:
        raise ValueError(f"Original targeted issues remain: {checked['findings']}")
    print(json.dumps(dict(changes=changes, rewritten=report["train_joint_rows_rewritten"],
                          held_out_diagnostics=cue_diagnostics(splits["test"])),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
