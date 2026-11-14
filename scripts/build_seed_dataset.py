"""Build the privacy-understanding corpus. No model weights or training code."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import unicodedata


SEED = 20261112
DOMAINS = ("health", "finance", "education", "legal", "engineering")
PAIR_COUNTS = {
    "required_irrelevant": 700,
    "required_optional": 350,
    "optional_irrelevant": 350,
    "private_public": 300,
    "conditional": 300,
}
INSTRUCTIONS = {
    "joint": "输出 JSON，包含实际任务 task 与私人事实 private_facts。每项事实包含 span、start、end、type、subject 和 necessity。偏移按 Unicode 字符计，左闭右开，从 0 开始。主体用 self 或 third_party，必要性用 required、optional 或 irrelevant。",
    "spans": "找出个人信息片段。输出 JSON，private_facts 中每项含 span、start、end、type。偏移按 Unicode 字符计，左闭右开，从 0 开始。",
    "candidate_necessity": "Judge whether the candidate passage is needed to fulfill the request. Output JSON with necessity equal to required or not_required.",
}
PRIVATE_WRAPPERS = (
    "背景：{holder}说「{fact}」。{task}",
    "{task}。补充一条背景：{holder}提到「{fact}」。",
    "{holder}的情况是「{fact}」。接下来想请你{task}。",
    "麻烦处理这一项：{task}。手头有一条由{holder}提供的记录：「{fact}」。",
    "已知材料里，{holder}写道「{fact}」。请{task}。",
    "{task}。{holder}此前留下了这样一条记录——「{fact}」。",
)
PUBLIC_WRAPPERS = (
    "背景：公开教材中的虚构人物说「{fact}」。{task}",
    "{task}。补充一条背景：公开教材中的虚构人物提到「{fact}」。",
    "公开教材的虚构案例写的是「{fact}」。接下来想请你{task}。",
    "麻烦处理这一项：{task}。手头有一条公开教材里的虚构记录：「{fact}」。",
    "已知材料里，公开教材的虚构人物写道「{fact}」。请{task}。",
    "{task}。公开教材里有这样一条虚构记录——「{fact}」。",
)
EXTRAS = {
    "train": ("我家里有人正在找工作", "我最近在照看亲属", "我这阵子与家人分开住", "我临时住在亲友家"),
    "validation": ("我在周末替亲戚看店", "我最近把自己的收藏寄存在朋友家"),
    "test": ("我工作日会把孩子托给亲属照看", "我和室友正在协商分开居住"),
}
IMPLICIT = {
    "health": ("上下楼时离开扶手就容易失去平衡", "电话里连续说两句就容易漏听后一句", "每次长时间坐着起身时都得先扶一下椅背"),
    "finance": ("发薪前连公交费也得先从备用金里拿", "每到扣款日就得从下一周伙食费里挪一部分", "账单一到就只能推迟买家里的必需品"),
    "education": ("同学已经翻到下一页时还在逐字读上一段", "跟上课堂前半段后就很难继续记笔记", "轮到口头汇报时只能照着稿子逐句念"),
    "legal": ("交接文件时总要避开另一方在场的时段", "每次去交材料都要等代理人能陪同", "收到对方新文件时需要先交给代理人转收"),
    "engineering": ("内部报表一到月末就得改在夜间分批生成", "内部汇总超过一页就要先停下等上一批结束", "团队通常要先停掉一个作业才能启动内部汇总"),
}
IMPLICIT_GOALS = {
    "health": (
        ("依据实际行动困难，为楼梯写一条必须满足的安全要求", "列出楼梯安全检查的通用项目"),
        ("根据实际听讲困难安排一种电话沟通辅助方式", "介绍电话沟通辅助方式有哪些"),
        ("按实际起身困难给候诊座位定一项必须满足的要求", "整理候诊设施的通用可达性项目"),
    ),
    "finance": (
        ("把实际动用备用金的那项日常开销填进现金流表", "说明现金流表如何记录临时支出"),
        ("指出为了覆盖实际扣款挪用了哪笔生活预算", "解释记账时如何区分固定扣款和生活预算"),
        ("根据描述概括哪一类购买被账单推迟", "说明家庭账单管理表的常见用途"),
    ),
    "education": (
        ("从这段具体表现归纳课堂阅读最需要哪种进度辅助", "介绍课堂阅读活动的通用辅助方式"),
        ("根据实际表现指出听课记录应在哪个阶段加辅助", "整理课堂笔记可采用的辅助工具"),
        ("按照实际汇报方式写一项口头展示所需的现场支持", "介绍口头展示常见的现场支持"),
    ),
    "legal": (
        ("依据这份描述，写出此次文件交接排程必须避开的情况", "介绍文件交接排程的通用注意事项"),
        ("根据交材料的实际限制列出本次到场的必要安排", "列出材料递交通常可以预先确认的事项"),
        ("根据已知收件流程指出这次文件先交给谁", "列出收件交接记录的通用栏目"),
    ),
    "engineering": (
        ("按实际限制给月末报表写一条执行计划", "说明报表执行计划的通用字段"),
        ("根据实际操作写出内部汇总下一批的启动前置条件", "介绍分批汇总常见的进度控制方法"),
        ("根据实际作业约束写出启动内部汇总前必须做的一步", "列出启动作业前的通用检查项"),
    ),
}


def compact(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(compact(row) + "\n")


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normal(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def exact_span(text, span):
    if not span or text.count(span) != 1:
        return None
    start = text.index(span)
    return start, start + len(span)


def make_row(source, split, group, domain, prompt, mode, target, metadata, candidate=None):
    user = prompt if candidate is None else compact({"prompt": prompt, "candidate": candidate})
    messages = [
        {"role": "system", "content": INSTRUCTIONS[mode]},
        {"role": "user", "content": user},
        {"role": "assistant", "content": compact(target)},
    ]
    return {
        "id": digest(source + "\0" + group + "\0" + user)[:24],
        "source": source, "split": split, "group_id": group, "domain": domain,
        "prompt": prompt, "mode": mode, "target": target, "metadata": metadata,
        "messages": messages,
    }


def fact_annotation(prompt, fact, kind, subject, necessity, reason):
    offset = exact_span(prompt, fact)
    if offset is None:
        raise ValueError("Non-unique generated evidence")
    return dict(span=fact, start=offset[0], end=offset[1], type=kind,
                subject=subject, necessity=necessity), reason


def fit_tokenizer(path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

    def length(row):
        messages = row["messages"]
        # Same template for the prompt prefix and the complete SFT sequence.
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        return len(tokenizer.encode(text, add_special_tokens=False))
    return length


def check_length(row, length, maximum):
    row["token_count"] = length(row)
    return row["token_count"] <= maximum


def pii_kind(label):
    if re.search(r"PASSWORD|USERNAME|SECRET|TOKEN|IPADDRESS|^PIN$", label):
        return "CREDENTIAL"
    if re.search(r"MAIL|TELEPHONE|PHONE|FAX|URL", label):
        return "CONTACT"
    if re.search(r"STREET|CITY|STATE|COUNTRY|ADDRESS|ZIP|BUILDING|GEO|COUNTY", label):
        return "LOCATION"
    if re.search(r"BANK|CARD|CREDIT|IBAN|SALARY|AMOUNT|CURRENCY|BIC", label):
        return "FINANCIAL"
    if re.search(r"NAME|PASSPORT|LICENSE|IDCARD|SOCIALNUM|TAXNUM|TITLE|IDNUM", label):
        return "IDENTITY"
    if re.search(r"AGE|GENDER|SEX|DATE|BIRTH|HEIGHT|WEIGHT", label):
        return "DEMOGRAPHIC"
    return "OTHER_PII"


def openpii(path, rng, length, maximum, rejected):
    candidates = read_jsonl(path)
    rng.shuffle(candidates)
    selected, seen, seen_uid = [], set(), set()
    for row in candidates:
        if (row.get("language"), row.get("region"), row.get("split")) != ("zh", "CN", "train"):
            rejected["openpii:wrong_split_or_locale"] += 1
            continue
        text, masks = row["source_text"], row["privacy_mask"]
        key = normal(text)
        if key in seen or row["uid"] in seen_uid:
            rejected["openpii:duplicate"] += 1
            continue
        if not text.strip():
            rejected["openpii:empty"] += 1
            continue
        facts, raw_types = [], []
        for mask in sorted(masks, key=lambda m: (m["start"], m["end"])):
            a, b, span = mask["start"], mask["end"], mask["value"]
            if not (0 <= a < b <= len(text) and text[a:b] == span and
                    (not facts or a >= facts[-1]["end"])):
                rejected["openpii:bad_offset_or_overlap"] += 1
                break
            facts.append(dict(start=a, end=b, span=span, type=pii_kind(mask["label"])))
            raw_types.append(mask["label"])
        else:
            item = make_row("openpii", "train", "open:" + str(row["uid"]), "mixed", text,
                            "spans", {"private_facts": facts},
                            dict(uid=row["uid"], original_types=raw_types, lang="zh-CN",
                                 review="source_labels_offset_checked"))
            if not check_length(item, length, maximum):
                rejected["openpii:overlength"] += 1
                continue
            selected.append(item)
            seen.add(key)
            seen_uid.add(row["uid"])
            if len(selected) == 3000:
                break
    if len(selected) != 3000:
        raise ValueError(f"OpenPII has only {len(selected)} eligible rows")
    return selected


def delegate_domain(text, origin):
    if origin == "Med":
        return "health"
    patterns = (
        ("health", r"patient|health|doctor|symptom|medication|allerg"),
        ("finance", r"budget|salary|bank|income|loan|invest|tax"),
        ("education", r"school|student|classroom|lesson|teacher|university"),
        ("legal", r"contract|tenant|legal|lawyer|landlord|court|lawsuit"),
        ("engineering", r"python|javascript|react|code|sql|software|database|script"),
    )
    scores = [(len(re.findall(pattern, text.casefold())), domain) for domain, pattern in patterns]
    scores.sort(reverse=True)
    return scores[0][1] if scores[0][0] > scores[1][0] else "mixed"


def delegate(train, held_out, rng, length, maximum, rejected):
    queries = []
    seen = set()
    for index, row in enumerate(train, 1):
        text = row["question"]
        key = normal(text)
        if digest(key) in held_out or key in seen:
            rejected["delegate:duplicate_or_official_test"] += 1
            continue
        if row.get("split") != "train" or row.get("data_source") not in ("Med", "general_synthesis"):
            rejected["delegate:non_synthetic_train_origin"] += 1
            continue
        seen.add(key)
        group = "delegate:" + digest(key)[:24]
        sets = [list(dict.fromkeys(row[field])) for field in ("essential_spans", "non_essential_spans")]
        items = defaultdict(list)
        for i, (label, spans) in enumerate(zip(("required", "not_required"), sets)):
            for span in spans:
                pos = exact_span(text, span)
                other = [exact_span(text, s) for s in sets[1-i]]
                if (pos is None or span.strip() != span or len(span) < 3 or
                        "\n" in span or re.match(r"^[A-E][.)]\s", span) or
                        any(p is not None and max(p[0], pos[0]) < min(p[1], pos[1]) for p in other)):
                    rejected["delegate:unresolved_span"] += 1
                    continue
                candidate = dict(span=span, start=pos[0], end=pos[1])
                item = make_row("delegateci", "train", group, delegate_domain(text, row["data_source"]),
                                text, "candidate_necessity", {"necessity": label},
                                dict(row_number=index, source_question_sha256=digest(text),
                                     origin=row["data_source"], lang="en", candidate=candidate,
                                     review="source_binary_exact_alignment"), candidate=candidate)
                if not check_length(item, length, maximum):
                    rejected["delegate:overlength_candidate"] += 1
                    continue
                items[label].append(item)
        for label in items:
            rng.shuffle(items[label])
        if items:
            queries.append(items)
    # Prefer one aligned candidate of each label per original question.
    rng.shuffle(queries)
    pairs = [[q["required"][0], q["not_required"][0]] for q in queries
             if q["required"] and q["not_required"]]
    rows = [item for pair in pairs[:1000] for item in pair]
    if len(rows) != 2000:
        raise ValueError(f"DelegateCI has only {len(rows)} accepted candidate rows")
    return rows


def task_reason(label):
    return {
        "required": "遮盖该事实后，指定结果会随可能的事实取值改变，无法确定原请求的结果。",
        "optional": "不使用该事实也能交付通用结果，使用它只会让示例或表述更贴合背景。",
        "irrelevant": "遮盖该事实不影响交付目标或约束，也不改善请求所需结果。",
    }[label]


def render_fact(scenario, rng):
    _, _, template, values, *_ = scenario
    # Use per-scenario ranges to keep all generated numerical questions coherent.
    key = scenario[0]
    limits = {
        "sleep_shift": (1, 5), "test_score": (20, 59), "credits": (10, 79),
        "deductible": (2, 90), "language_vocab": (10, 99),
    }
    return template.format(n=rng.randint(*limits.get(key, (3, 90))),
                           value=rng.choice(values) if values else "")


def synthetic(split, scenarios, rng, length, maximum):
    rows, seen = [], set()
    variants = ("train", "validation", "test").index(split)
    holders = (
        ("我", "同事", "家人", "队友", "朋友", "合作方", "客户", "邻居"),
        ("我本人", "亲属", "搭档", "学习伙伴", "委托方", "组员"),
        ("我自己", "项目伙伴", "表亲", "同组同学", "受访者", "共同申请人"),
    )[variants]
    # A wrapper and an ordinary, non-private delivery constraint provide
    # surface variation while keeping the target task and dependencies fixed.
    qualifiers = (
        "答复用简体中文", "用短句给出结果", "请按步骤说明", "先给结论再解释",
        "避免缩写", "列成便于阅读的条目", "以一段简洁说明呈现", "保留明确的单位",
        "表述尽量直白", "请把关键依据单列", "用适合新手的表述", "只保留和交付有关的说明",
    )
    qualifier_styles = (
        ("{body}\n{q}。", "{q}。\n{body}", "{body}\n输出要求：{q}。"),
        ("{body}\n答复格式是：{q}。", "说明方式——{q}。\n{body}"),
        ("{q}；这是完整请求：\n{body}", "{body}\n最后一点，{q}。"),
    )[variants]
    for domain in DOMAINS:
        catalog = scenarios[domain][split]
        for pair_kind, total in PAIR_COUNTS.items():
            count = total // 5 if split == "train" else total // 25
            for i in range(count):
                group = f"syn:{split}:{domain}:{pair_kind}:{i:03d}"
                for attempt in range(200):
                    sc = catalog[(i + attempt) % len(catalog)]
                    family, kind, *_ = sc
                    fact = render_fact(sc, rng)
                    required, optional, irrelevant = sc[4:7]
                    # Generic explanations can benefit from the fact even
                    # when the user has not explicitly requested an example.
                    if i % 3 != 0 or split != "train":
                        optional = optional.split("，")[0]
                    if family == "food_allergy":
                        required = ("从花生、腰果、虾、牛奶、芝麻、鸡蛋、芒果和小麦中"
                                    "筛出这次餐食应排除的成分")
                    tasks = dict(zip(("required", "optional", "irrelevant"), sc[7:10]))
                    requests = dict(zip(tasks, (required, optional, irrelevant)))
                    holder = rng.choice(holders)
                    subject = "self" if holder == holders[0] else "third_party"
                    w = rng.randrange(len(PRIVATE_WRAPPERS))
                    q, fmt = rng.choice(qualifiers), rng.choice(qualifier_styles)
                    pair_labels = {
                        "required_irrelevant": ("required", "irrelevant"),
                        "required_optional": ("required", "optional"),
                        "optional_irrelevant": ("optional", "irrelevant"),
                        "private_public": ("required", "public"),
                        "conditional": ("required", "irrelevant"),
                    }[pair_kind]
                    pair = []
                    for variant, label in enumerate(pair_labels):
                        public = label == "public"
                        task_label = "required" if public else label
                        request = requests[task_label]
                        if pair_kind == "conditional":
                            # Keep the fact, rules and one public condition
                            # fixed. Toggle only the second public condition.
                            gate = ("甲", "乙") if i % 2 == 0 else ("乙", "甲")
                            current = gate[variant]
                            if i % 4 < 2:
                                request = (f"用途为验算且表单选{gate[0]}时，请{required}；"
                                           f"其他情况请{irrelevant}。用途是验算，表单选{current}")
                            else:
                                request = (f"仅在表单选{gate[0]}且需提交明细时才{required}；"
                                           f"否则{irrelevant}。表单选{current}，需要提交明细")
                        body = (PUBLIC_WRAPPERS if public else PRIVATE_WRAPPERS)[w].format(
                            holder=holder, fact=fact, task=request)
                        prompt = fmt.format(body=body, q=q)
                        facts, reasons = [], []
                        if not public:
                            annotation, reason = fact_annotation(prompt, fact, kind, subject,
                                                                 label, task_reason(label))
                            facts.append(annotation)
                            reasons.append(reason)
                        pair.append(make_row(
                            "counterfactual", split, group, domain, prompt, "joint",
                            {"task": tasks[task_label], "private_facts": facts},
                            dict(family=family, pair_type=pair_kind, pair_variant=variant,
                                 intervention="public_provenance" if pair_kind == "private_public"
                                 else "deliverable_selection" if pair_kind == "conditional" else "task",
                                 anchor=fact, expected=label, rationale=reasons,
                                 review="rule_generated_unreviewed", lang="zh-CN")))
                    keys = {normal(item["prompt"]) for item in pair}
                    if len(keys) == 2 and not keys & seen and all(
                        check_length(item, length, maximum) for item in pair
                    ):
                        rows.extend(pair)
                        seen.update(keys)
                        break
                else:
                    raise ValueError(f"Could not generate distinct pair {group}")
        for i in range(200 if split == "train" else 40):
            group = f"syn:{split}:{domain}:complex:{i:03d}"
            category = ("multi_fact", "subject", "implicit", "public_only", "format_negative")[i % 5]
            for attempt in range(200):
                sc = catalog[(i // 5 + attempt) % len(catalog)]
                family, kind, *_ = sc
                fact = render_fact(sc, rng)
                task_i = (i // 5) % 3
                label = ("required", "optional", "irrelevant")[task_i]
                task, request = sc[7+task_i], sc[4+task_i]
                if label == "optional" and (i % 3 != 0 or split != "train"):
                    request = request.split("，")[0]
                holder = rng.choice(holders[1:])
                extra = rng.choice(EXTRAS[split])
                facts_spec = []
                if category == "multi_fact":
                    body = f"{holder}说「{fact}」。另有一条来自我自己的背景：「{extra}」。{request}。"
                    facts_spec = [(fact, kind, "third_party", label), (extra, "PERSONAL", "self", "irrelevant")]
                elif category == "subject":
                    body = f"我在代办这件事。上游材料由{holder}确认：「{fact}」。请{request}。"
                    facts_spec = [(fact, kind, "third_party", label)]
                elif category == "implicit":
                    anchor = IMPLICIT[domain][variants]
                    if label == "irrelevant":
                        task, request = sc[9], sc[6]
                    else:
                        task = request = IMPLICIT_GOALS[domain][variants][task_i]
                    body = f"之前{holder}只说过一句「{anchor}」。现在请{request}。"
                    facts_spec = [(anchor, kind, "third_party", label)]
                elif category == "public_only":
                    body = f"公开练习册中有一条虚构背景「{fact}」。练习要求：{request}。"
                else:
                    task, request = sc[9], sc[6]
                    body = (f"表单里只填了说明文字「联系邮箱待填写」，样例字段值是"
                            f"「ITEM-{rng.randrange(10000, 99999)}」，没有填写实际个人资料。请{request}。")
                prompt = rng.choice(qualifier_styles).format(body=body, q=rng.choice(qualifiers))
                if normal(prompt) in seen:
                    continue
                annotations = [fact_annotation(prompt, f, k, s, y, task_reason(y))
                               for f, k, s, y in facts_spec]
                item = make_row("complex", split, group, domain, prompt, "joint",
                                {"task": task, "private_facts": [a for a, _ in annotations]},
                                dict(family=family, difficulty=category, lang="zh-CN",
                                     rationale=[r for _, r in annotations],
                                     review="rule_generated_unreviewed"))
                if check_length(item, length, maximum):
                    rows.append(item)
                    seen.add(normal(prompt))
                    break
            else:
                raise ValueError(f"Could not generate distinct difficult row {group}")
    return rows


def clients(rows, rng, non_iid):
    buckets = [[] for _ in range(10)]
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    source_groups = defaultdict(list)
    for group in groups.values():
        item = group[0]
        source_groups[item["source"], item["metadata"].get("pair_type", "")].append(group)
    for (source, pair_kind), entries in source_groups.items():
        rng.shuffle(entries)
        capacity = {"openpii": 300, "delegateci": 200, "complex": 100}.get(source)
        if source == "counterfactual":
            capacity = PAIR_COUNTS[pair_kind] // 5
        remaining = [capacity] * 10
        if non_iid and source in ("counterfactual", "complex"):
            by_domain = defaultdict(list)
            for group in entries:
                by_domain[group[0]["domain"]].append(group)
            spill = []
            for j, domain in enumerate(DOMAINS):
                items = by_domain[domain]
                take = len(items) * 4 // 5
                for i, group in enumerate(items[:take]):
                    client = j*2 + i%2
                    buckets[client].extend(group)
                    remaining[client] -= len(group)
                spill.extend(items[take:])
            entries = spill
        if source == "delegateci":
            entries.sort(key=len, reverse=True)
        for group in entries:
            domain = group[0]["domain"]
            available = [i for i in range(10) if remaining[i] >= len(group)]
            if not available:
                raise ValueError(f"Group allocation failed for {source}")
            if non_iid and source == "delegateci" and domain in DOMAINS:
                preferred = [i for i in available if DOMAINS[i//2] == domain]
                available = preferred or available
            elif non_iid and source in ("counterfactual", "complex"):
                # The held-back 20% is sent to a different focus domain.
                destination = [(DOMAINS.index(domain)+1) % len(DOMAINS)]
                preferred = [i for i in available if i//2 in destination]
                available = preferred or available
            client = rng.choice([i for i in available if remaining[i] == max(remaining[j] for j in available)])
            remaining[client] -= len(group)
            buckets[client].extend(group)
        if any(remaining):
            raise ValueError(f"Unfilled quota for {source}: {remaining}")
    return buckets


def audit(splits, partitions, maximum):
    ids, groups, prompts, families = {}, {}, {}, {}
    for split, rows in splits.items():
        expected = 10000 if split == "train" else 1000
        assert len(rows) == expected, (split, len(rows))
        for row in rows:
            assert row["id"] not in ids and row["split"] == split
            ids[row["id"]] = split
            for registry, key in ((groups, row["group_id"]), (prompts, normal(row["prompt"]))):
                assert registry.setdefault(key, split) == split
            family = row["metadata"].get("family")
            if family:
                assert families.setdefault(family, split) == split
            assert row["token_count"] <= maximum
            assert json.loads(row["messages"][-1]["content"]) == row["target"]
            for span in row["target"].get("private_facts", []) + (
                [row["metadata"]["candidate"]] if row["mode"] == "candidate_necessity" else []
            ):
                assert 0 <= span["start"] < span["end"] <= len(row["prompt"])
                assert row["prompt"][span["start"]:span["end"]] == span["span"]
        pair_groups = defaultdict(list)
        for row in rows:
            if row["source"] == "counterfactual":
                pair_groups[row["group_id"]].append(row)
        for pair in pair_groups.values():
            assert len(pair) == 2
            a, b = (row["metadata"] for row in pair)
            assert (a["pair_variant"], b["pair_variant"]) == (0, 1)
            assert a["pair_type"] == b["pair_type"] and a["expected"] != b["expected"]
            assert a["anchor"] == b["anchor"] and all(a["anchor"] in row["prompt"] for row in pair)
        multiplier = 1 if split == "train" else 5
        assert Counter(pair[0]["metadata"]["pair_type"] for pair in pair_groups.values()) == {
            k: v//multiplier for k, v in PAIR_COUNTS.items()}
    for buckets in partitions.values():
        assert {r["id"] for bucket in buckets for r in bucket} == {
            r["id"] for r in splits["train"]}
        ownership = {}
        for index, bucket in enumerate(buckets):
            assert len(bucket) == 1000
            assert Counter(r["source"] for r in bucket) == dict(
                openpii=300, delegateci=200, counterfactual=400, complex=100)
            for row in bucket:
                assert ownership.setdefault(row["group_id"], index) == index
            assert {f["necessity"] for r in bucket if r["mode"] == "joint"
                    for f in r["target"]["private_facts"]} == {"required", "optional", "irrelevant"}


def summarize(rows):
    return {
        "rows": len(rows),
        "unique_groups": len({r["group_id"] for r in rows}),
        "unique_raw_prompts": len({normal(r["prompt"]) for r in rows}),
        "source": dict(Counter(r["source"] for r in rows)),
        "domains": dict(Counter(r["domain"] for r in rows)),
        "synthetic_domains": dict(Counter(r["domain"] for r in rows if r["mode"] == "joint")),
        "joint_fact_necessity": dict(Counter(f["necessity"] for r in rows if r["mode"] == "joint"
                                           for f in r["target"]["private_facts"])),
        "binary_necessity": dict(Counter(r["target"]["necessity"] for r in rows
                                         if r["mode"] == "candidate_necessity")),
        "explicit_empty_targets": dict(Counter(r["source"] for r in rows
                                              if r["target"].get("private_facts") == [])),
        "joint_task_classes": len({r["target"]["task"] for r in rows if r["mode"] == "joint"}),
        "joint_difficulties": dict(Counter(r["metadata"]["difficulty"] for r in rows
                                         if r["source"] == "complex")),
        "tokens": dict(min=min(r["token_count"] for r in rows),
                       median=statistics.median(r["token_count"] for r in rows),
                       max=max(r["token_count"] for r in rows)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output directory exists; choose a new version.")
    source_manifest = json.loads((args.raw/"source_manifest.json").read_text(encoding="utf-8"))
    for name, expected in source_manifest["files_sha256"].items():
        if hashlib.sha256((args.raw/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Staged source hash mismatch: {name}")
    rng = random.Random(SEED)
    length = fit_tokenizer(args.tokenizer)
    scenarios_path = Path(__file__).with_name("privacy_fed_scenarios.json")
    scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))
    rejected = Counter()
    rows_open = openpii(args.raw/"open_zh_candidates.jsonl", rng, length, args.max_length, rejected)
    official_train = read_jsonl(args.raw/"delegate_train.jsonl")
    official_test = json.loads((args.raw/"delegate_test_fingerprints.json").read_text(encoding="utf-8"))
    rows_delegate = delegate(official_train, set(official_test["normalized_question_sha256"]),
                             rng, length, args.max_length, rejected)
    splits = {split: synthetic(split, scenarios, rng, length, args.max_length)
              for split in ("train", "validation", "test")}
    splits["train"].extend(rows_open + rows_delegate)
    partitions = {mode: clients(splits["train"], random.Random(SEED+i), mode == "non_iid")
                  for i, mode in enumerate(("iid", "non_iid"))}
    audit(splits, partitions, args.max_length)
    args.out.mkdir(parents=True)
    for split, rows in splits.items():
        # Keep pair IDs as metadata; training order does not encode the label.
        rng.shuffle(rows)
        write_jsonl(args.out/(split+".jsonl"), rows)
    for mode, buckets in partitions.items():
        destination = args.out/"clients"/mode
        destination.mkdir(parents=True)
        for i, bucket in enumerate(buckets, 1):
            rng.shuffle(bucket)
            write_jsonl(destination/f"C{i:02}.jsonl", [{"id": r["id"]} for r in bucket])
    review_queue = []
    for split in ("test", "validation", "train"):
        grouped = defaultdict(list)
        for row in splits[split]:
            if row["mode"] == "joint":
                grouped[row["group_id"]].append(row["id"])
        review_queue.extend(dict(split=split, group_id=g, row_ids=ids, status="pending")
                            for g, ids in sorted(grouped.items()))
    write_jsonl(args.out/"review_queue.jsonl", review_queue)
    report = {
        "version": "v0-candidate", "seed": SEED, "max_length": args.max_length,
        "tokenizer": "Qwen3-1.7B local chat template",
        "tokenizer_input_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(Path(args.tokenizer).glob("tokenizer*")) if p.is_file()},
        "scenario_sha256": hashlib.sha256(scenarios_path.read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "split": {s: summarize(rows) for s, rows in splits.items()},
        "delegateci": dict(official_train_queries=len(official_train), official_test_queries=official_test["rows"],
                           derived_rows=len(rows_delegate),
                           unique_queries=len({r["group_id"] for r in rows_delegate}),
                           official_test_rows_used=0, translated_rows=0),
        "source_filters": dict(rejected),
        "clients": {mode: {f"C{i+1:02}": dict(summarize(rows), focus_domain=DOMAINS[i//2]
                                              if mode == "non_iid" else None)
                            for i, rows in enumerate(buckets)} for mode, buckets in partitions.items()},
        "checks": dict(all_counts=True, no_cross_split_groups_or_exact_prompts=True,
                       no_cross_split_scenario_families=True, no_cross_client_groups=True,
                       span_offsets_exact=True, full_chat_sequences_within_max_length=True,
                       synthetic_labels_human_reviewed=False, training_started=False),
        "sha256": {str(p.relative_to(args.out)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(args.out.rglob("*.jsonl"))},
    }
    (args.out/"quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n",
                                               encoding="utf-8")
    (args.out/"source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(split=report["split"], delegateci=report["delegateci"],
                          rejected=report["source_filters"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
