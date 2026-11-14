"""Fetch pinned public inputs for the dataset builder without retaining official test text."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from build_seed_dataset import digest, normal, read_jsonl, write_jsonl


OPEN_REPO = "ai4privacy/pii-masking-openpii-1.5m"
OPEN_REV = "a785eb528e28be2693c3718a27e066970de5dadb"
DELEGATE_REPO = "Qianfan625/DelegateCI"
DELEGATE_REV = "ff05060c04205150fd09d31f5a25c727fcdbb0c5"
RANGE_END = (128 << 20) - 1


def url(repo, rev, relative):
    return f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{relative}"


def test_fingerprints(data):
    questions = [json.loads(line)["question"] for line in data.splitlines() if line.strip()]
    return dict(rows=len(questions), upstream_bytes_sha256=hashlib.sha256(data).hexdigest(),
                normalized_question_sha256=sorted({digest(normal(q)) for q in questions}))


def fetch_open(destination):
    request = urllib.request.Request(url(OPEN_REPO, OPEN_REV, "data/train.jsonl"),
                                     headers={"Range": f"bytes=0-{RANGE_END}",
                                              "Accept-Encoding": "identity"})
    temporary = destination.with_suffix(".partial")
    with urllib.request.urlopen(request, timeout=90) as response:
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(
                f"bytes 0-{RANGE_END}/"):
            raise RuntimeError("Server did not honor the bounded OpenPII byte range")
        def candidates():
            for line in response:
                if not line.endswith(b"\n"):
                    continue
                row = json.loads(line)
                if row["language"] == "zh" and row["region"] == "CN":
                    yield {k: row[k] for k in ("uid", "source_text", "privacy_mask",
                                               "language", "region", "split", "source_dataset")}
        write_jsonl(temporary, candidates())
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--use-staged", action="store_true",
                        help="Use already fetched inputs and convert staged official test to hashes")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    opened = args.out/"open_zh_candidates.jsonl"
    train = args.out/"delegate_train.jsonl"
    test_path = args.out/"delegate_test_fingerprints.json"
    if args.use_staged:
        raw_test = args.out/"delegate_test.jsonl"
        test = (test_fingerprints(raw_test.read_bytes()) if raw_test.exists()
                else json.loads(test_path.read_text(encoding="utf-8")))
        write_jsonl(opened.with_suffix(".canonical"), read_jsonl(opened))
        opened.with_suffix(".canonical").replace(opened)
    else:
        fetch_open(opened)
        with urllib.request.urlopen(url(DELEGATE_REPO, DELEGATE_REV, "data/train.jsonl"), timeout=90) as r:
            train.write_bytes(r.read())
        with urllib.request.urlopen(url(DELEGATE_REPO, DELEGATE_REV, "data/test.jsonl"), timeout=90) as r:
            test = test_fingerprints(r.read())
    test_path.write_text(json.dumps(test, indent=2)+"\n", encoding="utf-8")
    manifest = {
        "openpii": dict(repo=OPEN_REPO, revision=OPEN_REV, split="train", locale="zh-CN",
                        upstream_prefix_bytes=RANGE_END+1, staged_rows=len(read_jsonl(opened)),
                        license_as_stated_by_publisher="CC BY 4.0", attribution="Ai Suisse SA"),
        "delegateci": dict(repo=DELEGATE_REPO, revision=DELEGATE_REV, split="train",
                           upstream_training_rows=len(read_jsonl(train)), official_test_rows=test["rows"],
                           official_test_bytes_sha256=test["upstream_bytes_sha256"],
                           published_license_field=None),
        "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (opened, train, test_path)},
    }
    (args.out/"source_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
