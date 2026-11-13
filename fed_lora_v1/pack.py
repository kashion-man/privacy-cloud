"""Package verified Non-IID training data without the held-out test split."""

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    source = args.dataset
    quality = json.loads((source/"quality_report.json").read_text(encoding="utf-8"))
    if quality["client_partition"]!="non_iid" or quality["client_count"]!=10:
        raise ValueError("Expected the audited Non-IID-only dataset")
    payload = {}
    digests = {}
    for split,keys in (("train",("id","group_id","mode","messages","token_count")),
                       ("validation",("id","group_id","mode","messages","token_count","prompt","target"))):
        name = split+".jsonl"
        raw = (source/name).read_bytes()
        digests[name] = sha256(raw).hexdigest()
        if digests[name]!=quality["sha256"][name]:
            raise ValueError(f"Unverified dataset: {name}")
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        packed = [json.dumps({k:r[k] for k in keys},ensure_ascii=False,separators=(",",":"))
                  for r in rows]
        payload[name] = ("\n".join(packed)+"\n").encode()
    for i in range(1,11):
        short = f"C{i:02}.jsonl"
        name = "clients/non_iid/"+short
        raw = (source/name).read_bytes()
        if sha256(raw).hexdigest()!=quality["sha256"][name]:
            raise ValueError(f"Unverified client map: {short}")
        payload["clients/"+short] = raw
    manifest = dict(dataset_version=quality["version"],client_partition="non_iid",
                    original_splits_sha256=digests,
                    sha256={k:sha256(v).hexdigest() for k,v in payload.items()})
    payload["manifest.json"] = json.dumps(manifest,indent=2).encode()
    files = {"data/"+k:v for k,v in payload.items()}
    root = Path(__file__).parent
    for name in ("run.py","worker.py","config.json","privacy.py","privacy.example.json"):
        files[name] = (root/name).read_bytes()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(args.output,"w:gz",compresslevel=6) as archive:
        for name,data in files.items():
            info = tarfile.TarInfo(name)
            info.size,info.mode = len(data),0o600
            archive.addfile(info,io.BytesIO(data))
    print(f"bundle {args.output.stat().st_size} bytes sha256 {sha256(args.output.read_bytes()).hexdigest()}")


if __name__=="__main__":
    main()
