"""Fingerprint public checkpoint files, independent of prompts or token IDs."""
import hashlib
import json
from pathlib import Path


def fingerprint(directory):
    root = Path(directory)
    paths = sorted(root.glob('*.safetensors'))
    if not paths:
        raise ValueError('Missing Qwen safetensors checkpoint')
    digest = hashlib.sha256()
    config = json.loads((root/'config.json').read_text())
    digest.update(json.dumps(config,sort_keys=True,separators=(',',':')).encode())
    for path in paths:
        digest.update(path.name.encode()+b'\x00')
        with path.open('rb') as handle:
            while True:
                block = handle.read(8*1024*1024)
                if not block:
                    break
                digest.update(block)
    return digest.hexdigest()
