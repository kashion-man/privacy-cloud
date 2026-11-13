"""Read public safetensors weights without importing a plaintext model runtime."""
from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np


class Weights:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.config = json.loads((self.directory / 'config.json').read_text())
        if self.config.get('model_type') != 'qwen3':
            raise ValueError('Only dense Qwen3 checkpoints are supported')
        self.entries = {}
        for path in sorted(self.directory.glob('*.safetensors')):
            with path.open('rb') as handle:
                size = struct.unpack('<Q', handle.read(8))[0]
                if size > 16 * 1024 * 1024:
                    raise ValueError('Invalid safetensors header')
                header = json.loads(handle.read(size))
            for name, info in header.items():
                if name != '__metadata__':
                    if name in self.entries:
                        raise ValueError('Duplicate weight: ' + name)
                    self.entries[name] = (path, 8 + size, info)

    def tensor(self, name):
        if name == 'lm_head.weight' and name not in self.entries:
            if not self.config.get('tie_word_embeddings'):
                raise KeyError(name)
            name = 'model.embed_tokens.weight'
        path, start, info = self.entries[name]
        dtype = {'BF16': '<u2', 'F16': '<f2', 'F32': '<f4', 'F64': '<f8'}[info['dtype']]
        offsets = info['data_offsets']
        array = np.memmap(path, mode='r', dtype=dtype,
                          offset=start + offsets[0], shape=tuple(info['shape']))
        if array.nbytes != offsets[1] - offsets[0]:
            raise ValueError('Weight shape does not match its byte range')
        if info['dtype'] == 'BF16':
            return np.left_shift(np.asarray(array, dtype=np.uint32), 16).view(np.float32)
        return np.asarray(array, dtype=np.float32)

    def embedding(self, ids):
        return self.tensor('model.embed_tokens.weight')[np.asarray(ids, dtype=np.int64)]
