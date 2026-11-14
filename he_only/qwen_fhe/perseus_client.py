"""Trusted native client; its random secret is never written to the evaluator bundle."""
import ctypes
import gc
import json
from pathlib import Path
import numpy as np

from .perseus_context import configuration, write_context, load_programs, rotation_steps


class PerseusClient:
    def __init__(self, width=2048):
        from perseus import _pc as pc
        if width < 2 or width & (width-1) or 2048 % width:
            raise ValueError('Unsupported logical width')
        self.module, self.width = pc, width
        secret = pc.ClientSecret.generate()
        self._client = pc.Client(secret)
        self.spec = {'backend': 'perseus_n32_sm86', 'scheme': 'CKKS',
            'security': 'upstream_perseus_n32_parameters',
            'width': width, 'native_slots': 32768, 'bootstrap_slots': 2048,
            'ring_dimension': 65536, 'composite_degree': 2, 'input_bound': 1.0,
            'start_level': 32, 'residual_precision': 12,
            'refresh': 'two_genuine_bootstraps_with_residual_correction',
            'rotations': rotation_steps(width), 'pool': 112 if width == 2048 else 32,
            'kv_cache': False}

    def save_public_bundle(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        cfg = configuration()
        write_context(directory, cfg)
        programs = load_programs(directory, cfg)
        keys = self.module.KeySource.generate(self.spec['rotations'], layout='packed')
        manifest = self.module.Session.key_manifest(cfg, keys,
            programs=list(programs.values()), regs=1, pool=self.spec['pool'], pt_slots=4)
        self._client.keygen(manifest, str(directory/'public.pckb'), threads=4)
        (directory/'parameters.json').write_text(json.dumps(self.spec, indent=2))
        gc.collect()
        trim = getattr(ctypes.CDLL(None), 'malloc_trim', None)
        if trim is not None:
            trim.argtypes, trim.restype = [ctypes.c_size_t], ctypes.c_int
            trim(0)

    def encrypt(self, vector):
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (self.width,) or not np.isfinite(vector).all():
            raise ValueError('Invalid private logical vector')
        if np.max(np.abs(vector)) > self.spec['input_bound']:
            raise ValueError('Input exceeds the public encryption bound')
        return self._client.encrypt(np.tile(vector, 32768//self.width), self.spec['start_level'])

    def write_ciphertext(self, ciphertext, path):
        Path(path).write_bytes(ciphertext)

    def read_output(self, path):
        value = np.asarray(self._client.decrypt(Path(path).read_bytes())[0])
        return value.real[:self.width]
