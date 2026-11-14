"""One trusted secret; separate public bundles for arithmetic and refresh GPUs."""
import ctypes
import gc
import json
import os
from pathlib import Path
from .perseus_client import PerseusClient
from .perseus_context import configuration, write_context, load_programs


class PerseusMultiClient(PerseusClient):
    def __init__(self, width=2048):
        super().__init__(width)
        self.spec['evaluation_layout'] = 'separate_arithmetic_and_bootstrap_gpus'
        self.spec['bootstrap_pool'] = 4
        self.spec['bootstrap_level_budget'] = [4, 4]
        self.spec['bootstrap_raw_landing'] = 34
        self.spec['host_ciphertext_storage'] = os.environ.get('QWEN_FHE_HOST_SPILL') == '1'
        if self.spec['host_ciphertext_storage']:
            self.spec['pool'] = 48 if width == 2048 else 24
        self.spec['spill_output_chunks'] = 8
        self.spec['temporary_ciphertext_storage'] = (
            'serialized_in_host_memory' if self.spec['host_ciphertext_storage'] else 'device')

    def save_public_bundle(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        cfg = configuration()
        (directory/'config.json').write_text(cfg.to_json())
        # No bootstrap routes or bootstrap keys on the arithmetic GPU.
        model_keys = self.module.KeySource.generate(self.spec['rotations'],
                                                   conj=False, layout='packed')
        manifest = self.module.Session.key_manifest(cfg, model_keys,
            pool=self.spec['pool'], pt_slots=4)
        self._client.keygen(manifest, str(directory/'public.pckb'), threads=4)
        bootstrap = directory/'bootstrap'
        bootstrap.mkdir()
        bootstrap_cfg = configuration(self.spec['bootstrap_level_budget'])
        write_context(bootstrap, bootstrap_cfg)
        programs = load_programs(bootstrap, bootstrap_cfg)
        refresh_keys = self.module.KeySource.generate([], layout='packed')
        manifest = self.module.Session.key_manifest(bootstrap_cfg, refresh_keys,
            programs=list(programs.values()), regs=1,
            reg_levels=[self.spec['bootstrap_raw_landing']],
            pool=self.spec['bootstrap_pool'], pt_slots=1)
        self._client.keygen(manifest, str(bootstrap/'public.pckb'), threads=4)
        for folder in [directory, bootstrap]:
            (folder/'parameters.json').write_text(json.dumps(self.spec, indent=2))
        gc.collect()
        trim = getattr(ctypes.CDLL(None), 'malloc_trim', None)
        if trim is not None:
            trim.argtypes, trim.restype = [ctypes.c_size_t], ctypes.c_int
            trim(0)
