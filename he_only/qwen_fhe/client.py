"""Client-only secret-key ownership and final-output decryption."""
from __future__ import annotations

import json
import os
import gc
import sys
import ctypes
from pathlib import Path
import numpy as np


class Client:
    def __init__(self, width=2048, usable_depth=12, scaling_bits=59):
        import openfhe as fhe
        self.module = fhe
        self.width = width
        self.level_budget = [3, 3]
        distribution = fhe.SecretKeyDist.UNIFORM_TERNARY
        helper = getattr(fhe, 'FHECKKSRNS', fhe.CryptoContext)
        bootstrap_depth = helper.GetBootstrapDepth(self.level_budget, distribution)
        self.depth = usable_depth+bootstrap_depth+1  # two-iteration refresh reserve
        self.start_level = bootstrap_depth
        parameters = fhe.CCParamsCKKSRNS()
        parameters.SetSecretKeyDist(distribution)
        parameters.SetSecurityLevel(fhe.SecurityLevel.HEStd_128_classic)
        parameters.SetScalingTechnique(fhe.ScalingTechnique.FLEXIBLEAUTO)
        parameters.SetScalingModSize(scaling_bits)
        if not 40 <= scaling_bits <= 59:
            raise ValueError('Unsupported CKKS scale')
        parameters.SetFirstModSize(scaling_bits+1)
        parameters.SetBatchSize(width)
        parameters.SetMultiplicativeDepth(self.depth)
        self.context = fhe.GenCryptoContext(parameters)
        for feature in ['PKE', 'KEYSWITCH', 'LEVELEDSHE', 'ADVANCEDSHE', 'FHE']:
            self.context.Enable(getattr(fhe.PKESchemeFeature, feature))
        self.context.EvalBootstrapSetup(self.level_budget, [0, 0], width)
        self._keys = self.context.KeyGen()
        self.context.EvalMultKeyGen(self._keys.secretKey)
        self.context.EvalBootstrapKeyGen(self._keys.secretKey, width)
        rotations = set()
        shift = 1
        while shift < width:
            rotations.update([shift, -shift])
            shift *= 2
        self.context.EvalRotateKeyGen(self._keys.secretKey, sorted(rotations))
        self.spec = {'scheme': 'CKKS', 'security': 'HEStd_128_classic',
                     'refresh': 'EvalBootstrap', 'width': width, 'depth': self.depth,
                     'start_level': self.start_level, 'level_budget': self.level_budget,
                     'scaling_bits': scaling_bits,
                     'ring_dimension': self.context.GetRingDimension(), 'kv_cache': False}
        self.spec.update(input_bound=1.0, bootstrap_iterations=2, bootstrap_precision=17)

    def save_public_bundle(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        module, context = self.module, self.context
        if not module.SerializeToFile(str(directory/'context.bin'), context, module.BINARY):
            raise RuntimeError('Cannot serialize public context')
        if not context.SerializeEvalMultKey(str(directory/'multiply.bin'), module.BINARY):
            raise RuntimeError('Cannot serialize multiplication keys')
        if not context.SerializeEvalAutomorphismKey(str(directory/'automorphism.bin'), module.BINARY):
            raise RuntimeError('Cannot serialize automorphism keys')
        (directory/'parameters.json').write_text(json.dumps(self.spec, indent=2))
        # Only the evaluator needs rotation/bootstrapping keys after handoff.
        # The trusted client retains its secret/public pair for encrypt/decrypt.
        context.ClearEvalAutomorphismKeys()
        if hasattr(context, 'ClearEvalMultKeys'):
            context.ClearEvalMultKeys()
        gc.collect()
        # Large C++ key allocations can remain in glibc arenas after clear.
        # Return those free pages before the separate evaluator loads its copy.
        if sys.platform.startswith('linux'):
            trim = getattr(ctypes.CDLL(None), 'malloc_trim', None)
            if trim is not None:
                trim.argtypes = [ctypes.c_size_t]
                trim.restype = ctypes.c_int
                trim(0)

    def encrypt(self, vector):
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (self.width,) or not np.isfinite(vector).all():
            raise ValueError('Invalid client input')
        if np.max(np.abs(vector)) > self.spec['input_bound']:
            raise ValueError('Client embedding exceeds the public encryption input bound')
        plaintext = self.context.MakeCKKSPackedPlaintext(vector.tolist(), 1, self.start_level,
                                                       None, self.width)
        return self.context.Encrypt(self._keys.publicKey, plaintext)

    def decrypt_output(self, ciphertext):
        plaintext = self.context.Decrypt(self._keys.secretKey, ciphertext)
        plaintext.SetLength(self.width)
        return np.asarray(plaintext.GetRealPackedValue())[:self.width]

    def write_ciphertext(self, ciphertext, path):
        if not self.module.SerializeToFile(str(path), ciphertext, self.module.BINARY):
            raise RuntimeError('Cannot serialize encrypted input')

    def read_output(self, path):
        ciphertext, success = self.module.DeserializeCiphertext(str(path), self.module.BINARY)
        if not success:
            raise ValueError('Cannot read encrypted output')
        return self.decrypt_output(ciphertext)
