"""CPU CKKS baseline: public evaluation context, real EvalBootstrap, no secret key.

The service loads only public parameters/evaluation keys. Encryption and decryption
live in client.py, which the service never imports. Ubuntu 20.04 uses OpenFHE 1.4.1.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

from .ops import VectorCircuits


class OpenFHEOps(VectorCircuits):
    is_encrypted = True

    def __init__(self, bundle):
        import openfhe as fhe
        self.module = fhe
        bundle = Path(bundle)
        spec = json.loads((bundle/'parameters.json').read_text())
        if spec.get('security') != 'HEStd_128_classic' or spec.get('refresh') != 'EvalBootstrap':
            raise ValueError('Unapproved HE parameter bundle')
        super().__init__(int(spec['width']))
        self.depth = int(spec['depth'])
        self.spec = spec
        # Public arithmetic bounds only; no slot inspection. IDs are registered
        # for every returned native object, including after deserialization.
        self._bounds = {}
        self.context, success = fhe.DeserializeCryptoContext(str(bundle/'context.bin'), fhe.BINARY)
        if not success:
            raise ValueError('Cannot deserialize CKKS public context')
        for method, name in [('DeserializeEvalMultKey', 'multiply.bin'),
                             ('DeserializeEvalAutomorphismKey', 'automorphism.bin')]:
            if not getattr(self.context, method)(str(bundle/name), fhe.BINARY):
                raise ValueError('Cannot load CKKS evaluation keys')
        self.context.EvalBootstrapSetup(spec['level_budget'], [0, 0], self.width)

    def _ready(self, value, required=4):
        if isinstance(value, np.ndarray) or not hasattr(value, 'GetLevel'):
            raise TypeError('Server arithmetic accepts native CKKS ciphertexts only')
        remaining = self.depth-value.GetLevel()-(value.GetNoiseScaleDeg()-1)
        if remaining < required:
            started = time.monotonic()
            trace = os.environ.get('QWEN_HE_TRACE') == '1'
            if trace:
                print(json.dumps({'stage': 'bootstrap_start', 'index': self.counts['bootstraps']}), file=sys.stderr, flush=True)
            bound = self._get_bound(value)
            factor = 2.0**max(0, math.ceil(math.log2(max(1.0, bound))))
            normalized = self.context.EvalMult(value, 1.0/factor) if factor > 1 else value
            # Single-round precision measured on this host is at least 17 bits.
            # Two-iteration refinement protects small activations and large
            # inverse-norm intermediates after restoring their public scale.
            value = self.context.EvalBootstrap(normalized, 2, 17)
            if factor > 1:
                value = self.context.EvalMult(value, factor)
            self._tag(value, bound)
            self.counts['bootstraps'] += 1
            if trace:
                print(json.dumps({'stage': 'bootstrap_done', 'seconds': time.monotonic()-started}), file=sys.stderr, flush=True)
            if self.depth-value.GetLevel()-(value.GetNoiseScaleDeg()-1) < required:
                raise RuntimeError('Real bootstrapping did not restore enough depth')
        return value

    def _get_bound(self, value):
        try:
            return self._bounds[id(value)]
        except KeyError:
            raise TypeError('Unregistered native ciphertext') from None

    def _tag(self, value, bound):
        if not math.isfinite(bound) or bound < 0:
            raise ValueError('Invalid public ciphertext range')
        self._bounds[id(value)] = float(bound)
        return value

    def range_hint(self, value, lo, hi):
        if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
            raise ValueError('Invalid public interval')
        return self._tag(value, max(abs(lo), abs(hi)))

    def linear_output_range(self, value, xs, weight):
        limits = np.concatenate([np.full(self.width, self._get_bound(x)) for x in xs])
        bound = float(np.max(np.abs(weight) @ limits[:weight.shape[1]]))
        return self._tag(value, bound)

    def add(self, a, b):
        return self._tag(self.context.EvalAdd(a, b), self._get_bound(a)+self._get_bound(b))

    def prepare_value(self, x):
        return self._ready(x)

    def linear_progress(self):
        if os.environ.get('QWEN_HE_TRACE') == '1':
            print(json.dumps({'stage': 'linear_progress', 'counts': self.counts,
                              'time_unix': time.time()}), file=sys.stderr, flush=True)

    def sub(self, a, b):
        return self._tag(self.context.EvalSub(a, b), self._get_bound(a)+self._get_bound(b))

    def mul(self, a, b):
        bound = self._get_bound(a)*self._get_bound(b)
        ready_a = self._ready(a)
        ready_b = ready_a if a is b else self._ready(b)
        return self._tag(self.context.EvalMult(ready_a, ready_b), bound)

    def zero_like(self, x):
        return self._tag(self.context.EvalSub(x, x), 0.0)

    def _plain(self, vector, value):
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (self.width,) or not np.isfinite(vector).all():
            raise ValueError('Invalid public SIMD operand')
        return self.context.MakeCKKSPackedPlaintext(vector.tolist(), 1, value.GetLevel(), None, self.width)

    def mul_plain(self, a, b):
        a = self._ready(a)
        if np.isscalar(b):
            if not np.isfinite(b):
                raise ValueError('Invalid public scalar')
            return self._tag(self.context.EvalMult(a, float(b)), self._get_bound(a)*abs(float(b)))
        return self._tag(self.context.EvalMult(a, self._plain(b, a)),
                         self._get_bound(a)*float(np.max(np.abs(b))))

    def add_plain(self, a, b):
        if np.isscalar(b):
            return self._tag(self.context.EvalAdd(a, float(b)), self._get_bound(a)+abs(float(b)))
        return self._tag(self.context.EvalAdd(a, self._plain(b, a)),
                         self._get_bound(a)+float(np.max(np.abs(b))))

    def rotate(self, a, shift):
        return self._tag(self.context.EvalRotate(a, int(shift)), self._get_bound(a))

    def chebyshev(self, x, coefficients, lo, hi):
        # Use the backend's native polynomial circuit. OpenFHE's convention is
        # c0/2 + sum(c_i*T_i), whereas numpy and our profiles store c0 directly.
        degree = len(coefficients)-1
        if degree < 2:
            return super().chebyshev(x, coefficients, lo, hi)
        x = self._ready(x, int(math.ceil(math.log2(degree+1)))+2)
        native = [float(c) for c in coefficients]
        native[0] *= 2
        return self._tag(self.context.EvalChebyshevSeries(x, native, float(lo), float(hi)),
                         float(np.sum(np.abs(coefficients))))

    def import_ciphertext(self, path):
        result, success = self.module.DeserializeCiphertext(str(path), self.module.BINARY)
        if not success:
            raise ValueError('Cannot deserialize ciphertext')
        return self._tag(result, float(self.spec.get('input_bound', 1.0)))

    def export_ciphertext(self, value, path):
        if not self.module.SerializeToFile(str(path), value, self.module.BINARY):
            raise RuntimeError('Cannot serialize ciphertext')
