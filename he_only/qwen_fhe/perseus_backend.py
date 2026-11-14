"""Sequential native GPU evaluator. Only public keys and ciphertexts are loaded.

Sparse Classic's native iteration count repeats a refresh; it does not refine
its error. We implement the residual correction explicitly with two genuine
single-refresh programs. No secret, decrypt/re-encrypt path, or KV state exists.
"""
import json
import math
import time
from pathlib import Path
import numpy as np

from .ops import VectorCircuits
from .openfhe_backend import OpenFHEOps
from .perseus_context import load_programs


class PerseusOps(OpenFHEOps):
    """Reuse only the public-bound helpers; all crypto primitives are overridden."""
    is_encrypted = True

    def __init__(self, bundle):
        from perseus import _pc as pc
        self.module = pc
        bundle = Path(bundle)
        self.spec = json.loads((bundle/'parameters.json').read_text())
        if self.spec.get('backend') != 'perseus_n32_sm86':
            raise ValueError('Expected a native Perseus public bundle')
        VectorCircuits.__init__(self, int(self.spec['width']))
        if 2048 % self.width or self.width > 2048:
            raise ValueError('Logical vectors must repeat within 2048 slots')
        self._bounds = {}
        self.cfg = pc.Config.from_json((bundle/'config.json').read_text())
        self.programs = load_programs(bundle, self.cfg)
        keys = pc.KeySource.generate(self.spec['rotations'], layout='packed')
        self.session = pc.Session.from_bundle(self.cfg, keys, str(bundle/'public.pckb'),
            programs=list(self.programs.values()), regs=1,
            pool=int(self.spec['pool']), pt_slots=4)
        for program in self.programs.values():
            program.bind(self.session)
        self.session.complete_setup()
        self.fhe = self.session.fhe
        if self.session.has_secret or self.fhe.has_secret_key or self.fhe.debug_refresh:
            raise RuntimeError('Evaluator must hold no secret or debug refresh service')

    def _tag_layout(self, value):
        value.tag = self.module.Tag(slots=32768, period=2048,
                                    field=self.module.tag.F_REAL)
        return value

    def _bootstrap(self, value, level):
        self.fhe.realize(value)
        if value.level > level:
            raise RuntimeError('Insufficient levels for the public refresh envelope')
        self.fhe.drop_to_level(value, level)
        self._tag_layout(value)
        metrics = self.programs[level].run([value], outputs=[(0, 0)])
        if metrics['meta_mismatches']:
            raise RuntimeError('Planned refresh metadata mismatch: %r' % dict(metrics))
        result = self.module.Program.reg_ct(self.session, 0)
        self.counts['bootstraps'] += 1
        return self._tag_layout(result)

    def _ready(self, value, required=4):
        if isinstance(value, np.ndarray) or not hasattr(value, 'noise_deg'):
            raise TypeError('Native evaluator accepts ciphertexts only')
        remaining = (48-int(value.level)-2*(int(value.noise_deg)-1))//2
        if remaining >= required:
            return value
        bound = self._get_bound(value)
        power = max(0, int(math.ceil(math.log2(max(1.0, bound)))))
        factor = 2.0**power
        original = self.fhe.add(value, 0.0)
        if power:
            original = self.fhe.mult(original, 1.0/factor)
        # A raw modulus drop preserves the old scale. FLEXIBLEAUTO's cross-level
        # addition requires its canonical scale, so pay one public scalar level
        # before retaining the source used by the residual subtraction.
        self.fhe.realize(original)
        if original.level > 44:
            raise RuntimeError('Cannot enter scale-aligned refresh without reserved levels')
        self.fhe.drop_to_level(original, 44)
        original = self.fhe.canonicalize_scale(original)
        self.fhe.realize(original)
        if original.level != 46:
            raise RuntimeError('Unexpected scale-alignment landing')
        first = self._bootstrap(original, 46)
        precision = int(self.spec.get('residual_precision', 12))
        residual = self.fhe.sub(original, first)
        residual = self.fhe.mult(residual, 2.0**precision)
        correction = self._bootstrap(residual, 48)
        correction = self.fhe.mult(correction, 2.0**(-precision))
        result = self.fhe.add(first, correction)
        # Integer doubling restores the public magnitude without spending levels.
        for _ in range(power):
            result = self.fhe.add(result, result)
        result = self._tag(result, bound)
        remaining = (48-int(result.level)-2*(int(result.noise_deg)-1))//2
        if remaining < required:
            raise RuntimeError('Native residual refresh did not restore enough levels')
        return result

    def add(self, a, b):
        return self._tag(self.fhe.add(a, b), self._get_bound(a)+self._get_bound(b))

    def sub(self, a, b):
        return self._tag(self.fhe.sub(a, b), self._get_bound(a)+self._get_bound(b))

    def mul(self, a, b):
        bound = self._get_bound(a)*self._get_bound(b)
        first = self._ready(a)
        second = first if a is b else self._ready(b)
        return self._tag(self.fhe.mult(first, second), bound)

    def zero_like(self, x):
        return self._tag(self.fhe.sub(x, x), 0.0)

    def _vector(self, value):
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (self.width,) or not np.isfinite(value).all():
            raise ValueError('Invalid public logical vector')
        return np.tile(value, 32768//self.width)

    def mul_plain(self, a, b):
        a = self._ready(a)
        if np.isscalar(b):
            if not math.isfinite(b):
                raise ValueError('Invalid public scalar')
            value = self.fhe.mult(a, float(b))
            bound = abs(float(b))
        else:
            vector = self._vector(b)
            value = self.fhe.mult_pt(a, vector)
            bound = float(np.max(np.abs(vector)))
        return self._tag(value, self._get_bound(a)*bound)

    def add_plain(self, a, b):
        if np.isscalar(b):
            if not math.isfinite(b):
                raise ValueError('Invalid public scalar')
            value, bound = self.fhe.add(a, float(b)), abs(float(b))
        else:
            vector = self._vector(b)
            value, bound = self.fhe.add_pt(a, vector), float(np.max(np.abs(vector)))
        return self._tag(value, self._get_bound(a)+bound)

    def rotate(self, a, shift):
        return self._tag(self.fhe.rotate(a, int(shift)), self._get_bound(a))

    def linear(self, xs, weight):
        if not isinstance(xs, list):
            raise TypeError('Native linear requires an owned mutable input list')
        # Replacing equivalent ciphertext handles avoids holding two complete
        # vocabulary vectors during encrypted embedding feedback.
        for i in range(len(xs)):
            xs[i] = self.prepare_value(xs[i])
        return VectorCircuits.linear(self, xs, weight)

    def chebyshev(self, x, coefficients, lo, hi):
        z = self.add_plain(self.mul_plain(x, 2.0/(hi-lo)), -(hi+lo)/(hi-lo))
        z = self.range_hint(z, -1, 1)
        previous = self.add_plain(self.zero_like(z), 1.0)
        current = z
        result = self.mul_plain(previous, float(coefficients[0]))
        if len(coefficients) > 1:
            result = self.add(result, self.mul_plain(current, float(coefficients[1])))
        for coefficient in coefficients[2:]:
            z, current = self.prepare_value(z), self.prepare_value(current)
            following = self.sub(self.mul_plain(self.mul(z, current), 2.0), previous)
            following = self.range_hint(following, -1.00001, 1.00001)
            result = self.add(result, self.mul_plain(following, float(coefficient)))
            previous, current = current, following
        return self._tag(result, float(np.sum(np.abs(coefficients))))

    def import_ciphertext(self, path):
        value = self.module.import_ct(self.session, Path(path).read_bytes())
        return self._tag(self._tag_layout(value), float(self.spec['input_bound']))

    def export_ciphertext(self, value, path):
        Path(path).write_bytes(self.module.export_ct(self.session, value))
