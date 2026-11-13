"""Plain evaluator for arithmetic validation, and reusable CKKS vector circuits.

NumpyReference is explicitly not encryption. The ciphertext evaluator subclasses
VectorCircuits and implements its five arithmetic primitives without decryption.
"""
from __future__ import annotations

import math
import numpy as np


class VectorCircuits:
    is_encrypted = False

    def __init__(self, width):
        if width < 2 or width & (width - 1):
            raise ValueError('SIMD width must be a power of two')
        self.width = width
        self.counts = {'linear_diagonals': 0, 'bootstraps': 0}

    def zero_like(self, x):
        return self.mul_plain(x, 0.0)

    def prepare_value(self, x):
        return x

    def prepare_linear(self, x):
        return self.prepare_value(x)

    def linear_progress(self):
        pass

    def range_hint(self, value, lo, hi):
        """Public circuit precondition, never a ciphertext-dependent estimate."""
        return value

    def linear_output_range(self, value, xs, weight):
        return value

    def sum_values(self, values):
        values = iter(values)
        result = next(values)
        for value in values:
            result = self.add(result, value)
        return result

    def linear(self, xs, weight):
        """Plain weights, encrypted inputs: direct cyclic diagonals, no BSGS/cache."""
        out_dim, in_dim = weight.shape
        B = self.width
        if len(xs) != math.ceil(in_dim / B):
            raise ValueError('Linear input shape mismatch')
        outputs = []
        # Retain a refreshed input. Refreshing a temporary inside mul_plain would
        # otherwise bootstrap the same exhausted source once per diagonal.
        xs = [self.prepare_linear(x) for x in xs]
        row = np.arange(B)
        for out_base in range(0, out_dim, B):
            acc = None
            nout = min(B, out_dim - out_base)
            for block, x in enumerate(xs):
                nin = min(B, in_dim - block * B)
                tile = np.zeros((B, B), dtype=np.float64)
                tile[:nout, :nin] = weight[out_base:out_base+nout, block*B:block*B+nin]
                current = x
                for shift in range(B):
                    diagonal = tile[row, (row + shift) % B]
                    if np.any(diagonal):
                        product = self.mul_plain(current, diagonal)
                        acc = product if acc is None else self.add(acc, product)
                        self.counts['linear_diagonals'] += 1
                        if self.counts['linear_diagonals'] % 256 == 0:
                            self.linear_progress()
                    if shift + 1 < B:
                        current = self.rotate(current, 1)
            acc = self.zero_like(xs[0]) if acc is None else acc
            outputs.append(self.linear_output_range(acc, xs, weight[out_base:out_base+nout]))
        return outputs

    def group_sum(self, x, group_size):
        """Broadcast each head's feature sum within that head's public SIMD segment."""
        B = self.width
        if group_size & (group_size-1) or B % group_size:
            raise ValueError('Invalid head width')
        result = x
        shift = 1
        while shift < group_size:
            result = self.add(result, self.rotate(result, shift))
            shift *= 2
        if group_size == B:
            return result
        mask = (np.arange(B) % group_size == 0).astype(np.float64)
        result = self.mul_plain(result, mask)
        shift = 1
        while shift < group_size:
            result = self.add(result, self.rotate(result, -shift))
            shift *= 2
        return result

    def chebyshev(self, x, coefficients, lo, hi):
        if not 0 <= len(coefficients) - 1 <= 255 or not lo < hi:
            raise ValueError('Invalid polynomial')
        z = self.add_plain(self.mul_plain(x, 2.0/(hi-lo)), -(hi+lo)/(hi-lo))
        b1 = self.zero_like(x)
        b2 = self.zero_like(x)
        for coefficient in coefficients[:0:-1]:
            b0 = self.add_plain(self.sub(self.mul_plain(self.mul(z, b1), 2.0), b2), float(coefficient))
            b2, b1 = b1, b0
        return self.add_plain(self.sub(self.mul(z, b1), b2), float(coefficients[0]))

    def rms_norm(self, x, gamma, epsilon, approximation):
        group = len(gamma)
        input_limit = math.sqrt(group*approximation['hi'])
        x = self.range_hint(x, -input_limit, input_limit)
        variance = self.mul_plain(self.group_sum(self.mul(x, x), group), 1.0/group)
        variance = self.add_plain(variance, epsilon)
        variance = self.range_hint(variance, approximation['lo'], approximation['hi'])
        if approximation.get('method') == 'newton':
            inverse = self.add_plain(self.zero_like(variance), approximation['initial'])
        else:
            inverse = self.chebyshev(variance, approximation['coefficients'],
                                     approximation['lo'], approximation['hi'])
        cap = 1.01/math.sqrt(approximation['lo'])
        initial_cap = approximation['initial'] if approximation.get('method') == 'newton' else cap
        inverse = self.range_hint(inverse, 0, initial_cap)
        for step in range(approximation.get('newton_steps', 1)):
            # Both branches must retain the same refreshed inverse rather than
            # independently refreshing the exhausted source a second time.
            inverse = self.prepare_value(inverse)
            term = self.mul(variance, self.mul(inverse, inverse))
            term = self.range_hint(term, 0, 1.1)
            inverse = self.mul_plain(self.mul(inverse, self.sub(self.add_plain(self.zero_like(term), 3.0), term)), 0.5)
            inverse = self.range_hint(inverse, 0, min(cap, initial_cap*1.5**(step+1)))
        scales = np.tile(np.asarray(gamma), self.width // group)
        result = self.mul_plain(self.mul(x, inverse), scales)
        limit = 1.01*math.sqrt(group)*float(np.max(np.abs(gamma)))
        return self.range_hint(result, -limit, limit)

    def rope(self, x, position, head_dim, theta):
        half = head_dim // 2
        local = np.arange(self.width) % head_dim
        first = local < half
        rotated = self.add(self.mul_plain(self.rotate(x, half), first.astype(float) * -1),
                           self.mul_plain(self.rotate(x, -half), (~first).astype(float)))
        angle = position * theta ** (-(np.arange(self.width) % half) * 2.0/head_dim)
        return self.add(self.mul_plain(x, np.cos(angle)), self.mul_plain(rotated, np.sin(angle)))

    def softmax(self, scores, rounds=10, inverse_steps=9):
        """Repeated square-and-normalize approximation using CKKS arithmetic only.

        Its public input band is (-2**rounds, 2**rounds), with accuracy checked
        separately on calibrated Q/K scores. There is no plaintext max operation.
        """
        if len(scores) == 1:
            return [self.add_plain(self.zero_like(scores[0]), 1.0)]
        scores = [self.range_hint(x, -2.0**(rounds-1), 2.0**(rounds-1)) for x in scores]
        values = [self.add_plain(self.mul_plain(x, 2.0**(-rounds)), 1.0) for x in scores]
        for stage in range(rounds+1):
            if stage:
                values = [self.mul(x, x) for x in values]
            denominator = self.sum_values(values)
            denominator = self.prepare_value(denominator)
            seed = 1.0/len(values) if stage == 0 else 1.0
            inverse = self.add_plain(self.zero_like(denominator), seed)
            for _ in range(inverse_steps):
                inverse = self.prepare_value(inverse)
                inverse = self.mul(inverse, self.sub(self.add_plain(self.zero_like(inverse), 2.0), self.mul(denominator, inverse)))
                inverse = self.range_hint(inverse, 0, 2.01/len(values) if stage == 0 else 1.01*len(values))
            values = [self.range_hint(self.mul(x, inverse), 0, 1.00001) for x in values]
        return values

    def silu(self, x, approximation):
        if approximation.get('method') != 'tanh_double_angle':
            return self.chebyshev(x, approximation['coefficients'],
                                  approximation['lo'], approximation['hi'])
        rounds = int(approximation['doublings'])
        span = 2.0**(rounds-2)
        x = self.range_hint(x, approximation.get('lo', -span), approximation.get('hi', span))
        # SiLU(x) = x*(1+tanh(x/2))/2. Start in [-1/8,1/8],
        # then use tanh(2z)=2*tanh(z)/(1+tanh(z)^2), all encrypted.
        z = self.mul_plain(x, 2.0**(-rounds-1))
        z2 = self.mul(z, z)
        p = self.add_plain(self.mul_plain(z2, -17.0/315), 2.0/15)
        p = self.add_plain(self.mul(z2, p), -1.0/3)
        p = self.add_plain(self.mul(z2, p), 1.0)
        y = self.mul(z, p)
        for _ in range(rounds):
            y = self.prepare_value(y)
            denominator = self.add_plain(self.mul(y, y), 1.0)
            inverse = self.add_plain(self.zero_like(denominator), .5)
            for _ in range(5):
                inverse = self.prepare_value(inverse)
                inverse = self.mul(inverse, self.sub(self.add_plain(self.zero_like(inverse), 2.0),
                                                     self.mul(denominator, inverse)))
                inverse = self.range_hint(inverse, .49, 1.00001)
            y = self.mul_plain(self.mul(y, inverse), 2.0)
            y = self.range_hint(y, -1.00001, 1.00001)
        return self.mul(x, self.mul_plain(self.add_plain(y, 1.0), .5))


class NumpyReference(VectorCircuits):
    """Plaintext arithmetic mirror for tests. Never accepted by the HE server."""
    def add(self, a, b): return a+b
    def sub(self, a, b): return a-b
    def mul(self, a, b): return a*b
    def mul_plain(self, a, b): return a*b
    def add_plain(self, a, b): return a+b
    def rotate(self, a, shift): return np.roll(a, -shift)

    def range_hint(self, value, lo, hi):
        slack = 1e-7*max(1.0, abs(lo), abs(hi))
        if not np.isfinite(value).all() or np.any(value < lo-slack) or np.any(value > hi+slack):
            raise ValueError('Public circuit range violated: observed [%g,%g], expected [%g,%g]'
                             % (np.min(value), np.max(value), lo, hi))
        return value


class DenseNumpyReference(NumpyReference):
    """Fast plaintext oracle; server rejects it just like NumpyReference."""
    def linear(self, xs, weight):
        x = np.concatenate(xs)[:weight.shape[1]]
        result = np.asarray(weight, dtype=np.float64) @ x
        padded = np.pad(result, (0, (-len(result)) % self.width))
        return list(padded.reshape(-1, self.width))


def fit_chebyshev(function, lo, hi, degree=63, **extra):
    if not 0 < degree <= 255 or not lo < hi:
        raise ValueError('Invalid calibration interval')
    nodes = np.cos(np.pi*(np.arange(degree+1)+0.5)/(degree+1))
    values = function((nodes+1)*(hi-lo)/2+lo)
    coefficients = np.polynomial.chebyshev.chebfit(nodes, values, degree)
    return {'lo': float(lo), 'hi': float(hi), 'coefficients': coefficients.tolist(), **extra}
