"""Exact BFV linear arithmetic. The server never imports a secret key.

Fixed-point exponents travel with ciphertexts. All secret multiplications,
nonlinearities, truncations and comparisons go through two-party MPC.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import secrets
import tempfile
import time
import json
import sys

import numpy as np
import openfhe as fhe

from mpc import T, F, G
from wire import pack, unpack


@dataclass
class Value:
    cipher: object
    scale: int = F


class BFV:
    def __init__(self, directory, width):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.width = width
        if width < 2 or width & (width-1):
            raise ValueError('Width must be a power of two')
        self.context = None

    def generate(self):
        params = fhe.CCParamsBFVRNS()
        params.SetPlaintextModulus(T)
        # No ciphertext multiplication is evaluated in HE. One BFV depth is
        # sufficient for public matvecs; secret nonlinear circuits refresh in MPC.
        params.SetMultiplicativeDepth(1)
        params.SetSecurityLevel(fhe.HEStd_128_classic)
        self.context = fhe.GenCryptoContext(params)
        for feature in [fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE]:
            self.context.Enable(feature)
        self.keys = self.context.KeyGen()
        rotations = []
        shift = 1
        while shift < self.width:
            rotations += [shift, -shift]
            shift *= 2
        self.context.EvalRotateKeyGen(self.keys.secretKey, rotations)
        self.slots = self.context.GetRingDimension()
        if self.slots % self.width:
            raise ValueError('Logical width incompatible with BFV ring')
        return self

    def _temporary(self):
        return tempfile.NamedTemporaryFile(dir=str(self.directory), delete=False)

    def serialize(self, native):
        f = self._temporary(); f.close()
        try:
            if not fhe.SerializeToFile(f.name, native, fhe.BINARY):
                raise RuntimeError('HE serialization failed')
            return pack(Path(f.name).read_bytes())
        finally:
            Path(f.name).unlink()

    def deserialize(self, encoded, kind):
        f = self._temporary()
        try:
            f.write(unpack(encoded)); f.close()
            native, ok = getattr(fhe, 'Deserialize'+kind)(f.name, fhe.BINARY)
            if not ok:
                raise ValueError('Invalid HE '+kind)
            return native
        finally:
            f.close(); Path(f.name).unlink()

    def public_bundle(self):
        f = self._temporary(); f.close()
        try:
            if not self.context.SerializeEvalAutomorphismKey(f.name, fhe.BINARY):
                raise RuntimeError('Cannot serialize rotation keys')
            return dict(context=self.serialize(self.context),
                        rotations=pack(Path(f.name).read_bytes()), width=self.width,
                        public_key=self.serialize(self.keys.publicKey),
                        modulus=T, fraction_bits=F, security='HEStd_128_classic')
        finally:
            Path(f.name).unlink()

    def load_bundle(self, bundle):
        if (bundle['width'] != self.width or bundle['modulus'] != T or
                bundle['fraction_bits'] != F or bundle['security'] != 'HEStd_128_classic'):
            raise ValueError('HE parameter mismatch')
        self.context = self.deserialize(bundle['context'], 'CryptoContext')
        self.public_key = self.deserialize(bundle['public_key'], 'PublicKey')
        self.slots = self.context.GetRingDimension()
        if self.slots % self.width:
            raise ValueError('Incompatible ring dimension')
        f = self._temporary()
        try:
            f.write(unpack(bundle['rotations'])); f.close()
            if not self.context.DeserializeEvalAutomorphismKey(f.name, fhe.BINARY):
                raise ValueError('Invalid rotation keys')
        finally:
            f.close(); Path(f.name).unlink()
        return self

    def plain(self, residues):
        values = [int(v) % T for v in residues]
        if len(values) != self.width:
            raise ValueError('Invalid plaintext width')
        values = [v if v <= T//2 else v-T for v in values]
        return self.context.MakePackedPlaintext(values*(self.slots//self.width))

    def encrypt_residues(self, residues):
        key = self.keys.publicKey if hasattr(self, 'keys') else self.public_key
        return Value(self.context.Encrypt(key, self.plain(residues)))

    def encrypt(self, vector):
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (self.width,) or not np.isfinite(vector).all():
            raise ValueError('Invalid client vector')
        return self.encrypt_residues(np.rint(vector*(2**F)).astype(np.int64))

    def decrypt_residues(self, value):
        if not hasattr(self, 'keys'):
            raise RuntimeError('Only the WSL client has a secret key')
        pt = self.context.Decrypt(self.keys.secretKey, value.cipher)
        pt.SetLength(self.width)
        return [int(v) % T for v in pt.GetPackedValue()[:self.width]]

    def decrypt(self, value):
        return np.asarray([v if v <= T//2 else v-T for v in
                           self.decrypt_residues(value)], dtype=np.float64)/(2**value.scale)

    def encode(self, value):
        return {'ciphertext': self.serialize(value.cipher), 'scale': value.scale}

    def decode(self, data):
        scale = int(data['scale'])
        if not 0 <= scale <= 2*F:
            raise ValueError('Unsupported ciphertext scale')
        return Value(self.deserialize(data['ciphertext'], 'Ciphertext'), scale)

    def random_mask(self):
        return [secrets.randbelow(T) for _ in range(self.width)]


class HybridOps:
    is_encrypted = True

    def __init__(self, he, mpc, wire):
        self.he, self.mpc, self.wire = he, mpc, wire
        self.width = he.width
        self.counts = {'linear_diagonals': 0, 'linear_seconds': 0.0,
                       'mpc_calls': 0, 'bootstraps': 0}

    def zero_like(self, x):
        return self.mul_plain(x, 0)

    def sum_values(self, xs):
        xs = iter(xs); value = next(xs)
        for x in xs:
            value = self.add(value, x)
        return value

    def _scale(self, x, scale):
        if x.scale == scale:
            return x
        if scale < x.scale:
            raise ValueError('Truncation must run inside MPC')
        factor = 1 << (scale-x.scale)
        return Value(self.he.context.EvalMult(x.cipher, self.he.plain([factor]*self.width)), scale)

    def add(self, a, b):
        scale = max(a.scale, b.scale)
        a, b = self._scale(a, scale), self._scale(b, scale)
        return Value(self.he.context.EvalAdd(a.cipher, b.cipher), scale)

    def mul_plain(self, x, coefficients):
        v = np.broadcast_to(np.asarray(coefficients, dtype=np.float64), (self.width,))
        if not np.isfinite(v).all():
            raise ValueError('Nonfinite public coefficient')
        bits = 0 if np.all(v == np.rint(v)) else F
        if x.scale+bits > 2*F:
            x = self.mpc_op('identity', [x])[0]
        q = np.rint(v*(2**bits)).astype(np.int64)
        return Value(self.he.context.EvalMult(x.cipher, self.he.plain(q)), x.scale+bits)

    def add_plain(self, x, coefficient):
        v = np.broadcast_to(np.asarray(coefficient), (self.width,))
        q = np.rint(v*(2**x.scale)).astype(np.int64)
        return Value(self.he.context.EvalAdd(x.cipher, self.he.plain(q)), x.scale)

    def rotate(self, x, shift):
        if not shift:
            return x
        sign = 1 if shift > 0 else -1
        shift = abs(shift) % self.width
        c = x.cipher
        bit = 1
        while shift:
            if shift & 1:
                c = self.he.context.EvalRotate(c, sign*bit)
            shift >>= 1; bit <<= 1
        return Value(c, x.scale)

    def linear(self, xs, weight):
        """Public dense matvec, using baby-step/giant-step BFV diagonals."""
        started = time.monotonic()
        out_dim, in_dim = weight.shape
        print(json.dumps(dict(event='he_linear_start', shape=[out_dim,in_dim])), file=sys.stderr, flush=True)
        B = self.width
        if len(xs) != math.ceil(in_dim/B):
            raise ValueError('Linear shape mismatch')
        xs = [self.mpc_op('identity', [x])[0] if x.scale != F else x for x in xs]
        baby = min(32, B)
        rotations = []
        for x in xs:
            r = [x]
            for i in range(1, baby):
                r.append(self.rotate(r[-1], 1))
            rotations.append(r)
        rows = np.arange(B)
        outputs = []
        for base in range(0, out_dim, B):
            output = None
            for block, x in enumerate(xs):
                tile = np.zeros((B, B), dtype=np.float64)
                nr, nc = min(B, out_dim-base), min(B, in_dim-block*B)
                tile[:nr, :nc] = weight[base:base+nr, block*B:block*B+nc]
                for giant in range(0, B, baby):
                    accumulator = None
                    for small in range(baby):
                        diagonal = tile[rows, (rows+giant+small) % B]
                        if not np.any(diagonal):
                            continue
                        diagonal = np.roll(diagonal, giant)
                        # Always encode public model weights at F fractional bits.
                        q = np.rint(diagonal*(2**F)).astype(np.int64)
                        p = self.he.context.EvalMult(rotations[block][small].cipher,
                                                    self.he.plain(q))
                        v = Value(p, 2*F)
                        accumulator = v if accumulator is None else self.add(accumulator, v)
                        self.counts['linear_diagonals'] += 1
                    if accumulator is not None:
                        accumulator = self.rotate(accumulator, giant)
                        output = accumulator if output is None else self.add(output, accumulator)
            if output is None:
                output = Value(self.zero_like(xs[0]).cipher, 2*F)
            outputs.append(output)
        self.counts['linear_seconds'] += time.monotonic()-started
        print(json.dumps(dict(event='he_linear_done', shape=[out_dim,in_dim],
                              seconds=time.monotonic()-started)), file=sys.stderr, flush=True)
        return outputs

    def mpc_op(self, operation, values, **parameters):
        spec = dict(op=operation, width=self.width, scales=[x.scale for x in values], **parameters)
        inputs, masked = [], []
        for x in values:
            r = self.he.random_mask()
            inputs.append(r)
            masked.append(Value(self.he.context.EvalSub(x.cipher, self.he.plain(r)), x.scale))
        if operation == 'norm':
            inputs.append(np.rint(np.asarray(parameters.pop('gamma'))*(2**G)).astype(np.int64))
            spec.pop('gamma')
        output_count = len(values) if operation in ('softmax', 'argmax') else 1
        masks = [self.he.random_mask() for _ in range(output_count)]
        inputs.extend(masks)
        call = self.counts['mpc_calls']
        print(json.dumps(dict(event='mpc_start', id=call, operation=operation,
                              input_blocks=len(values))), file=sys.stderr, flush=True)
        # Compile both parties before starting either VM. Inputs stay on their host.
        self.mpc.compile(spec)
        self.wire.send(dict(type='mpc', id=call, spec=spec,
                            values=[self.he.encode(x) for x in masked], count=output_count))
        response = self.wire.recv()
        if response != {'type': 'mpc_ready', 'id': call}:
            raise RuntimeError('Client rejected MPC operation: '+str(response)[:1000])
        state = self.mpc.start(spec, inputs, call)
        self.wire.send(dict(type='mpc_start', id=call))
        self.mpc.finish(state, output_count*self.width)
        response = self.wire.recv()
        if response.get('type') != 'mpc_result' or response.get('id') != call:
            raise RuntimeError('Invalid MPC response: '+str(response)[:1000])
        if len(response['values']) != output_count:
            raise ValueError('Invalid MPC output block count')
        outputs = []
        for data, r in zip(response['values'], masks):
            encrypted_share = self.he.decode(data)
            if encrypted_share.scale != F:
                raise ValueError('MPC output scale mismatch')
            outputs.append(Value(self.he.context.EvalAdd(encrypted_share.cipher,
                                                        self.he.plain(r)), F))
        self.counts['mpc_calls'] += 1
        print(json.dumps(dict(event='mpc_done', id=call, operation=operation)), file=sys.stderr, flush=True)
        return outputs

    def mul(self, a, b):
        return self.mpc_op('mul', [a, b])[0]

    def rms_norm(self, x, gamma, epsilon, approximation=None):
        return self.mpc_op('norm', [x], group=len(gamma), gamma=gamma,
                           epsilon=epsilon)[0]

    def silu(self, x, approximation=None):
        return self.mpc_op('silu', [x])[0]

    def softmax(self, scores, **unused):
        if len(scores) == 1:
            return [self.add_plain(self.zero_like(scores[0]), 1)]
        return self.mpc_op('softmax', scores)

    def group_sum(self, x, group_size, scale=1.0):
        result = x
        shift = 1
        while shift < group_size:
            result = self.add(result, self.rotate(result, shift)); shift *= 2
        if group_size == self.width:
            return self.mul_plain(result, scale) if scale != 1 else result
        # Fuse public attention scaling into the segment-selection plaintext.
        result = self.mul_plain(result, (np.arange(self.width) % group_size == 0).astype(float)*scale)
        shift = 1
        while shift < group_size:
            result = self.add(result, self.rotate(result, -shift)); shift *= 2
        return result

    def rope(self, x, position, head_dim, theta):
        if position == 0:
            return x
        half = head_dim//2
        first = np.arange(self.width) % head_dim < half
        angle = position * theta**(-(np.arange(self.width) % half)*2.0/head_dim)
        # One plaintext multiplication per branch. Materializing a masked rotate
        # and then multiplying by sin would unnecessarily consume two BFV depths.
        return self.sum_values([
            self.mul_plain(x, np.cos(angle)),
            self.mul_plain(self.rotate(x, half), -first.astype(float)*np.sin(angle)),
            self.mul_plain(self.rotate(x, -half), (~first).astype(float)*np.sin(angle))])
