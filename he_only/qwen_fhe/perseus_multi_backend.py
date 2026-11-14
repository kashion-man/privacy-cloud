"""Arithmetic on one GPU, public-key planned refresh on a second GPU."""
import atexit
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid
import numpy as np
from .ops import VectorCircuits
from .perseus_backend import PerseusOps
from .cipher_rpc import send_metadata, receive_metadata, send_frame, receive_frame


class StoredCiphertext:
    """Serialized encrypted RNS data and a public magnitude bound, never slots."""
    __slots__ = ('frame', 'bound')

    def __init__(self, frame, bound):
        self.frame, self.bound = frame, bound


class PerseusMultiOps(PerseusOps):
    def __init__(self, bundle):
        from perseus import _pc as pc
        bundle = Path(bundle).resolve()
        self.module = pc
        self.spec = json.loads((bundle/'parameters.json').read_text())
        if (self.spec.get('backend') != 'perseus_n32_sm86'
            or self.spec.get('evaluation_layout') != 'separate_arithmetic_and_bootstrap_gpus'):
            raise ValueError('Expected a split public bundle')
        VectorCircuits.__init__(self, int(self.spec['width']))
        if self.width > 2048 or 2048 % self.width:
            raise ValueError('Unsupported logical ciphertext width')
        self._bounds = {}
        self.cfg = pc.Config.from_json((bundle/'config.json').read_text())
        keys = pc.KeySource.generate(self.spec['rotations'], conj=False, layout='packed')
        self.session = pc.Session.from_bundle(self.cfg, keys, str(bundle/'public.pckb'),
                                              pool=int(self.spec['pool']), pt_slots=4)
        self.session.complete_setup()
        self.fhe = self.session.fhe
        if self.session.has_secret or self.fhe.has_secret_key or self.fhe.debug_refresh:
            raise RuntimeError('Arithmetic evaluator must have no secret or debug refresh')
        model_gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
        bootstrap_gpu = os.environ.get('QWEN_FHE_BOOTSTRAP_GPU')
        if not bootstrap_gpu or bootstrap_gpu == model_gpu:
            raise ValueError('Provide a distinct QWEN_FHE_BOOTSTRAP_GPU')
        self._socket_path = Path('/tmp/qwen-fhe-'+uuid.uuid4().hex+'.sock')
        self._connection = None
        self._log = (bundle.parent/'bootstrap_evaluator.log').open('a')
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = bootstrap_gpu
        self._service = subprocess.Popen([sys.executable, '-u', '-m',
            'qwen_fhe.bootstrap_service', '--bundle', str(bundle/'bootstrap'),
            '--socket', str(self._socket_path)], env=env, stdin=subprocess.DEVNULL,
            stdout=self._log, stderr=subprocess.STDOUT)
        atexit.register(self.close)
        deadline = time.monotonic()+300
        while not self._socket_path.exists():
            if self._service.poll() is not None:
                raise RuntimeError('Refresh GPU setup failed; see bootstrap_evaluator.log')
            if time.monotonic() > deadline:
                raise RuntimeError('Refresh GPU setup timed out')
            time.sleep(.2)
        self._connection = socket.socket(socket.AF_UNIX)
        self._connection.settimeout(600)
        self._connection.connect(str(self._socket_path))
        send_metadata(self._connection, {'op': 'status'})
        self.service_status = receive_metadata(self._connection)
        if (self.service_status.get('server_has_secret') is not False
            or self.service_status.get('debug_refresh') is not False
            or self.service_status.get('genuine_planned_bootstraps') != [1, 1]):
            raise RuntimeError('Invalid public refresh service')
        print(json.dumps({'stage': 'split_gpu_evaluator_ready',
            'arithmetic_gpu': model_gpu, 'bootstrap_gpu': bootstrap_gpu,
            'server_has_secret': False, 'debug_refresh': False,
            'arithmetic_memory': str(self.session.mem_report)}), flush=True)

    def _get_bound(self, value):
        return value.bound if isinstance(value, StoredCiphertext) else super()._get_bound(value)

    def spill_ciphertext(self, value):
        if not self.spec.get('host_ciphertext_storage', False):
            return value
        if isinstance(value, StoredCiphertext):
            return value
        return StoredCiphertext(self.module.export_ct(self.session, value), self._get_bound(value))

    def restore_ciphertext(self, value):
        if not isinstance(value, StoredCiphertext):
            return value
        result = self.module.import_ct(self.session, value.frame)
        return self._tag(self._tag_layout(result), value.bound)

    def export_ciphertext(self, value, path):
        if isinstance(value, StoredCiphertext):
            Path(path).write_bytes(value.frame)
        else:
            super().export_ciphertext(value, path)

    def linear(self, xs, weight):
        """Same direct diagonal circuit, with one host-stored source loaded at a time."""
        if not self.spec.get('host_ciphertext_storage', False):
            return super().linear(xs, weight)
        if not isinstance(xs, list):
            raise TypeError('Linear requires an owned list')
        out_dim, in_dim = weight.shape
        size = self.width
        if len(xs) != math.ceil(in_dim/size):
            raise ValueError('Linear input shape mismatch')
        for i in range(len(xs)):
            if isinstance(xs[i], StoredCiphertext):
                native = self.prepare_value(self.restore_ciphertext(xs[i]))
                xs[i] = self.spill_ciphertext(native)
                del native
            else:
                xs[i] = self.prepare_value(xs[i])
        row = np.arange(size)
        outputs = []
        store_outputs = math.ceil(out_dim/size) >= int(self.spec['spill_output_chunks'])
        for out_base in range(0, out_dim, size):
            acc = None
            nout = min(size, out_dim-out_base)
            for block, source in enumerate(xs):
                x = self.restore_ciphertext(source)
                nin = min(size, in_dim-block*size)
                tile = np.zeros((size, size), dtype=np.float64)
                tile[:nout, :nin] = weight[out_base:out_base+nout, block*size:block*size+nin]
                current = x
                for shift in range(size):
                    diagonal = tile[row, (row+shift) % size]
                    if np.any(diagonal):
                        product = self.mul_plain(current, diagonal)
                        acc = product if acc is None else self.add(acc, product)
                        self.counts['linear_diagonals'] += 1
                        if self.counts['linear_diagonals'] % 256 == 0:
                            self.linear_progress()
                    if shift+1 < size:
                        current = self.rotate(current, 1)
                del current, x
            if acc is None:
                native = self.restore_ciphertext(xs[0])
                acc = self.zero_like(native)
                del native
            acc = self.linear_output_range(acc, xs, weight[out_base:out_base+nout])
            outputs.append(self.spill_ciphertext(acc) if store_outputs else acc)
        return outputs

    def _bootstrap(self, value, level):
        self.fhe.realize(value)
        if value.level > level:
            raise RuntimeError('Insufficient levels for cross-GPU refresh')
        self.fhe.drop_to_level(value, level)
        self._tag_layout(value)
        send_metadata(self._connection, {'op': 'bootstrap', 'level': level})
        send_frame(self._connection, self.module.export_ct(self.session, value))
        result = self.module.import_ct(self.session, receive_frame(self._connection))
        self.counts['bootstraps'] += 1
        return self._tag_layout(result)

    def close(self):
        connection = getattr(self, '_connection', None)
        if connection is not None:
            try:
                send_metadata(connection, {'op': 'shutdown'})
                receive_metadata(connection)
            except (OSError, EOFError):
                pass
            connection.close()
            self._connection = None
        service = getattr(self, '_service', None)
        if service is not None and service.poll() is None:
            try:
                service.wait(timeout=10)
            except subprocess.TimeoutExpired:
                service.terminate()
                service.wait(timeout=10)
        log = getattr(self, '_log', None)
        if log is not None:
            log.close()
        address = getattr(self, '_socket_path', None)
        if address is not None:
            address.unlink(missing_ok=True)
