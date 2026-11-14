"""Real semi-honest two-party OT MPC, using the official MP-SPDZ VM.

Private shares are transient mode-0600 local files. Neither party receives the
other party's private input file. MPC reveals only a fresh masked share to P1.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import ctypes
import signal

T = 35184372744193
F = 14  # HE range > 65536 after a scale-28 matvec; Qwen has ~11840 outliers.
G = 20  # MPC uses extra fractional bits for normalization and exponentials.


def protocol_fingerprint():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ['mpc.py','mpc_circuit.py.in','he.py','qwen.py','weights.py','wire.py']:
        digest.update(name.encode()+b'\x00'+(root/name).read_bytes())
    return digest.hexdigest()


def _child_setup():
    os.umask(0o077)
    # A failed SSH/client process must not leave an MPC VM waiting indefinitely.
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)


class MPC:
    def __init__(self, root, party, port, run_directory):
        self.root = Path(root).resolve()
        self.party = party
        self.port = port
        self.run_directory = Path(run_directory).resolve()
        self.run_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.hosts = self.run_directory/'hosts'
        self.hosts.write_text('127.0.0.1:%d\n127.0.0.1:%d\n' % (port, port+1))
        self.calls = 0
        self.seconds = 0.0

    def certificate(self):
        data = self.root/'Player-Data'
        data.mkdir(exist_ok=True)
        key, cert = data/('P%d.key' % self.party), data/('P%d.pem' % self.party)
        if not key.exists():
            subprocess.run(['openssl', 'req', '-newkey', 'rsa:2048', '-nodes', '-x509',
                            '-days', '30', '-keyout', str(key), '-out', str(cert),
                            '-subj', '/CN=P%d' % self.party],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            key.chmod(0o600)
        return cert.read_text()

    def trust_peer(self, certificate):
        if len(certificate) > 8192 or not certificate.startswith('-----BEGIN CERTIFICATE-----'):
            raise ValueError('Invalid public peer certificate')
        (self.root/'Player-Data'/('P%d.pem' % (1-self.party))).write_text(certificate)
        subprocess.run(['openssl', 'rehash', str(self.root/'Player-Data')], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def compile(self, spec):
        canonical = json.dumps(spec, sort_keys=True, separators=(',', ':'))
        template = Path(__file__).with_name('mpc_circuit.py.in').read_text()
        tag = hashlib.sha256(('field192-int64-memorder-v4:%d:%d:%d:' % (T,F,G)+canonical+template).encode()).hexdigest()[:24]
        name = 'qwen-hempc-' + tag
        if not (self.root/'Programs/Schedules'/ (name+'.sch')).exists():
            code = ('OP=%r\nN=%d\nGROUP=%d\nSCALES=%r\nEPS=%r\nVOCAB=%d\nT=%d\nF=%d\nOUT_F=%d\n' %
                    (spec['op'], spec['width'], spec.get('group', spec['width']),
                     spec['scales'], spec.get('epsilon', 1e-6),
                     spec.get('vocab', spec['width']), T, G, F)) + template
            source = self.root/'Programs/Source'/(name+'.mpc')
            source.write_text(code)
            # -F is usable INTEGER width, not the VM prime field size.
            # The VM uses a 192-bit prime for statistical masking headroom.
            result = subprocess.run([sys.executable, 'compile.py', '-M', '-F', '64', name],
                                    cwd=str(self.root), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            (self.run_directory/(name+'.compile.log')).write_text(result.stdout)
            if result.returncode:
                raise RuntimeError('MPC compile failed: '+result.stdout[-5000:])
        return name

    def start(self, spec, inputs, call_id):
        name = self.compile(spec)
        prefix = self.run_directory/('input-%06d' % call_id)
        private = Path(str(prefix)+'-P%d-0' % self.party)
        fd = os.open(str(private), os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'w') as f:
            for vector in inputs:
                f.write(' '.join(str(int(x)) for x in vector)+'\n')
        out = self.run_directory/('output-%06d' % call_id)
        log = self.run_directory/('mpc-%06d.log' % call_id)
        handle = log.open('w')
        command = [str(self.root/'semi-party.x'), '-N', '2', '-p', str(self.party),
                   '-ip', str(self.hosts), '-lgp', '192', '-e',
                   '-b', str(min(32768,max(1024,spec['width']*len(spec['scales'])*16))),
                   '-IF', str(prefix), '-OF', str(out), name]
        started = time.monotonic()
        env = dict(os.environ)
        libs = self.root/'native-libs'
        if libs.exists():
            env['LD_LIBRARY_PATH'] = str(libs)+':'+str(self.root)+':'+env.get('LD_LIBRARY_PATH','')
        process = subprocess.Popen(command, cwd=str(self.root), stdin=subprocess.DEVNULL,
                                   stdout=handle, stderr=subprocess.STDOUT, env=env,
                                   preexec_fn=_child_setup)
        handle.close()
        return process, private, out, log, started

    def finish(self, state, count):
        process, private, out, log, started = state
        try:
            rc = process.wait(timeout=7200)
            if rc:
                raise RuntimeError('Real MPC failed: '+log.read_text()[-5000:])
            self.calls += 1
            self.seconds += time.monotonic()-started
            if self.party == 1:
                result = Path(str(out)+'-P1-0')
                data = result.read_text()
                values = [int(v) for v in re.findall(r'^S (-?[0-9]+)$', data, re.M)]
                result.unlink()
                if len(values) != count or any(v < 0 or v >= T for v in values):
                    raise ValueError('Invalid masked MPC output: expected=%d parsed=%d out_of_range=%d lines=%d' %
                                     (count,len(values),sum(v<0 or v>=T for v in values),len(data.splitlines())))
                return values
        except BaseException:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
        finally:
            private.unlink(missing_ok=True)
