"""WSL Qwen3 endpoint: embeddings, BFV keys, real MPC party one, final decode."""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import shlex
import socket

import numpy as np

from he import BFV
from mpc import MPC, F, G, T, protocol_fingerprint
from wire import Wire
from checkpoint import fingerprint


def ssh_command(port, target, transport='auto'):
    windows_ssh = '/mnt/c/Windows/System32/OpenSSH/ssh.exe'
    if transport == 'auto':
        transport = 'windows' if Path(windows_ssh).exists() else 'wsl'
    binary = windows_ssh if transport == 'windows' else 'ssh'
    common = ['-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
              '-o', 'ConnectTimeout=180', '-o', 'ServerAliveInterval=15',
              '-o', 'ServerAliveCountMax=20', '-o', 'IPQoS=none']
    return [binary]+common+['-o', 'ExitOnForwardFailure=yes',
                            '-L', '127.0.0.1:%d:127.0.0.1:%d' % (port, port),
                            '-R', '127.0.0.1:%d:127.0.0.1:%d' % (port+1, port+1),
                            '-L', '127.0.0.1:%d:127.0.0.1:%d' % (port+2, port+2),
                            '--', target]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', help='Public input for scripted use; normally enter the prompt privately')
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--chat', action='store_true')
    parser.add_argument('--port', type=int, default=23760)
    parser.add_argument('--ssh-transport', choices=['auto','windows','wsl'], default='auto')
    parser.add_argument('--root', default=str(Path(__file__).resolve().parent))
    parser.add_argument('--ssh-host', required=True)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--remote-model', required=True)
    parser.add_argument('--remote-python', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 4 or not 1024 <= args.port <= 65533:
        parser.error('Invalid generation budget or port')
    os.umask(0o077)
    root = Path(args.root).resolve()
    run_id = uuid.uuid4().hex
    work = root/'runtime'/run_id
    work.mkdir(parents=True, mode=0o700)
    report = dict(run_id=run_id, mode='qwen', success=False,
                  endpoint='WSL', evaluator='WHU216', protocol='BFV + OT semi-honest 2PC',
                  final_tokens_decrypted_after_all_steps=True, kv_cache=False,
                  he_scheme='BFVRNS', he_security='HEStd_128_classic', he_modulus=T,
                  he_fraction_bits=F, mpc_fraction_bits=G, mpc_prime_bits=192,
                  mpc_security_model='semi-honest', mpc_ot='SoftSpokenOT')
    report.update(mpc_port=args.port,rpc_port=args.port+2)
    report['protocol_fingerprint'] = protocol_fingerprint()
    print(json.dumps(dict(event='run_started',run_id=run_id,mode='qwen',
                          evaluator=report['evaluator'],pid=os.getpid())),flush=True)
    def record_exception(kind, error, traceback):
        report.update(success=False, error=repr(error))
        (work/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        sys.__excepthook__(kind, error, traceback)
    sys.excepthook = record_exception
    (root/'runtime/latest.json').write_text(json.dumps(dict(run_id=run_id,pid=os.getpid(),mode='qwen',
                                                     evaluator=report['evaluator'])))
    started = time.monotonic()
    from transformers import AutoTokenizer
    from weights import Weights

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    prompt = args.prompt if args.prompt is not None else getpass.getpass('Private prompt: ')
    if args.chat:
        prompt = tokenizer.apply_chat_template([dict(role='user', content=prompt)],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or len(ids)+args.steps-1 > 32:
        raise ValueError('Prompt exceeds public context limit')
    weights = Weights(args.model)
    vectors = weights.embedding(ids)
    probe = weights.embedding([0,1,2,3]).astype('<f4')
    digest = hashlib.sha256(probe.tobytes()).hexdigest()
    checkpoint_hash = fingerprint(args.model)
    report.update(checkpoint_fingerprint=checkpoint_hash,input_token_count=len(ids))
    width = vectors.shape[1]
    he = BFV(work/'he', width).generate()
    mpc = MPC(root/'third_party/mp-spdz-0.4.4', 1, args.port, work/'mpc')
    command = ssh_command(args.port,args.ssh_host,args.ssh_transport)+[shlex.join([
        args.remote_python, '-u', args.remote_root+'/server.py', '--root', args.remote_root,
        '--model', args.remote_model, '--port', str(args.port), '--rpc-port',str(args.port+2)])]
    report['ssh_transport'] = 'windows-interop' if command[0].endswith('.exe') else 'wsl-native'
    log = (work/'server.log').open('w')
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log)
    log.close()
    wire = None
    connection = None
    try:
        startup = json.loads(process.stdout.readline(4096))
        if startup.get('type')!='rpc_ready' or startup.get('port')!=args.port+2:
            raise RuntimeError('Invalid SSH RPC startup response')
        connection = socket.create_connection(('127.0.0.1',args.port+2),timeout=180)
        connection.settimeout(None)
        wire = Wire(connection.makefile('rb'),connection.makefile('wb'))
        wire.send(dict(type='hello', run_id=run_id, mode='qwen', bundle=he.public_bundle(),
                       certificate=mpc.certificate(), embedding_probe=digest,
                       checkpoint_fingerprint=checkpoint_hash,
                       protocol_fingerprint=report['protocol_fingerprint'], rpc_nonce=startup['nonce']))
        ready = wire.recv()
        if ready.get('type') != 'ready' or ready.get('server_has_secret_key') is not False:
            raise RuntimeError('Invalid server public handshake')
        mpc.trust_peer(ready['certificate'])
        wire.send(dict(type='inputs', values=[he.encode(he.encrypt(x)) for x in vectors], steps=args.steps))
        encrypted_tokens = []
        masked_decryptions = 0
        while True:
            message = wire.recv()
            if message.get('type') == 'done':
                break
            if message.get('type') == 'tokens':
                if message['step'] != len(encrypted_tokens):
                    raise ValueError('Encrypted token order mismatch')
                # Queue ciphertexts. Decryption waits for the final done barrier.
                encrypted_tokens.append(message['values'])
                continue
            if message.get('type') != 'mpc':
                raise RuntimeError('Unexpected remote operation')
            call, spec = int(message['id']), message['spec']
            if call != mpc.calls:
                raise ValueError('MPC operation order mismatch')
            masked_shares = [he.decrypt_residues(he.decode(x)) for x in message['values']]
            masked_decryptions += len(masked_shares)
            mpc.compile(spec)
            wire.send(dict(type='mpc_ready', id=call))
            start = wire.recv()
            if start != dict(type='mpc_start', id=call):
                raise RuntimeError('MPC barrier mismatch')
            state = mpc.start(spec, masked_shares, call)
            count = int(message['count'])
            result = mpc.finish(state, count*width)
            encrypted = [he.encode(he.encrypt_residues(result[i*width:(i+1)*width]))
                         for i in range(count)]
            wire.send(dict(type='mpc_result', id=call, values=encrypted))
        tokens, max_errors = [], []
        if len(encrypted_tokens) != args.steps:
            raise ValueError('Incomplete encrypted generation')
        for encoded in encrypted_tokens:
            values = np.concatenate([he.decrypt(he.decode(x)) for x in encoded])[:message['vocab']]
            token = int(np.argmax(values))
            expected = np.zeros_like(values); expected[token] = 1
            error = float(np.max(np.abs(values-expected)))
            if not np.isfinite(values).all() or error > 1e-6:
                raise AssertionError('Decrypted token is not one-hot')
            tokens.append(token); max_errors.append(error)
        report.update(generated_token_ids=tokens, token_onehot_max_errors=max_errors)
        report['generated_text'] = tokenizer.decode(tokens,skip_special_tokens=True)
        report.update(message)
        report.update(success=True, elapsed_seconds=time.monotonic()-started,
                      protocol_bytes_sent=wire.sent, protocol_bytes_received=wire.received,
                      client_mpc_calls=mpc.calls, client_mpc_seconds=mpc.seconds,
                      masked_ciphertext_decryptions=masked_decryptions,
                      ring_dimension=he.slots, secret_key_location='WSL memory only')
        wire.writer.close();wire.reader.close();connection.close()
        if process.wait(timeout=30):
            raise RuntimeError('Remote evaluator exited unsuccessfully')
    except BaseException as error:
        report['success'] = False
        report['error'] = repr(error)
        process.terminate()
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        raise
    finally:
        if connection is not None:
            connection.close()
        (work/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
