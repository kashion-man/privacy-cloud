"""WHU216 evaluator: public weights and HE evaluation keys; MPC party zero."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import socket
import secrets

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

from he import BFV, HybridOps
from mpc import MPC, protocol_fingerprint
from wire import Wire
from checkpoint import fingerprint


def progress(event, **extra):
    print(json.dumps(dict(event=event, **extra)), file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--port', type=int, default=23760)
    parser.add_argument('--rpc-port', type=int)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    nonce = None
    if args.rpc_port:
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        listener.bind(('127.0.0.1',args.rpc_port));listener.listen(1)
        listener.settimeout(180)
        nonce = secrets.token_hex(32)
        # Only this short startup line uses SSH stdio. Ciphertext frames use a
        # loopback TCP channel forwarded by the same authenticated SSH session.
        print(json.dumps(dict(type='rpc_ready',port=args.rpc_port,nonce=nonce)),flush=True)
        connection,_ = listener.accept();listener.close();connection.settimeout(None)
        wire = Wire(connection.makefile('rb'),connection.makefile('wb'))
    else:
        wire = Wire(sys.stdin.buffer, sys.stdout.buffer)
    hello = wire.recv()
    if hello.get('type') != 'hello':
        raise ValueError('Expected a public handshake')
    if nonce is not None and not secrets.compare_digest(hello.get('rpc_nonce',''),nonce):
        raise ValueError('RPC does not belong to the authenticated SSH session')
    if hello.get('protocol_fingerprint') != protocol_fingerprint():
        raise ValueError('Client/cloud HE-MPC circuit sources differ; redeploy both sides')
    run_id = hello['run_id']
    if not run_id.isalnum() or len(run_id) > 48:
        raise ValueError('Invalid run id')
    work = root/'runtime'/run_id
    work.mkdir(mode=0o700, parents=True, exist_ok=False)
    he = BFV(work/'he', int(hello['bundle']['width'])).load_bundle(hello['bundle'])
    mpc = MPC(root/'third_party/mp-spdz-0.4.4', 0, args.port, work/'mpc')
    cert = mpc.certificate(); mpc.trust_peer(hello['certificate'])
    if hello.get('mode') != 'qwen':
        raise ValueError('Expected a Qwen3 inference request')
    from weights import Weights
    weights = Weights(args.model)
    if fingerprint(args.model) != hello['checkpoint_fingerprint']:
        raise ValueError('Client/cloud full public checkpoints differ')
    c = weights.config
    if int(c['hidden_size']) != he.width:
        raise ValueError('Qwen embedding width mismatch')
    probe = weights.embedding([0, 1, 2, 3]).astype('<f4').tobytes()
    if hashlib.sha256(probe).hexdigest() != hello['embedding_probe']:
        raise ValueError('Client/cloud Qwen checkpoints differ')
    if hasattr(he, 'keys'):
        raise AssertionError('Cloud evaluator must not hold a client key pair')
    wire.send(dict(type='ready', certificate=cert, mode='qwen',
                   ring_dimension=he.slots, server_has_secret_key=hasattr(he, 'keys')))
    request = wire.recv()
    if request.get('type') != 'inputs':
        raise ValueError('Expected encrypted inputs')
    prefix = [he.decode(x) for x in request['values']]
    ops = HybridOps(he, mpc, wire)
    started = time.monotonic()
    from qwen import QwenForward
    steps = int(request['steps'])
    if not 1 <= steps <= 4 or not prefix or len(prefix)+steps-1 > 32:
        raise ValueError('v0 supports up to four generated tokens and context 32')
    profile = dict(norm={k: {} for k in ['final']+[
        '%d.%s' % (layer, site) for layer in range(c['num_hidden_layers'])
        for site in ['input','q','k','post']
    ]}, silu={str(layer): {} for layer in range(c['num_hidden_layers'])})
    model = QwenForward(weights, ops, profile, max_context=32,
                        progress=lambda event, layer: progress(event, layer=layer, counts=ops.counts))
    vocab = int(c['vocab_size'])
    embedding = None
    tokens = []
    for step in range(steps):
        progress('forward_start', step=step, prefix_length=len(prefix))
        logits = model.forward(prefix)
        token = ops.mpc_op('argmax', logits, vocab=vocab)
        tokens.append(token)
        progress('encrypted_token_ready', step=step, counts=ops.counts)
        if step+1 < steps:
            if embedding is None:
                embedding = weights.tensor('model.embed_tokens.weight')
            feedback = ops.linear(token, embedding.T)
            if len(feedback) != 1:
                raise ValueError('Invalid encrypted embedding feedback')
            prefix.append(feedback[0])
    # The client obtains unmasked tokens only once ALL requested steps finish.
    # Chunk only after the whole generation loop finishes. A full Qwen
    # vocabulary produces many BFV blocks and must not exceed frame limits.
    for step, token in enumerate(tokens):
        wire.send(dict(type='tokens', step=step, values=[he.encode(x) for x in token]))
    wire.send(dict(type='done',
                   counts=ops.counts, mpc_seconds=mpc.seconds, seconds=time.monotonic()-started,
                   kv_cache=False, server_has_secret_key=False,
                   intermediate_activation_reconstructions=0, model="qwen", vocab=vocab))
    progress('done', counts=ops.counts)


if __name__ == '__main__':
    main()
