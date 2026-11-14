"""Evaluate one complete Qwen3 prefix. No client, tokenizer, keygen or decrypt import."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .model import QwenForward
from .openfhe_backend import OpenFHEOps
from .weights import Weights
from .generation import EncryptedGreedy, generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--approximation', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-context', type=int, default=32)
    parser.add_argument('--generate', type=int, default=0,
                        help='Encrypted autoregressive steps; zero evaluates logits only')
    parser.add_argument('--logit-bound', type=float, default=128.0)
    parser.add_argument('--selection-margin', type=float, default=.005)
    parser.add_argument('--sign-steps', type=int, default=40)
    parser.add_argument('--backend', choices=['openfhe', 'perseus', 'perseus-multigpu'], default='openfhe')
    args = parser.parse_args()
    request = Path(args.input)
    meta = json.loads((request/'request.json').read_text())
    count = meta['tokens']
    if type(count) is not int or not 0 < count <= args.max_context:
        raise ValueError('Invalid public request shape')
    output = Path(args.output)
    output.mkdir(exist_ok=False, parents=True)
    started = time.monotonic()
    weights = Weights(args.model)
    if args.backend == 'perseus-multigpu':
        from .perseus_multi_backend import PerseusMultiOps
        ops = PerseusMultiOps(args.bundle)
    elif args.backend == 'perseus':
        from .perseus_backend import PerseusOps
        ops = PerseusOps(args.bundle)
    else:
        ops = OpenFHEOps(args.bundle)
    inputs = [ops.import_ciphertext(request/('%d.bin' % i)) for i in range(count)]
    approximation = json.loads(Path(args.approximation).read_text())

    def progress(stage, layer):
        print(json.dumps({'stage': stage, 'layer': layer,
                          'seconds': round(time.monotonic()-started, 2),
                          'counts': ops.counts}), flush=True)

    model = QwenForward(weights, ops, approximation, max_context=args.max_context, progress=progress)
    if args.generate:
        selector_type = EncryptedGreedy
        if args.backend.startswith('perseus'):
            from .streaming_selection import StreamingEncryptedGreedy
            selector_type = StreamingEncryptedGreedy
        selector = selector_type(ops, weights.config['vocab_size'],
            logit_bound=args.logit_bound, margin=args.selection_margin,
            sign_steps=args.sign_steps)

        def emit(step, token):
            directory = output/('token_%d' % step)
            directory.mkdir()
            for i, ciphertext in enumerate(token):
                ops.export_ciphertext(ciphertext, directory/('%d.bin' % i))

        generate(model, inputs, args.generate, selector, emit=emit, progress=progress)
        (output/'result.json').write_text(json.dumps({
            'tokens': args.generate, 'chunks': (weights.config['vocab_size']+ops.width-1)//ops.width,
            'vocab_size': weights.config['vocab_size'], 'kv_cache': False,
            'token_encoding': 'encrypted_approximate_one_hot',
            'intermediate_decryptions': 0, 'feedback': 'encrypted_embedding_matmul',
            'backend': args.backend, 'seconds': time.monotonic()-started,
            'logit_bound': args.logit_bound, 'selection_margin': args.selection_margin,
            'sign_steps': args.sign_steps, 'counts': ops.counts}))
        return
    result = model.forward(inputs)
    for i, ciphertext in enumerate(result):
        ops.export_ciphertext(ciphertext, output/('%d.bin' % i))
    (output/'result.json').write_text(json.dumps({'chunks': len(result),
        'vocab_size': weights.config['vocab_size'], 'kv_cache': False,
        'backend': args.backend, 'seconds': time.monotonic()-started, 'counts': ops.counts}))


if __name__ == '__main__':
    main()
