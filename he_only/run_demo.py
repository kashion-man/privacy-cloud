"""Colocated protocol demo. Deploy client.py on the trusted client for private serving.

The evaluator is a separate process and receives a public bundle and ciphertext
files only. This convenience demo itself is the key-holding client and must not
be presented as a client protected against the host on which this demo runs.
"""
import argparse
import getpass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from qwen_fhe.client import Client
from qwen_fhe.weights import Weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--approximation', required=True)
    parser.add_argument('--work', required=True)
    parser.add_argument('--prompt', help='For public test prompts; otherwise use hidden input')
    parser.add_argument('--tokens', type=int, default=2)
    parser.add_argument('--max-context', type=int, default=32)
    parser.add_argument('--input-ids', help='Comma-separated public synthetic IDs for protocol tests')
    parser.add_argument('--logit-bound', type=float, default=128.0)
    parser.add_argument('--selection-margin', type=float, default=.005)
    parser.add_argument('--sign-steps', type=int, default=40)
    parser.add_argument('--backend', choices=['openfhe', 'perseus', 'perseus-multigpu'], default='openfhe')
    parser.add_argument('--token-max-error', type=float, default=1e-3)
    parser.add_argument('--token-l1-error', type=float, default=.02,
                        help='Final one-hot L1 limit; zero disables this check')
    parser.add_argument('--token-l2-error', type=float, default=0.,
                        help='Final one-hot L2 limit; zero disables this check')
    args = parser.parse_args()
    if (not all(math.isfinite(x) for x in [args.token_max_error, args.token_l1_error, args.token_l2_error])
        or not 0 < args.token_max_error < .5
        or min(args.token_l1_error, args.token_l2_error) < 0
        or max(args.token_l1_error, args.token_l2_error) == 0):
        parser.error('Require max error in (0,.5) and at least one positive global error limit')
    if not 1 <= args.tokens <= args.max_context:
        parser.error('Invalid generation budget')
    os.environ['HF_HUB_OFFLINE'] = '1'
    tokenizer = None
    if args.input_ids:
        ids = [int(value) for value in args.input_ids.split(',')]
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        prompt = args.prompt if args.prompt is not None else getpass.getpass('Private prompt: ')
        ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or len(ids)+args.tokens-1 > args.max_context:
        parser.error('Full prefix exceeds context limit; input truncation is forbidden')
    weights = Weights(args.model)
    if min(ids) < 0 or max(ids) >= weights.config['vocab_size']:
        parser.error('Token ID outside vocabulary')
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=False, mode=0o700)
    started = time.monotonic()
    print(json.dumps({'stage': 'client_keygen', 'context_tokens': len(ids)}), flush=True)
    client_type = Client
    if args.backend == 'perseus':
        from qwen_fhe.perseus_client import PerseusClient
        client_type = PerseusClient
    elif args.backend == 'perseus-multigpu':
        from qwen_fhe.perseus_multi_client import PerseusMultiClient
        client_type = PerseusMultiClient
    client = client_type(width=int(weights.config['hidden_size']))
    client.save_public_bundle(work/'public')
    print(json.dumps({'stage': 'keygen_done', 'parameters': client.spec,
                      'seconds': time.monotonic()-started}), flush=True)
    request, response = work/'request', work/'response'
    request.mkdir()
    for index, embedding in enumerate(weights.embedding(ids)):
        client.write_ciphertext(client.encrypt(embedding), request/('%d.bin' % index))
    (request/'request.json').write_text(json.dumps({'tokens': len(ids)}))
    command = [sys.executable, '-u', '-m', 'qwen_fhe.server',
               '--model', str(Path(args.model).resolve()), '--bundle', str(work/'public'),
               '--approximation', str(Path(args.approximation).resolve()),
               '--input', str(request), '--output', str(response),
               '--max-context', str(args.max_context), '--generate', str(args.tokens),
               '--logit-bound', str(args.logit_bound),
               '--selection-margin', str(args.selection_margin),
               '--sign-steps', str(args.sign_steps)]
    command.extend(['--backend', args.backend])
    with (work/'evaluator.log').open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    # No decryption is called until the evaluator finishes every generation step.
    meta = json.loads((response/'result.json').read_text())
    if meta['tokens'] != args.tokens or meta['intermediate_decryptions'] != 0:
        raise RuntimeError('Evaluator did not finish the encrypted generation protocol')
    generated, token_errors, token_details, accepted = [], [], [], []
    for step in range(meta['tokens']):
        directory = response/('token_%d' % step)
        values = np.concatenate([client.read_output(directory/('%d.bin' % i))
                                 for i in range(meta['chunks'])])[:meta['vocab_size']]
        if not np.isfinite(values).all():
            raise RuntimeError('Non-finite decrypted encrypted token')
        token = int(values.argmax())
        expected = np.zeros_like(values)
        expected[token] = 1.0
        error = float(np.max(np.abs(values-expected)))
        l1_error = float(np.sum(np.abs(values-expected)))
        l2_error = float(np.linalg.norm(values-expected))
        passed = not (error > args.token_max_error
            or (args.token_l1_error and l1_error > args.token_l1_error)
            or (args.token_l2_error and l2_error > args.token_l2_error))
        accepted.append(passed)
        generated.append(token)
        token_errors.append(error)
        token_details.append({'max_abs_error': error, 'l1_error': l1_error,
                              'l2_error': l2_error, 'winner': float(values[token]),
                              'runner_up': float(np.partition(values, -2)[-2]),
                              'accepted': passed})
        print(json.dumps({'stage': 'final_token_decrypted', 'step': step,
                          'token_id': token, 'one_hot_max_abs_error': error,
                          'accepted': passed,
                          'seconds': time.monotonic()-started}), flush=True)
    if tokenizer is not None and all(accepted):
        print(tokenizer.decode(generated, skip_special_tokens=True), flush=True)
    else:
        print(json.dumps({'synthetic_output_ids': generated}), flush=True)
    (work/'client_metrics.json').write_text(json.dumps({'tokens_generated': len(generated),
        'token_ids': generated, 'one_hot_errors': token_errors,
        'token_details': token_details,
        'accuracy_limits': {'max_abs': args.token_max_error,
                            'l1': args.token_l1_error, 'l2': args.token_l2_error},
        'token_accuracy_passed': all(accepted),
        'intermediate_decryptions': 0, 'server': meta,
        'seconds': time.monotonic()-started, 'kv_cache': False,
        'mode': 'colocated_encrypted_autoregressive_protocol_test'}))
    if not all(accepted):
        raise RuntimeError('Encrypted selection failed accuracy limits; final metrics were retained')


if __name__ == '__main__':
    main()
