"""Fit public-data approximation bands; never called on an encrypted user request."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_fhe_v0.ops import fit_chebyshev


PUBLIC_TEXTS = [
    'The quick brown fox jumps over the lazy dog.',
    'Explain the difference between a list and a dictionary in Python.',
    'A software application shows a blank screen after startup.',
    'Please summarize the following technical issue and suggest troubleshooting steps.',
    'What is privacy preserving inference?',
    'One plus one equals two. The capital of France is Paris.',
    '\u8bf7\u7528\u4e00\u53e5\u8bdd\u89e3\u91ca\u4ec0\u4e48\u662f\u9690\u79c1\u63a8\u7406\u3002',
    '\u4eca\u5929\u5929\u6c14\u5f88\u597d\uff0c\u6211\u4eec\u53bb\u56fe\u4e66\u9986\u5b66\u4e60\u3002',
]


def fit_inverse(lo, hi):
    grid = np.geomspace(lo, hi, 2049)
    if hi/lo > 1000:
        initial = 1/np.sqrt(hi)
        y = np.full_like(grid, initial)
        for steps in range(1, 65):
            y = 0.5*y*(3-grid*y*y)
            error = float(np.max(np.abs(y*np.sqrt(grid)-1)))
            if error < 1e-8:
                return {'method': 'newton', 'lo': float(lo), 'hi': float(hi),
                        'initial': float(initial), 'newton_steps': steps,
                        'max_relative_fit_error': error}
        raise ValueError('Cannot construct a convergent inverse-sqrt circuit')
    for degree in [31, 63, 127, 255]:
        fit = fit_chebyshev(lambda x: 1/np.sqrt(x), lo, hi, degree, newton_steps=1)
        y = np.polynomial.chebyshev.chebval((2*grid-hi-lo)/(hi-lo), fit['coefficients'])
        y = 0.5*y*(3-grid*y*y)
        error = float(np.max(np.abs(y*np.sqrt(grid)-1)))
        if error < 1e-5:
            fit['max_relative_fit_error'] = error
            return fit
    raise ValueError('Public calibration band is too wide for the v0 inverse sqrt: %g..%g' % (lo, hi))


def fit_model(model, token_sequences, degree=127):
    c = model.config
    stats, gates, handles = {}, {}, []
    sites = [('final', model.model.norm)]
    for i, layer in enumerate(model.model.layers):
        sites += [('%d.input' % i, layer.input_layernorm),
                  ('%d.post' % i, layer.post_attention_layernorm),
                  ('%d.q' % i, layer.self_attn.q_norm),
                  ('%d.k' % i, layer.self_attn.k_norm)]
        def gate_hook(module, inp, out, i=i):
            low, high = float(out.min()), float(out.max())
            old = gates.get(str(i), (low, high))
            gates[str(i)] = (min(low, old[0]), max(high, old[1]))
        handles.append(layer.mlp.gate_proj.register_forward_hook(gate_hook))
    for name, module in sites:
        def norm_hook(module, inp, out, name=name):
            x = inp[0].detach().float()
            v = x.square().mean(-1)+float(c.rms_norm_eps)
            low, high = float(v.min()), float(v.max())
            old = stats.get(name, (low, high))
            stats[name] = (min(low, old[0]), max(high, old[1]))
        handles.append(module.register_forward_hook(norm_hook))
    try:
        with torch.inference_mode():
            for ids in token_sequences:
                model(torch.tensor([ids], device=model.device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    norms = {name: fit_inverse(max(1e-12, lo/2), hi*2) for name, (lo, hi) in stats.items()}
    acts = {}
    for name, (lo, hi) in gates.items():
        span = max(abs(lo), abs(hi), 1.0)*1.5
        doublings = max(0, int(np.ceil(np.log2(4*span))))
        fit = {'method': 'tanh_double_angle', 'lo': -span, 'hi': span,
               'doublings': doublings, 'observed_band': [lo, hi]}
        acts[name] = fit
    return {'norm': norms, 'silu': acts, 'softmax': {'rounds': 10, 'inverse_steps': 9},
            'calibration': {'source': 'bundled public synthetic sentences',
                            'sequences': len(token_sequences), 'kv_cache': False,
                            'note': 'Approximation bands are calibrated, not universal guarantees.'}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--max-context', type=int, default=32)
    a = p.parse_args()
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.float32, attn_implementation='eager').to(a.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    sequences = [tokenizer.encode(text, add_special_tokens=False)[:a.max_context] for text in PUBLIC_TEXTS]
    result = fit_model(model, sequences)
    result['model_config'] = model.config.to_dict()
    Path(a.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({'calibrated_norm_sites': len(result['norm']), 'layers': len(result['silu']),
                      'output': a.output}), flush=True)


if __name__ == '__main__':
    main()
