"""Full-history dense Qwen3 forward over ciphertexts. No attention state survives a call."""
from __future__ import annotations

import math
import numpy as np


class QwenForward:
    def __init__(self, weights, ops, approximations, *, max_context=32,
                 require_encrypted=True, progress=None):
        if require_encrypted and not ops.is_encrypted:
            raise ValueError('HE inference refuses a plaintext or simulated backend')
        self.weights, self.ops = weights, ops
        self.approximations = approximations
        self.config = weights.config
        self.max_context = max_context
        self.progress = progress or (lambda *args: None)
        c = self.config
        self.hidden = int(c['hidden_size'])
        self.heads = int(c['num_attention_heads'])
        self.kv_heads = int(c['num_key_value_heads'])
        self.head_dim = int(c.get('head_dim', self.hidden//self.heads))
        if ops.width != self.hidden or self.hidden != self.heads*self.head_dim:
            raise ValueError('The encrypted evaluator uses one SIMD vector per hidden token')
        if self.heads % self.kv_heads:
            raise ValueError('Invalid GQA geometry')
        self.rope_config = c.get('rope_parameters') or c.get('rope_scaling') or {}
        if self.rope_config.get('rope_type', self.rope_config.get('type', 'default')) != 'default':
            raise ValueError('The encrypted evaluator supports default RoPE only')
        if c.get('attention_bias', False):
            raise ValueError('The encrypted evaluator expects the bias-free Qwen3 checkpoint')

    def _weight(self, layer, name):
        return self.weights.tensor('model.layers.%d.%s' % (layer, name))

    def _norm(self, x, gamma, site):
        return self.ops.rms_norm(x, gamma, float(self.config['rms_norm_eps']),
                                 self.approximations['norm'][site])

    def forward(self, encrypted_embeddings):
        ops, c = self.ops, self.config
        if not 0 < len(encrypted_embeddings) <= self.max_context:
            raise ValueError('Context outside the configured limit; truncation is forbidden')
        rows = list(encrypted_embeddings)
        expansion = self.heads // self.kv_heads
        for layer in range(int(c['num_hidden_layers'])):
            self.progress('layer_start', layer)
            gamma = self._weight(layer, 'input_layernorm.weight')
            normalized = [self._norm(x, gamma, '%d.input' % layer) for x in rows]
            q_weight = self._weight(layer, 'self_attn.q_proj.weight')
            k_weight = np.repeat(self._weight(layer, 'self_attn.k_proj.weight').reshape(self.kv_heads, self.head_dim, self.hidden), expansion, axis=0).reshape(self.hidden, self.hidden)
            v_weight = np.repeat(self._weight(layer, 'self_attn.v_proj.weight').reshape(self.kv_heads, self.head_dim, self.hidden), expansion, axis=0).reshape(self.hidden, self.hidden)
            q = [ops.linear([x], q_weight)[0] for x in normalized]
            k = [ops.linear([x], k_weight)[0] for x in normalized]
            v = [ops.linear([x], v_weight)[0] for x in normalized]
            q_gamma = self._weight(layer, 'self_attn.q_norm.weight')
            k_gamma = self._weight(layer, 'self_attn.k_norm.weight')
            theta = float(self.rope_config.get('rope_theta', c.get('rope_theta', 1000000.0)))
            q = [ops.rope(self._norm(x, q_gamma, '%d.q' % layer), p, self.head_dim, theta) for p, x in enumerate(q)]
            k = [ops.rope(self._norm(x, k_gamma, '%d.k' % layer), p, self.head_dim, theta) for p, x in enumerate(k)]
            attention = []
            for position, query in enumerate(q):
                scores = [ops.mul_plain(ops.group_sum(ops.mul(query, key), self.head_dim), 1.0/math.sqrt(self.head_dim)) for key in k[:position+1]]
                probabilities = ops.softmax(scores, **self.approximations.get('softmax', {}))
                attention.append(ops.sum_values(ops.mul(prob, value) for prob, value in zip(probabilities, v[:position+1])))
            out_weight = self._weight(layer, 'self_attn.o_proj.weight')
            rows = [ops.add(skip, ops.linear([a], out_weight)[0]) for skip, a in zip(rows, attention)]
            del q, k, v, attention, normalized, q_weight, k_weight, v_weight, out_weight
            gamma = self._weight(layer, 'post_attention_layernorm.weight')
            normalized = [self._norm(x, gamma, '%d.post' % layer) for x in rows]
            gate_weight = self._weight(layer, 'mlp.gate_proj.weight')
            up_weight = self._weight(layer, 'mlp.up_proj.weight')
            down_weight = self._weight(layer, 'mlp.down_proj.weight')
            activation = self.approximations['silu'][str(layer)]
            updated = []
            for skip, x in zip(rows, normalized):
                gate = ops.linear([x], gate_weight)
                up = ops.linear([x], up_weight)
                active = [ops.silu(g, activation) for g in gate]
                hidden = [ops.mul(g, u) for g, u in zip(active, up)]
                updated.append(ops.add(skip, ops.linear(hidden, down_weight)[0]))
            rows = updated
            self.progress('layer_done', layer)
        last = self._norm(rows[-1], self.weights.tensor('model.norm.weight'), 'final')
        return ops.linear([last], self.weights.tensor('lm_head.weight'))
