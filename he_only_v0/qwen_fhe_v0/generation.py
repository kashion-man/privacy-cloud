"""Unoptimized encrypted greedy decoding; no plaintext-dependent branching.

Tokens use approximate encrypted one-hot vectors in the vocabulary layout.
The public bound and minimum gap are circuit preconditions, checked against a
separate reference run and against the final decrypted token vectors by client.
This is an arithmetic tournament baseline, not Perseus's optimized CutMax.
"""
import math
import numpy as np


class EncryptedGreedy:
    def __init__(self, ops, vocab_size, *, logit_bound=128.0, margin=.005,
                 sign_steps=40):
        if not 0 < margin < logit_bound or sign_steps < 1:
            raise ValueError('Invalid public selection parameters')
        self.ops, self.vocab_size = ops, int(vocab_size)
        self.bound, self.margin = float(logit_bound), float(margin)
        self.steps = int(sign_steps)

    def sign(self, x):
        # Converges to +/-1 on [-1,1] except at zero. No comparison or decrypt.
        o = self.ops
        for _ in range(self.steps):
            x = o.prepare_value(x)
            x2 = o.mul(x, x)
            x = o.mul(x, o.add_plain(o.mul_plain(x2, -.5), 1.5))
            x = o.range_hint(x, -1.00001, 1.00001)
        return x

    def maximum(self, a, b):
        o = self.ops
        difference = o.mul_plain(o.sub(b, a), .5)
        positive = o.mul_plain(o.add_plain(self.sign(difference), 1.0), .5)
        return o.range_hint(o.add(a, o.mul(o.sub(b, a), positive)), -1.00001, 1.00001)

    def select(self, logits):
        o, B, n = self.ops, self.ops.width, self.vocab_size
        if len(logits) != math.ceil(n/B):
            raise ValueError('Logit shape mismatch')
        masks = [(np.arange(B)+k*B < n).astype(float) for k in range(len(logits))]
        scaled = [o.range_hint(o.mul_plain(x, 1.0/self.bound), -1, 1) for x in logits]
        # Padding participates with the public lower bound, never with zero.
        candidates = [o.add_plain(o.mul_plain(x, m), -(1.0-m))
                      for x, m in zip(scaled, masks)]
        while len(candidates) > 1:
            next_level = [self.maximum(candidates[i], candidates[i+1])
                          for i in range(0, len(candidates)-1, 2)]
            if len(candidates) % 2:
                next_level.append(candidates[-1])
            candidates = next_level
        maximum = candidates[0]
        shift = 1
        while shift < B:
            maximum = self.maximum(maximum, o.rotate(maximum, shift))
            shift *= 2
        threshold = self.margin/self.bound
        tokens = []
        for x, mask in zip(scaled, masks):
            difference = o.mul_plain(o.add_plain(o.sub(x, maximum), threshold), .5)
            one_hot = o.mul_plain(o.add_plain(self.sign(difference), 1.0), .5)
            tokens.append(o.range_hint(o.mul_plain(one_hot, mask), 0, 1.00001))
        return tokens


def generate(model, prefix, steps, selector, *, emit=None, progress=None):
    """Keep selected tokens and all feedback encrypted for the entire request.

    Only emit serializes the encrypted token. It must not perform decryption.
    No K/V state: each forward recomputes every layer over the complete prefix.
    """
    if not 1 <= steps or len(prefix)+steps-1 > model.max_context:
        raise ValueError('Generation exceeds full-prefix context limit')
    prefix = list(prefix)
    emitted = []
    progress = progress or (lambda *args: None)
    for step in range(steps):
        progress('forward_start', step)
        logits = model.forward(prefix)
        progress('selection_start', step)
        token = selector.select(logits)
        del logits
        if emit:
            emit(step, token)
        else:
            emitted.append(token)
        progress('token_encrypted', step)
        if step+1 < steps:
            embedding = model.weights.tensor('model.embed_tokens.weight')
            feedback = model.ops.linear(token, embedding.T)
            if len(feedback) != 1:
                raise ValueError('Invalid feedback embedding shape')
            prefix.append(feedback[0])
            del feedback, embedding
            progress('feedback_done', step)
        del token
    return emitted
