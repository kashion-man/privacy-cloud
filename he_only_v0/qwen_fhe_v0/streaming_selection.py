"""The same arithmetic tournament, with bounded live ciphertext storage on SM86."""
import math
import numpy as np
from .generation import EncryptedGreedy


class StreamingEncryptedGreedy(EncryptedGreedy):
    def select(self, logits):
        if not isinstance(logits, list):
            raise TypeError('Selection consumes an owned ciphertext list')
        o, B, n = self.ops, self.ops.width, self.vocab_size
        if len(logits) != math.ceil(n/B):
            raise ValueError('Logit shape mismatch')
        # Binary carry reduction keeps at most log2(chunks) candidate maxima.
        # Original logits remain encrypted until the final indicator pass.
        stack = []
        restore = getattr(o, 'restore_ciphertext', lambda x: x)
        spill = getattr(o, 'spill_ciphertext', lambda x: x)
        for k, x in enumerate(logits):
            x = restore(x)
            mask = (np.arange(B)+k*B < n).astype(float)
            scaled = o.range_hint(o.mul_plain(x, 1.0/self.bound), -1, 1)
            candidate = o.add_plain(o.mul_plain(scaled, mask), -(1-mask))
            size = 1
            while stack and stack[-1][0] == size:
                _, left = stack.pop()
                candidate = self.maximum(left, candidate)
                size *= 2
                del left
            stack.append((size, candidate))
        maximum = stack.pop()[1]
        while stack:
            maximum = self.maximum(stack.pop()[1], maximum)
        shift = 1
        while shift < B:
            maximum = self.maximum(maximum, o.rotate(maximum, shift))
            shift *= 2
        tokens = []
        for k in range(len(logits)):
            mask = (np.arange(B)+k*B < n).astype(float)
            scaled = o.range_hint(o.mul_plain(restore(logits[k]), 1.0/self.bound), -1, 1)
            logits[k] = None
            difference = o.mul_plain(o.add_plain(o.sub(scaled, maximum), self.margin/self.bound), .5)
            one_hot = o.mul_plain(o.add_plain(self.sign(difference), 1.0), .5)
            tokens.append(spill(o.range_hint(o.mul_plain(one_hot, mask), 0, 1.00001)))
        return tokens
