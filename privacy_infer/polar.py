"""Normalized Polar kernel from STAMP §4, used by the optional STAMP mode.

vMF sampling via Wood rejection sampler and Householder rotation. Uses OS-backed
SystemRandom, never a fixed release seed. Fixed candidate vocabulary is required.
Only the fixed-group, fixed-budget token kernel has the paper's mLDP interpretation;
context-dependent NER, deletion and group changes do not inherit that guarantee.
"""
import math
import random

from .errors import PrivacyError

GROUP_RATIOS = {"G1": 2, "G2": 1, "G3": 4, "G4": 3}


def sample_vmf(mu, kappa):
    import torch
    if mu.ndim != 1 or mu.numel() < 2 or not math.isfinite(kappa) or kappa < 0:
        raise ValueError("Invalid vMF parameters")
    mu = mu.detach().to(device="cpu", dtype=torch.float64)
    norm = float(mu.norm())
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid direction")
    mu = mu / norm
    d = mu.numel()
    rng = random.SystemRandom()

    def sphere(n):
        for _ in range(128):
            vector = torch.tensor([rng.gauss(0, 1) for _ in range(n)], dtype=torch.float64)
            if vector.norm() > 1e-12:
                return vector / vector.norm()
        raise PrivacyError("Polar sampling failed")

    if kappa == 0:
        return sphere(d)
    # Stable rationalized expression for b.
    m = d - 1
    b = m / (2 * kappa + math.hypot(2 * kappa, m))
    x = (1 - b) / (1 + b)
    if not 0 <= x < 1:
        raise ValueError("Concentration outside numerical range")
    c = kappa * x + m * math.log1p(-x * x)
    for _ in range(10000):
        z = rng.betavariate(m / 2, m / 2)
        w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
        u = max(rng.random(), 1e-300)
        if kappa * w + m * math.log1p(-x * w) - c >= math.log(u):
            v = sphere(m)
            sample = torch.cat((torch.tensor([w], dtype=torch.float64),
                                math.sqrt(max(0, 1 - w * w)) * v))
            # H maps e1 to mu. Stable identity branch at mu=e1.
            h = torch.zeros_like(mu)
            h[0] = 1
            h = h - mu
            if float(h.norm()) > 1e-12:
                h = h / h.norm()
                sample = sample - 2 * h * torch.dot(h, sample)
            return sample / sample.norm()
    raise PrivacyError("Polar rejection limit reached")


class NormalizedPolar:
    def __init__(self, embeddings, candidate_ids=None, chunk_size=4096):
        import torch
        if embeddings.ndim != 2 or embeddings.shape[1] < 2 or chunk_size <= 0:
            raise ValueError("Invalid embedding table or chunk size")
        self.embeddings = embeddings.detach()
        self.ids = (list(range(embeddings.shape[0])) if candidate_ids is None else list(candidate_ids))
        if not self.ids or len(set(self.ids)) != len(self.ids) or any(type(i) is not int or not 0 <= i < embeddings.shape[0] for i in self.ids):
            raise ValueError("Invalid fixed candidate vocabulary")
        self.chunk_size = chunk_size
        # Validate all candidate vectors once without duplicating a full float32 table.
        for start in range(0, len(self.ids), chunk_size):
            block = self.embeddings[self.ids[start:start + chunk_size]].float()
            if not torch.isfinite(block).all() or (block.norm(dim=-1) <= 1e-12).any():
                raise ValueError("Invalid candidate embedding")

    def privatize(self, token_id, epsilon):
        return self.privatize_many([token_id], [epsilon])[0]

    def privatize_many(self, token_ids, epsilons, batch_size=32):
        """Independent sampling, batched exact decoding with bounded scratch memory."""
        import torch
        import torch.nn.functional as F
        if type(batch_size) is not int or batch_size <= 0 or len(token_ids) != len(epsilons):
            raise ValueError("Invalid batch size or budget alignment")
        if any(type(i) is not int or not 0 <= i < self.embeddings.shape[0] for i in token_ids):
            raise ValueError("Invalid token id")
        if any(isinstance(e, bool) or not math.isfinite(e) or e < 0 for e in epsilons):
            raise ValueError("Invalid token budget")
        result = []
        with torch.inference_mode():
            for first in range(0, len(token_ids), batch_size):
                directions = torch.stack([
                    sample_vmf(self.embeddings[i], e)
                    for i, e in zip(token_ids[first:first + batch_size], epsilons[first:first + batch_size])
                ]).to(device=self.embeddings.device, dtype=torch.float32)
                size = directions.shape[0]
                best_scores = torch.full((size,), -math.inf, device=directions.device)
                best_ids = torch.zeros(size, dtype=torch.long, device=directions.device)
                for start in range(0, len(self.ids), self.chunk_size):
                    ids = torch.tensor(self.ids[start:start + self.chunk_size], device=directions.device)
                    block = F.normalize(self.embeddings[ids].float(), dim=-1)
                    scores, indices = (directions @ block.T).max(dim=1)
                    improved = scores > best_scores
                    best_scores = torch.where(improved, scores, best_scores)
                    best_ids = torch.where(improved, ids[indices], best_ids)
                result.extend(best_ids.cpu().tolist())
        return result

    def privatize_groups(self, token_ids, groups, base_epsilon):
        if len(token_ids) != len(groups) or not math.isfinite(base_epsilon) or base_epsilon < 0:
            raise ValueError("Invalid grouped release")
        if any(group not in GROUP_RATIOS for group in groups):
            raise ValueError("Unknown STAMP group")
        return self.privatize_many(token_ids, [base_epsilon * GROUP_RATIOS[g] for g in groups])
