from __future__ import annotations
import torch
from torch import nn


def _gumbel_sample(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    scaled = logits.float() / temperatures.unsqueeze(1)
    probs = torch.softmax(scaled, dim=-1)
    gumbel = torch.empty_like(probs).exponential_().clamp_min_(1e-10)
    return probs.div_(gumbel).argmax(dim=-1)


class Sampler(nn.Module):
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ps: torch.Tensor | None = None,
        top_ks: torch.Tensor | None = None,
        min_ps: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        *,
        all_greedy: bool = False,
        any_greedy: bool = False,
        use_top_k: bool = False,
        use_top_p: bool = False,
        use_min_p: bool = False,
        max_top_k: int = 0,
    ) -> torch.Tensor:
        if all_greedy:
            return logits.argmax(dim=-1)

        greedy_mask = temperatures <= 1e-5 if any_greedy else None
        safe_temp = (
            torch.where(greedy_mask, torch.ones_like(temperatures), temperatures)
            if any_greedy
            else temperatures
        )

        if not use_top_k and not use_top_p and not use_min_p:
            if any_greedy:
                sample = _gumbel_sample(logits, safe_temp)
                return torch.where(greedy_mask, logits.argmax(dim=-1), sample)
            return _gumbel_sample(logits, safe_temp)

        scaled = logits.float() / safe_temp.unsqueeze(1)

        if use_top_k:
            vals, idx = torch.topk(scaled, max_top_k, dim=-1)
            col = torch.arange(max_top_k, device=logits.device)
            vals = vals.masked_fill(col.unsqueeze(0) >= top_ks.unsqueeze(1), float("-inf"))
            sample = self._sample_ordered(
                vals, idx, top_ps if use_top_p else None, min_ps if use_min_p else None, generator
            )
        elif use_top_p:
            vals, idx = torch.sort(scaled, descending=True, dim=-1)
            sample = self._sample_ordered(
                vals, idx, top_ps, min_ps if use_min_p else None, generator
            )
        else:
            probs = torch.softmax(scaled, dim=-1)
            max_p = probs.max(dim=-1, keepdim=True).values
            probs = probs.masked_fill(probs < min_ps.unsqueeze(1) * max_p, 0.0)
            sample = torch.multinomial(probs, 1, generator=generator).squeeze(1)

        if any_greedy:
            sample = torch.where(greedy_mask, logits.argmax(dim=-1), sample)
        return sample

    @staticmethod
    def _sample_ordered(
        vals: torch.Tensor,
        idx: torch.Tensor,
        top_ps: torch.Tensor | None,
        min_ps: torch.Tensor | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        probs = torch.softmax(vals, dim=-1)

        if top_ps is not None or min_ps is not None:
            remove = torch.zeros_like(probs, dtype=torch.bool)

            if top_ps is not None:
                cum = probs.cumsum(dim=-1)
                remove |= (cum - probs) > top_ps.unsqueeze(1)

            if min_ps is not None:
                remove |= probs < min_ps.unsqueeze(1) * probs[:, :1]

            remove[:, 0] = False
            probs = probs.masked_fill(remove, 0.0)

        pos = torch.multinomial(probs, 1, generator=generator)
        return idx.gather(1, pos).squeeze(1)
