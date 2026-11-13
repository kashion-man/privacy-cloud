"""ProSan preprint §IV-A gradient saliency, adapted to local sensitive spans.

Reference: https://arxiv.org/html/2406.14318v1#S4.SS1
This is the importance component, not ProSan's full sanitizer or privacy score.
"""
import math
import re
from dataclasses import dataclass

from .detection import merge_spans
from .errors import PrivacyError


def response_embedding_gradients(model, prompt_ids, response_ids, device, checkpointing=True):
    """Teacher-forced response NLL; freeze weights and differentiate input positions."""
    import torch
    import torch.nn.functional as F
    if not prompt_ids or not response_ids or any(p.requires_grad for p in model.parameters()):
        raise PrivacyError("ProSan requires nonempty input and frozen local weights")
    ids = torch.tensor([list(prompt_ids) + list(response_ids)], device=device)
    embeddings = model.model.embed_tokens(ids).detach().clone().requires_grad_(True)
    positions = torch.arange(ids.shape[1], device=device).unsqueeze(0)
    hidden = model.model.forward_embeds(embeddings, positions, checkpointing=checkpointing)
    # Causal shift: hidden at last prompt position predicts the first response token.
    logits = model.compute_logits(hidden[:, len(prompt_ids) - 1:-1]).float()
    targets = torch.tensor(response_ids, device=device, dtype=torch.long)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets, reduction='mean')
    gradient, = torch.autograd.grad(loss, embeddings, create_graph=False)
    result = gradient[0, :len(prompt_ids)].detach().float().cpu()
    if not torch.isfinite(loss) or not torch.isfinite(result).all():
        raise PrivacyError("Nonfinite ProSan loss or gradient; cloud request blocked")
    return result


def word_units(text, spans):
    """Keep NER spans whole. Background: Latin words, Han characters, Unicode letters.

    No Chinese segmentation dictionary is downloaded. Han background characters
    are normalization units; a multi-character sensitive entity is always one unit.
    """
    spans = merge_spans(spans, text)
    units, cursor = [], 0
    pattern = r"[A-Za-z0-9_]+(?:['’-][A-Za-z0-9_]+)*|[^\W\d_]"
    for span in spans:
        units.extend((cursor + m.start(), cursor + m.end())
                     for m in re.finditer(pattern, text[cursor:span.start]))
        units.append((span.start, span.end))
        cursor = span.end
    units.extend((cursor + m.start(), cursor + m.end())
                 for m in re.finditer(pattern, text[cursor:]))
    return units


def gradient_norm(gradients, offsets, start, end):
    """Average vectors BEFORE taking L2; never average per-token norms."""
    import torch
    indices = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
    if not indices:
        raise PrivacyError("ProSan word has no aligned gradient; cloud request blocked")
    value = float(torch.linalg.vector_norm(gradients[indices].double().mean(dim=0)))
    if not math.isfinite(value):
        raise PrivacyError("Nonfinite ProSan importance; cloud request blocked")
    return value


@dataclass(repr=False)
class ProSanAnalysis:
    texts: tuple
    fields: list
    threshold: float

    def __repr__(self):
        return 'ProSanAnalysis(<local gradients omitted>)'

    def scores(self, spans_by_field):
        import torch
        if len(spans_by_field) != len(self.texts) or len(self.fields) != len(self.texts):
            raise PrivacyError("ProSan fields are misaligned; cloud request blocked")
        raw_fields = []
        all_norms = []
        for text, (offsets, gradients), spans in zip(self.texts, self.fields, spans_by_field):
            if gradients.ndim != 2 or len(offsets) != gradients.shape[0] or not torch.isfinite(gradients).all():
                raise PrivacyError("Invalid ProSan gradients; cloud request blocked")
            coverage = [False] * len(text)
            for a, b in offsets:
                if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(text):
                    raise PrivacyError("Invalid ProSan character alignment; cloud request blocked")
                coverage[a:b] = [True] * (b - a)
            if any(not covered and not text[i].isspace() for i, covered in enumerate(coverage)):
                raise PrivacyError("Incomplete ProSan gradient coverage; cloud request blocked")
            units = word_units(text, spans)
            raw = {(a, b): gradient_norm(gradients, offsets, a, b) for a, b in units}
            raw_fields.append(raw)
            all_norms.extend(raw.values())
        if not all_norms:
            raise PrivacyError("No ProSan normalization units; cloud request blocked")
        low, high = min(all_norms), max(all_norms)
        # Equal gradients provide no ranking evidence. G1 now releases source
        # text, so never authorize that disclosure from an undefined ranking.
        degenerate = high - low <= 1e-12 * max(1.0, high)
        if degenerate:
            raise PrivacyError("Degenerate ProSan importance cannot authorize G1 retention; cloud request blocked")
        result = []
        for raw, spans in zip(raw_fields, spans_by_field):
            result.append([(raw[(s.start, s.end)] - low) / (high - low)
                           for s in spans])
        return result


class ProSanImportance:
    def __init__(self, model, config):
        self.model, self.config = model, config
        self.threshold = config.prosan_threshold

    def analyze(self, task, context):
        try:
            fields = self.model.prompt_gradients(task, context, self.config)
            return ProSanAnalysis((task, context), fields, self.threshold)
        except PrivacyError:
            raise
        except Exception:
            raise PrivacyError("ProSan local gradient analysis failed; cloud request blocked") from None
