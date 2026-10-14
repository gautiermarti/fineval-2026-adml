"""FinEval metrics, including the released official PolyFiQA tokenizer."""

from collections import Counter
from collections.abc import Sequence
import re
import unicodedata


VALID_ICR_LABELS = frozenset({"+2", "+1", "0", "-1", "-2"})
POLYFIQA_TOKEN_RE = re.compile(
    r"[\u4e00-\u9fff]|[\u3040-\u30ff]|[\u0600-\u06ff]|"
    r"[a-zà-öø-ÿ\u0370-\u03ff]+|\d+(?:[.,]\d+)?",
    re.IGNORECASE,
)


def accuracy(gold: Sequence[str], predictions: Sequence[str]) -> float:
    """Return exact-match accuracy after validating lengths and labels."""
    if len(gold) != len(predictions):
        raise ValueError("gold and predictions must have equal lengths")
    if not gold:
        raise ValueError("gold and predictions must not be empty")
    invalid = (set(gold) | set(predictions)) - VALID_ICR_LABELS
    if invalid:
        raise ValueError(f"invalid JF-ICR labels: {sorted(invalid)}")
    return sum(g == p for g, p in zip(gold, predictions, strict=True)) / len(gold)


def rouge1_f1(gold: Sequence[str], predictions: Sequence[str]) -> float:
    """Return the official macro item-level PolyFiQA ROUGE-1 F1."""
    if len(gold) != len(predictions):
        raise ValueError("gold and predictions must have equal lengths")
    if not gold:
        raise ValueError("gold and predictions must not be empty")
    scores = []
    for reference, prediction in zip(gold, predictions, strict=True):
        reference_tokens = POLYFIQA_TOKEN_RE.findall(
            unicodedata.normalize("NFKC", str(reference or "")).lower()
        )
        prediction_tokens = POLYFIQA_TOKEN_RE.findall(
            unicodedata.normalize("NFKC", str(prediction or "")).lower()
        )
        if not reference_tokens or not prediction_tokens:
            scores.append(float(reference_tokens == prediction_tokens))
            continue
        overlap = sum((Counter(reference_tokens) & Counter(prediction_tokens)).values())
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(reference_tokens)
        scores.append(
            0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        )
    return sum(scores) / len(scores)
