"""Multi-task components for ordinal JF-ICR and Japanese promise supervision."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


ORDINAL_LEVELS = (-2.0, -1.0, 0.0, 1.0, 2.0)


def normalized_span_bounds(text: str, span: str) -> tuple[int, int] | None:
    """Map a whitespace-insensitive annotated span back to source characters."""
    normalized_text: list[str] = []
    source_positions: list[int] = []
    for position, character in enumerate(text):
        if character == "\ufeff" or character.isspace():
            continue
        folded = character.casefold()
        normalized_text.extend(folded)
        source_positions.extend([position] * len(folded))
    normalized_span = "".join(
        character.casefold()
        for character in span.replace("\ufeff", "")
        if not character.isspace()
    )
    if not normalized_span:
        return None
    start = "".join(normalized_text).find(normalized_span)
    if start < 0:
        return None
    end = start + len(normalized_span) - 1
    return source_positions[start], source_positions[end] + 1


def ordinal_pairwise_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin_per_level: float = 0.25,
) -> torch.Tensor:
    """Hinge loss that orders expected scores for every unequal-label pair."""
    if logits.ndim != 2 or logits.shape[1] != len(ORDINAL_LEVELS):
        raise ValueError("ordinal logits must have shape [batch, 5]")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have shape [batch]")
    levels = logits.new_tensor(ORDINAL_LEVELS)
    scores = logits.softmax(dim=-1) @ levels
    label_delta = labels[:, None] - labels[None, :]
    score_delta = scores[:, None] - scores[None, :]
    mask = torch.triu(label_delta != 0, diagonal=1)
    if not bool(mask.any()):
        return logits.sum() * 0.0
    direction = label_delta.sign().to(score_delta.dtype)
    required = margin_per_level * label_delta.abs().to(score_delta.dtype)
    losses = torch.relu(required - direction * score_delta)
    return losses[mask].mean()


def window_span_tokens(
    prefix_ids: list[int],
    span_ids: list[int],
    suffix_ids: list[int],
    budget: int,
) -> tuple[list[int], list[int]]:
    """Return a context window that always retains the complete labelled span."""
    if budget <= 0:
        raise ValueError("token budget must be positive")
    if not span_ids:
        raise ValueError("positive promise span tokenization is empty")
    if len(span_ids) > budget:
        raise ValueError("promise span exceeds token budget")
    context_budget = budget - len(span_ids)
    left_budget = min(len(prefix_ids), (context_budget + 1) // 2)
    right_budget = min(len(suffix_ids), context_budget - left_budget)
    # Spend unused capacity on whichever side still has context.
    unused = context_budget - left_budget - right_budget
    if unused:
        extra_left = min(len(prefix_ids) - left_budget, unused)
        left_budget += extra_left
        unused -= extra_left
    if unused:
        right_budget += min(len(suffix_ids) - right_budget, unused)
    tokens = prefix_ids[-left_budget:] + span_ids + suffix_ids[:right_budget]
    labels = [0] * left_budget + [1] * len(span_ids) + [0] * right_budget
    if len(tokens) != len(labels) or len(tokens) > budget:
        raise AssertionError("span window construction failed")
    return tokens, labels


class ICRPromiseMultiTaskModel(nn.Module):
    """One encoder with ICR, paragraph-promise, and token-span heads."""

    def __init__(self, model_name_or_path: str, *, revision: str | None = None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name_or_path, revision=revision,
        )
        hidden = int(self.encoder.config.hidden_size)
        dropout_probability = float(getattr(
            self.encoder.config, "hidden_dropout_prob", 0.1,
        ))
        self.dropout = nn.Dropout(dropout_probability)
        self.icr_head = nn.Linear(hidden, len(ORDINAL_LEVELS))
        self.promise_head = nn.Linear(hidden, 2)
        self.span_head = nn.Linear(hidden, 2)

    @classmethod
    def from_checkpoint(cls, path: str) -> "ICRPromiseMultiTaskModel":
        model = cls(path)
        states = torch.load(
            f"{path}/multitask_heads.pt", map_location="cpu", weights_only=True,
        )
        model.icr_head.load_state_dict(states["icr_head"])
        model.promise_head.load_state_dict(states["promise_head"])
        model.span_head.load_state_dict(states["span_head"])
        return model

    def representations(self, **inputs) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(**inputs)
        tokens = outputs.last_hidden_state
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = tokens[:, 0]
        return self.dropout(pooled), self.dropout(tokens)

    def icr_logits(self, **inputs) -> torch.Tensor:
        pooled, _ = self.representations(**inputs)
        return self.icr_head(pooled)

    def promise_logits(
        self, **inputs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled, tokens = self.representations(**inputs)
        return self.promise_head(pooled), self.span_head(tokens)
