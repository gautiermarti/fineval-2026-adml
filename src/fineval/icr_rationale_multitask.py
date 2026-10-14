"""Token-alignment utilities for source-safe JF-ICR rationale supervision."""

from __future__ import annotations


def unique_subsequence_start(sequence: list[int], needle: list[int]) -> int | None:
    """Return the start of one unique exact token subsequence."""
    if not needle or len(needle) > len(sequence):
        return None
    starts = [
        index
        for index in range(len(sequence) - len(needle) + 1)
        if sequence[index:index + len(needle)] == needle
    ]
    return starts[0] if len(starts) == 1 else None


def response_token_bounds(input_ids: list[int], sep_token_id: int) -> tuple[int, int]:
    """Return half-open response bounds for a BERT question/response pair."""
    separators = [
        index for index, token_id in enumerate(input_ids)
        if token_id == sep_token_id
    ]
    if len(separators) < 2 or separators[-1] <= separators[0] + 0:
        raise ValueError("encoded pair does not contain two ordered separators")
    return separators[0] + 1, separators[-1]


def rationale_token_labels(
    input_ids: list[int],
    span_token_ids: list[list[int]],
    *,
    sep_token_id: int,
    unk_token_id: int | None,
) -> tuple[list[int], int]:
    """Build response-only binary targets from uniquely recoverable cue spans.

    Returns an all-ignore target when no span is safely recoverable. This is
    important: truncation or tokenization failure must not create a false
    all-negative rationale example.
    """
    response_start, response_end = response_token_bounds(input_ids, sep_token_id)
    response_ids = input_ids[response_start:response_end]
    recovered: list[tuple[int, int]] = []
    for span_ids in span_token_ids:
        if not span_ids or (
            unk_token_id is not None and unk_token_id in span_ids
        ):
            continue
        relative_start = unique_subsequence_start(response_ids, span_ids)
        if relative_start is None:
            continue
        start = response_start + relative_start
        recovered.append((start, start + len(span_ids)))
    labels = [-100] * len(input_ids)
    if not recovered:
        return labels, 0
    for index in range(response_start, response_end):
        labels[index] = 0
    for start, end in recovered:
        labels[start:end] = [1] * (end - start)
    return labels, len(recovered)
