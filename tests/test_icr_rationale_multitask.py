import pytest

from fineval.icr_rationale_multitask import (
    rationale_token_labels,
    response_token_bounds,
    unique_subsequence_start,
)


def test_unique_subsequence_requires_exactly_one_match():
    assert unique_subsequence_start([1, 2, 3, 4], [2, 3]) == 1
    assert unique_subsequence_start([1, 2, 1, 2], [1, 2]) is None
    assert unique_subsequence_start([1, 2], []) is None


def test_response_token_bounds_uses_pair_separators():
    assert response_token_bounds([2, 10, 3, 20, 21, 3], 3) == (3, 5)
    with pytest.raises(ValueError):
        response_token_bounds([2, 10, 3], 3)


def test_rationale_targets_ignore_question_and_union_nested_response_spans():
    ids = [2, 10, 3, 20, 21, 22, 23, 3]
    labels, recovered = rationale_token_labels(
        ids, [[21, 22], [22]], sep_token_id=3, unk_token_id=1,
    )
    assert recovered == 2
    assert labels == [-100, -100, -100, 0, 1, 1, 0, -100]


def test_no_recoverable_span_is_all_ignore_not_all_negative():
    ids = [2, 10, 3, 20, 21, 3]
    labels, recovered = rationale_token_labels(
        ids, [[1], [99]], sep_token_id=3, unk_token_id=1,
    )
    assert recovered == 0
    assert labels == [-100] * len(ids)


def test_repeated_response_span_is_rejected_even_if_absent_from_question():
    ids = [2, 20, 3, 20, 21, 20, 21, 3]
    labels, recovered = rationale_token_labels(
        ids, [[20, 21]], sep_token_id=3, unk_token_id=1,
    )
    assert recovered == 0
    assert labels == [-100] * len(ids)
