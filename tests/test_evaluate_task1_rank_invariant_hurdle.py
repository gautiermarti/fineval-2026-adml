import numpy as np

from scripts.evaluate_task1_rank_invariant_hurdle import (
    empirical_group_percentiles,
)


def test_percentiles_are_computed_within_group_and_average_ties():
    scores = np.asarray([1.0, 2.0, 2.0, 100.0, -5.0])
    groups = np.asarray([0, 0, 0, 1, 1])
    result = empirical_group_percentiles(scores, groups)
    assert np.allclose(result[:3], [0.0, 0.75, 0.75])
    assert np.allclose(result[3:], [1.0, 0.0])


def test_percentile_singleton_is_one_and_bad_shapes_fail():
    assert empirical_group_percentiles(
        np.asarray([4.0]), np.asarray([9]),
    ).tolist() == [1.0]
    try:
        empirical_group_percentiles(np.asarray([1.0]), np.asarray([0, 1]))
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch must fail")
