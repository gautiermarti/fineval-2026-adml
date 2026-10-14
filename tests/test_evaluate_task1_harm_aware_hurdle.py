import numpy as np

from scripts.evaluate_task1_harm_aware_hurdle import (
    combine_hurdle_probabilities,
    crossfit_hurdle_scores,
    promotion_gate,
)


def operation(group, delta, token):
    return {
        "fold": group,
        "target_delta": delta,
        "text_features": f"operation context {token}",
        "features": {"value": float(token), f"group={group}": 1},
    }


def test_frozen_hurdle_formula_is_signed_and_shape_checked():
    consequence = np.asarray([0.8, 0.8, 0.0])
    direction = np.asarray([0.75, 0.25, 0.9])
    assert np.allclose(
        combine_hurdle_probabilities(consequence, direction),
        [0.4, -0.4, 0.0],
    )
    try:
        combine_hurdle_probabilities(np.asarray([0.5]), np.asarray([0.5, 0.5]))
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch must fail")


def test_crossfit_hurdle_scores_cover_every_operation_and_are_bounded():
    operations = []
    # Every two-group training complement contains consequential positive,
    # consequential negative, and neutral examples for both frozen heads.
    for group in range(3):
        operations.extend([
            operation(group, 0.5, group * 10 + 1),
            operation(group, -0.5, group * 10 + 2),
            operation(group, 0.0, group * 10 + 3),
        ])
    scores = crossfit_hurdle_scores(operations, max_features=20000)
    assert scores.shape == (9,)
    assert np.isfinite(scores).all()
    assert (scores >= -1.0).all()
    assert (scores <= 1.0).all()


def test_promotion_gate_requires_calibration_and_no_validation_harm():
    metrics = {
        "delta": 0.006,
        "rows_improved": 2,
        "rows_harmed": 0,
        "type_deltas": {"P": 0.0, "O": 0.01, "L": 0.0},
        "bootstrap_probability_gain": 0.95,
    }
    assert promotion_gate(metrics, True)["passed"]
    assert not promotion_gate(metrics | {"rows_harmed": 1}, True)["passed"]
    assert not promotion_gate(metrics, False)["passed"]
