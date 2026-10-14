from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_task1_dual_raw_proposer import (
    choose_oof_gate,
    historical_gate,
    intersect_raw_proposals,
    validation_gate,
)


def record(row, entities):
    return {"row_index": row, "entities": entities}


def entity(start, end, label, score, votes=None):
    value = {
        "start": start, "end": end, "label": label, "score": score,
    }
    if votes is not None:
        value["vote_count"] = votes
    return value


def test_intersection_requires_exact_typed_span_and_uses_minimum_score():
    frame = pd.DataFrame({"text": ["ACME Athens"]})
    left = [record(0, [
        entity(0, 4, "organization", 0.9, 4),
        entity(5, 11, "location", 0.8, 2),
    ])]
    right = [record(0, [
        entity(0, 4, "organization", 0.7, 5),
        entity(0, 4, "location", 0.99, 5),
        entity(5, 11, "location", 0.95, 5),
    ])]
    rows, stats = intersect_raw_proposals(
        frame, left, right, minimum_votes=3,
    )
    assert len(rows[0]) == 1
    assert rows[0][0]["entity"] == "ACME"
    assert rows[0][0]["type"] == "ΟΡΓΑΝΙΣΜΟΣ"
    assert rows[0][0]["score"] == 0.7
    assert stats["shared_exact_spans"] == 1


def metric(threshold, *, delta, p05, folds, types, selected=30, harm=0.1):
    return {
        "threshold": threshold,
        "delta": delta,
        "bootstrap_p05": p05,
        "fold_deltas": {str(index): value for index, value in enumerate(folds)},
        "type_deltas": types,
        "selected_operations": selected,
        "harmful_operation_fraction": harm,
    }


def test_gate_rejects_fold_regression_and_harm_then_maximizes_oof_delta():
    kinds = {"ΠΡΟΣΩΠΟ": 0.01, "ΟΡΓΑΝΙΣΜΟΣ": 0.02, "ΤΟΠΟΘΕΣΙΑ": 0.01}
    rejected_fold = metric(
        0.4, delta=0.06, p05=0.03,
        folds=[0.1, 0.1, -0.001, 0.1, 0.1], types=kinds,
    )
    rejected_harm = metric(
        0.5, delta=0.05, p05=0.03,
        folds=[0.1] * 5, types=kinds, harm=0.151,
    )
    accepted_low = metric(
        0.6, delta=0.03, p05=0.02, folds=[0.01] * 5, types=kinds,
    )
    accepted_high = metric(
        0.7, delta=0.04, p05=0.02, folds=[0.01] * 5, types=kinds,
    )
    chosen = choose_oof_gate([
        rejected_fold, rejected_harm, accepted_low, accepted_high,
    ])
    assert chosen is accepted_high


def test_frozen_split_promotion_gates():
    validation = {
        "delta": 0.01, "rows_improved": 2, "rows_harmed": 0,
        "type_deltas": {"P": 0.0, "O": 0.01, "L": 0.0},
        "bootstrap_probability_gain": 0.81,
    }
    assert validation_gate(validation)
    assert not validation_gate(validation | {"rows_harmed": 1})
    historical = {
        "delta": 0.001, "rows_improved": 3, "rows_harmed": 2,
        "type_deltas": {"P": 0.0, "O": 0.01, "L": -0.004},
        "bootstrap_probability_gain": 0.61,
    }
    assert historical_gate(historical)
    assert not historical_gate(historical | {"rows_improved": 2})
