from collections import Counter

from scripts.evaluate_polyfiqa_nemotron30b_fact_filter import (
    EXPECTED_FOLD_COUNTS,
    MODEL,
    ROOT,
    grounded_intersection,
    parse_selection,
    prepare_rows,
    request_payload,
)


def test_parse_selection_is_exact_and_grounded():
    assert parse_selection({"selected": ["a", "b"]}, ["a", "b", "c"]) == (
        ["a", "b"],
        "valid",
    )
    assert parse_selection(
        {"selected": ["a"], "explanation": "x"}, ["a"]
    )[1] == "invalid_schema"
    assert parse_selection({"selected": ["a", "a"]}, ["a"])[1] == "duplicate_id"
    assert parse_selection({"selected": ["z"]}, ["a"])[1] == "unknown_id"


def test_grounded_intersection_preserves_incumbent_order_and_fails_closed():
    incumbent = ["b", "a", "c"]
    assert grounded_intersection(incumbent, ["c", "b"], True) == ["b", "c"]
    assert grounded_intersection(incumbent, None, False) == incumbent


def test_request_is_one_deterministic_nonthinking_structured_decode():
    payload = request_payload("prompt")
    assert payload["model"] == MODEL
    assert payload["temperature"] == 0.0
    assert payload["seed"] == 2026
    assert payload["max_tokens"] == 192
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["response_format"]["json_schema"]["strict"] is True


def test_frozen_full_population_is_unique_and_company_disjoint():
    rows = prepare_rows(
        ROOT / "artifacts/polyfiqa_graph_to_reference_v5",
        ROOT / "artifacts/polyfiqa_numeric_planner_v5/predictions.jsonl",
    )
    assert len(rows) == 154
    assert Counter(row["held_company"] for row in rows) == EXPECTED_FOLD_COUNTS
    identities = {
        (row["held_company"], row["task_id"], row["tier"], row["family"])
        for row in rows
    }
    assert len(identities) == 154
    for row in rows:
        assert all(
            demo["company"] != row["held_company"]
            for demo in row["demonstrations"]
        )
        assert set(row["incumbent"]) <= set(row["offered"])
