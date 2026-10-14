from scripts.evaluate_polyfiqa_large_content_planner import (
    COMPANIES, FAMILIES, load_jsonl, ordered_lcs_f1, parse_selection,
    select_demonstrations, select_screen, set_f1, summarize,
)
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_parse_selection_rejects_unknown_and_duplicates():
    assert parse_selection('{"selected":["a:0"]}', ["a:0", "b:0"]) == (["a:0"], None)
    assert parse_selection('{"selected":["x:0"]}', ["a:0"])[1] == "unknown_id"
    assert parse_selection('{"selected":["a:0","a:0"]}', ["a:0"])[1] == "duplicate_id"


def test_invalid_output_never_receives_empty_set_credit():
    assert set_f1([], None) == 0.0
    assert set_f1([], []) == 1.0


def test_set_and_order_scores_are_distinct():
    target = ["a:0", "b:0", "c:0"]
    reversed_prediction = list(reversed(target))
    assert set_f1(target, reversed_prediction) == 1.0
    assert ordered_lcs_f1(target, reversed_prediction) == 1 / 3


def test_summarize_handles_a_group_with_no_nonempty_targets():
    result = summarize(
        [{"target": [], "prediction": []}],
        "prediction",
    )
    assert result["macro_set_f1"] == 1.0
    assert result["nonempty_macro_set_f1"] == 0.0


def test_real_screen_has_complete_cells_and_company_disjoint_demos():
    corpus_dir = ROOT / "artifacts/polyfiqa_graph_to_reference_v5"
    screen = select_screen(corpus_dir / "loco")
    assert len(screen) == len(COMPANIES) * len(FAMILIES) == 32
    assert {
        (item["held_company"], item["record"]["plan"]["family"])
        for item in screen
    } == {(company, family) for company in COMPANIES for family in FAMILIES}
    corpus = load_jsonl(corpus_dir / "training.jsonl")
    for item in screen:
        held = item["held_company"]
        family = item["record"]["plan"]["family"]
        demos = select_demonstrations(corpus, held, family)
        assert {demo["company"] for demo in demos} == set(COMPANIES) - {held}
