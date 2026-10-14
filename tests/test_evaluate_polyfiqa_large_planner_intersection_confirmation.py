from pathlib import Path

from scripts.evaluate_polyfiqa_large_content_planner import COMPANIES
from scripts.evaluate_polyfiqa_large_planner_intersection_confirmation import (
    FAMILIES,
    confirmation_hash,
    intersection,
    select_confirmation_screen,
    v1_identities,
)


ROOT = Path(__file__).resolve().parents[1]


def test_intersection_only_deletes_and_preserves_incumbent_order():
    incumbent = ["b:0", "a:0", "c:0"]
    assert intersection(incumbent, ["c:0", "b:0", "x:0"]) == ["b:0", "c:0"]
    assert intersection(incumbent, []) == []
    assert intersection(incumbent, None) == incumbent


def test_confirmation_hash_is_stable_and_identity_sensitive():
    row = {"plan": {"task_id": "HON_1"}}
    value = confirmation_hash("HON", "cash", 3, row)
    assert value == confirmation_hash("HON", "cash", 3, row)
    assert value != confirmation_hash("HON", "cash", 4, row)


def test_real_confirmation_screen_is_complete_and_disjoint_from_v1():
    corpus_dir = ROOT / "artifacts/polyfiqa_graph_to_reference_v5"
    excluded = v1_identities(
        ROOT / "artifacts/polyfiqa_large_content_planner_v1/manifest.json"
    )
    screen = select_confirmation_screen(corpus_dir / "loco", excluded)
    assert len(screen) == len(COMPANIES) * len(FAMILIES) == 28
    assert {
        (item["held_company"], item["record"]["plan"]["family"])
        for item in screen
    } == {(company, family) for company in COMPANIES for family in FAMILIES}
    assert all(
        (
            item["held_company"],
            str(item["record"]["source"]),
            int(item["record"]["source_index"]),
        ) not in excluded
        for item in screen
    )
