import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from select_submission_candidates import choose_candidates


def manifest():
    names = [
        "task1_mixed_full_refit", "task1_lookup_safe",
        "task2_japanese_ordinal", "task2_known_source", "task2_lookup_safe",
        "task3_input_only", "task3_overlap_enhanced", "task3_overlap_capped",
    ]
    return {
        "team": "ADML",
        "built_at_utc": "frozen",
        "candidates": [
            {"name": name, "path": f"{name}.jsonl", "sha256": name}
            for name in names
        ],
        "decision_support": {
            "task1": {"public_overlap_fraction": 1.0},
            "task2": {
                "semantic_overlap_fraction": 0.5,
                "median_nearest_similarity": 0.1,
                "source_agreement_ge_2_fraction": 0.5,
            },
            "task3": {"packet_overlap_fraction": 0.75},
        },
    }


def selected(result):
    return {task: row["name"] for task, row in result["selected"].items()}


def test_default_is_policy_independent_despite_overlap():
    assert selected(choose_candidates(manifest())) == {
        "task1": "task1_mixed_full_refit",
        "task2": "task2_japanese_ordinal",
        "task3": "task3_input_only",
    }


def test_explicit_reuse_unlocks_overlap_lanes_and_cap():
    assert selected(choose_candidates(
        manifest(), allow_public_answer_reuse=True, hard_word_limit=True,
    )) == {
        "task1": "task1_lookup_safe",
        "task2": "task2_lookup_safe",
        "task3": "task3_overlap_capped",
    }


def test_source_continuity_diagnostics_do_not_replace_ordinal_primary():
    data = manifest()
    data["decision_support"]["task2"].update({
        "semantic_overlap_fraction": 0.0,
        "median_nearest_similarity": 0.16,
        "source_agreement_ge_2_fraction": 0.65,
    })
    assert selected(choose_candidates(data))["task2"] == "task2_japanese_ordinal"


def test_missing_required_candidate_fails_closed():
    data = manifest()
    data["candidates"] = [
        row for row in data["candidates"] if row["name"] != "task1_mixed_full_refit"
    ]
    try:
        choose_candidates(data)
    except ValueError as error:
        assert "task1_mixed_full_refit" in str(error)
    else:
        raise AssertionError("missing candidate should fail closed")


def test_promoted_task1_lane_supersedes_older_fallback_when_present():
    data = manifest()
    data["candidates"].extend([
        {
            "name": "task1_grouped_consensus_primary",
            "path": "grouped.jsonl", "sha256": "grouped",
        },
        {
            "name": "task1_grouped_lookup_safe",
            "path": "grouped_lookup.jsonl", "sha256": "grouped_lookup",
        },
    ])
    assert selected(choose_candidates(data))["task1"] == (
        "task1_grouped_consensus_primary"
    )
    assert selected(choose_candidates(
        data, allow_public_answer_reuse=True,
    ))["task1"] == "task1_grouped_lookup_safe"
