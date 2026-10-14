import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_submission_portfolio import deduplicate_candidate_records


def candidate(name: str, task: int, sha256: str) -> dict:
    return {"name": name, "task": task, "sha256": sha256}


def test_deduplicates_only_within_the_same_task():
    records = [
        candidate("task2_a", 2, "same"),
        candidate("task2_b", 2, "same"),
        candidate("task3_a", 3, "same"),
    ]
    unique, aliases = deduplicate_candidate_records(records)
    assert [row["name"] for row in unique] == ["task2_a", "task3_a"]
    assert aliases == [{**records[1], "duplicate_of": "task2_a"}]


def test_policy_preference_selects_the_canonical_name():
    records = [
        candidate("incidental", 2, "same"),
        candidate("task2_primary", 2, "same"),
    ]
    unique, aliases = deduplicate_candidate_records(
        records, preferred_names=("task2_primary",),
    )
    assert [row["name"] for row in unique] == ["task2_primary"]
    assert aliases == [{**records[0], "duplicate_of": "task2_primary"}]
