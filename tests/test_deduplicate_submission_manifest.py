import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from deduplicate_submission_manifest import deduplicate_manifest


def test_deduplicate_manifest_preserves_policy_primary_name():
    manifest = {
        "candidates": [
            {"name": "incidental", "task": 2, "sha256": "same"},
            {"name": "primary", "task": 2, "sha256": "same"},
        ],
        "decision_support": {
            "task2": {"policy_independent_primary": "primary"},
        },
    }
    result = deduplicate_manifest(manifest)
    assert [row["name"] for row in result["candidates"]] == ["primary"]
    assert result["deduplicated_candidates"] == [{
        "name": "incidental", "task": 2, "sha256": "same",
        "duplicate_of": "primary",
    }]
    assert result["manifest_format_version"] == 2
