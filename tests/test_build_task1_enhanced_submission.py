import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_task1_enhanced_submission as builder
from build_task1_enhanced_submission import normalize_and_validate, update_manifest


def test_normalize_and_validate_is_fail_closed(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(builder, "run", lambda *arguments: calls.append(arguments))
    source = tmp_path / "input.parquet"
    stale = tmp_path / "stale.jsonl"
    output = tmp_path / "normalized.jsonl"
    report = tmp_path / "normalization.json"

    normalize_and_validate(source, stale, output, report)

    assert calls == [
        (
            "scripts/normalize_task1_submission.py",
            "--input", str(source),
            "--predictions", str(stale),
            "--output", str(output),
            "--report", str(report),
        ),
        (
            "scripts/validate_submission.py", "--task", "1",
            "--input", str(source), "--predictions", str(output),
        ),
    ]


def test_enhanced_manifest_promotes_grouped_lane(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text('{"id": 1}\n')
    manifest = {
        "team": "ADML",
        "candidates": [{"name": "task1_mixed_full_refit", "sha256": "old"}],
        "decision_support": {"task1": {"public_overlap_fraction": 0.0}},
    }
    candidates = [
        {"name": "task1_mixed_full_refit", "task": 1, "sha256": "mixed"},
        {"name": "task1_grouped_consensus_primary", "task": 1, "sha256": "primary"},
        {"name": "task1_two_seed_consensus_diversity", "task": 1, "sha256": "diversity"},
        {"name": "task1_grouped_lookup_safe", "task": 1, "sha256": "lookup"},
    ]
    result = update_manifest(manifest, candidates, input_path=source)
    assert result["decision_support"]["task1"]["policy_independent_primary"] == (
        "task1_grouped_consensus_primary"
    )
    assert result["decision_support"]["task1"]["recommended_first_run"] == (
        "task1_grouped_consensus_primary"
    )
    assert {row["name"] for row in result["candidates"]} == {
        "task1_mixed_full_refit",
        "task1_grouped_consensus_primary",
        "task1_two_seed_consensus_diversity",
        "task1_grouped_lookup_safe",
    }
    mixed = next(
        row for row in result["candidates"]
        if row["name"] == "task1_mixed_full_refit"
    )
    assert mixed == {
        "name": "task1_mixed_full_refit", "task": 1, "sha256": "mixed",
    }


def test_enhanced_manifest_uses_measured_overlap_for_quarantine(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text('{"id": 1}\n')
    manifest = {"team": "ADML", "candidates": [], "decision_support": {}}
    candidates = [
        {"name": "task1_grouped_consensus_primary", "task": 1, "sha256": "primary"},
        {"name": "task1_grouped_lookup_safe", "task": 1, "sha256": "lookup"},
    ]

    result = update_manifest(
        manifest,
        candidates,
        input_path=source,
        overlap_fraction=0.25,
    )

    support = result["decision_support"]["task1"]
    assert support["public_overlap_fraction"] == 0.25
    assert support["recommended_first_run"] == (
        "task1_grouped_lookup_safe (POLICY CONFIRMATION REQUIRED)"
    )
