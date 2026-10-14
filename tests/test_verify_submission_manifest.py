import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from verify_submission_manifest import verify_manifest


def record(path: Path) -> dict:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "rows": sum(bool(line.strip()) for line in path.read_text().splitlines()),
    }


def valid_manifest(tmp_path: Path) -> dict:
    source = tmp_path / "test.jsonl"
    prediction = tmp_path / "prediction.jsonl"
    source.write_text('{"id": 1}\n{"id": 2}\n')
    prediction.write_text('{"id": 1, "answer": "+1"}\n{"id": 2, "answer": "0"}\n')
    return {
        "team": "ADML",
        "inputs": {"task2": record(source)},
        "candidates": [
            {"name": "task2_primary", "task": 2, **record(prediction)},
        ],
    }


def test_verifies_complete_unchanged_portfolio(tmp_path):
    assert verify_manifest(valid_manifest(tmp_path), root=tmp_path) == {
        "team": "ADML", "inputs_verified": 1, "candidates_verified": 1,
        "deduplicated_aliases_verified": 0,
    }


def test_rejects_mutated_candidate(tmp_path):
    manifest = valid_manifest(tmp_path)
    Path(manifest["candidates"][0]["path"]).write_text('{"id": 1}\n')
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_manifest(manifest, root=tmp_path)


def test_rejects_missing_input_fingerprints(tmp_path):
    manifest = valid_manifest(tmp_path)
    del manifest["inputs"]
    with pytest.raises(ValueError, match="no fingerprinted organizer inputs"):
        verify_manifest(manifest, root=tmp_path)


def test_version_two_rejects_duplicate_same_task_hashes(tmp_path):
    manifest = valid_manifest(tmp_path)
    manifest["manifest_format_version"] = 2
    duplicate = {**manifest["candidates"][0], "name": "task2_alias"}
    manifest["candidates"].append(duplicate)
    with pytest.raises(ValueError, match="duplicate same-task output hashes"):
        verify_manifest(manifest, root=tmp_path)


def test_verifies_deduplicated_alias(tmp_path):
    manifest = valid_manifest(tmp_path)
    manifest["manifest_format_version"] = 2
    alias = {**manifest["candidates"][0], "name": "task2_alias"}
    alias["duplicate_of"] = "task2_primary"
    manifest["deduplicated_candidates"] = [alias]
    assert verify_manifest(manifest, root=tmp_path)[
        "deduplicated_aliases_verified"
    ] == 1
