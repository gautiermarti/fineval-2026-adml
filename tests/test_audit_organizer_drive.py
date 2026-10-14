import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.audit_organizer_drive import inventory_differences


def test_inventory_differences_detects_new_missing_and_changed_ids():
    expected = {
        "same.parquet": {"remote_id": "same", "local": "same"},
        "changed.parquet": {"remote_id": "old", "local": "changed"},
        "missing.parquet": {"remote_id": "missing", "local": "missing"},
    }
    observed = [
        {"path": "same.parquet", "id": "same"},
        {"path": "changed.parquet", "id": "new"},
        {"path": "added.txt", "id": "added"},
    ]
    assert inventory_differences(expected, observed) == {
        "new_remote_paths": ["added.txt"],
        "missing_remote_paths": ["missing.parquet"],
        "changed_remote_ids": ["changed.parquet"],
    }


def test_inventory_differences_accepts_exact_inventory():
    expected = {"a": {"remote_id": "1", "local": "local-a"}}
    observed = [{"path": "a", "id": "1"}]
    assert inventory_differences(expected, observed) == {
        "new_remote_paths": [],
        "missing_remote_paths": [],
        "changed_remote_ids": [],
    }
