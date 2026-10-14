#!/usr/bin/env python3
"""Inventory organizer Drive folders and detect silent file replacements."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import gdown


ROOT = Path(__file__).resolve().parents[1]

FOLDERS: dict[str, dict[str, Any]] = {
    "task1": {
        "id": "1ySqWbFvPdv6Da1aWd-SeB7eb8aexyWO7",
        "files": {
            "greek_test_participant.parquet": {
                "remote_id": "1M_UvmwNBWBQITQcTQfYjplUwmTlAVBev",
                "local": (
                    "data/raw/test_release_20260811/task1/"
                    "greek_test_participant.parquet"
                ),
            },
            "train-00000-of-00001.parquet": {
                "remote_id": "1JBicdVScFYa4_q5NeT8f7H85IyJdIyow",
                "local": "data/raw/subtask1/train-00000-of-00001.parquet",
            },
            "validation-00000-of-00001.parquet": {
                "remote_id": "1aXZ4lbK7hBsPCXeWNf1ytQdSkYhQtC2X",
                "local": "data/raw/subtask1/validation-00000-of-00001.parquet",
            },
        },
    },
    "task2": {
        "id": "1XCXhjQG2ZCzYPhds-r-LkVX5xDer0h7c",
        "files": {
            "JF-ICR_public_set.parquet": {
                "remote_id": "1rLtMUd9fngFm2ua7Ex0WHE2tdE6zFAFT",
                "local": "data/raw/subtask2/JF-ICR_public_set.parquet",
            },
            "JF-ICR_test_participant.parquet": {
                "remote_id": "19Y4GtT1rRe2Hadks9Y7ZF1Eukyf0YRUX",
                "local": (
                    "data/raw/test_release_20260811/task2/"
                    "JF-ICR_test_participant.parquet"
                ),
            },
        },
    },
    "task3": {
        "id": "1V2qBZn7-LhCj74nuhdatBij_a2nBClSC",
        "files": {
            "PolyFiQA_test_participant.parquet": {
                "remote_id": "1CVHF_Et52fdOo3pBtWa_ZnYb99NT0oS9",
                "local": (
                    "data/raw/test_release_20260811/task3/"
                    "PolyFiQA_test_participant.parquet"
                ),
            },
            "public-00000-of-00001.parquet": {
                "remote_id": "1MjGTpK4IoZNEeo9o3CE3aehD3IqgC6yV",
                "local": "data/raw/subtask3/public-00000-of-00001.parquet",
            },
        },
    },
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def inventory_differences(
    expected: dict[str, dict[str, str]],
    observed: list[dict[str, str]],
) -> dict[str, list[str]]:
    """Return path/id differences without consulting file contents."""
    remote = {item["path"]: item["id"] for item in observed}
    return {
        "new_remote_paths": sorted(set(remote) - set(expected)),
        "missing_remote_paths": sorted(set(expected) - set(remote)),
        "changed_remote_ids": sorted(
            path for path in set(expected) & set(remote)
            if expected[path]["remote_id"] != remote[path]
        ),
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--inventory-only", action="store_true",
        help="List paths and IDs without downloading bytes for hash comparison.",
    )
    parser.add_argument(
        "--allow-differences", action="store_true",
        help="Return success after recording a mismatch (never changes local data).",
    )
    args = parser.parse_args()

    report: dict[str, Any] = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "organizer_site": "https://www.finnlp2026sharedtask.info/",
        "mutated_local_organizer_data": False,
        "inventory_only": args.inventory_only,
        "tasks": {},
    }
    overall_ok = True

    with tempfile.TemporaryDirectory(prefix="fineval-organizer-audit-") as temp:
        temp_root = Path(temp)
        for task, config in FOLDERS.items():
            folder_id = config["id"]
            listed = gdown.download_folder(
                id=folder_id,
                skip_download=True,
                quiet=True,
                use_cookies=False,
            )
            observed = [
                {"id": item.id, "path": item.path}
                for item in listed
            ]
            differences = inventory_differences(config["files"], observed)
            task_record: dict[str, Any] = {
                "folder_id": folder_id,
                "folder_url": f"https://drive.google.com/drive/folders/{folder_id}",
                "inventory": observed,
                "inventory_differences": differences,
                "files": [],
            }
            if any(differences.values()):
                overall_ok = False

            for item in observed:
                expected = config["files"].get(item["path"])
                record: dict[str, Any] = {
                    "path": item["path"],
                    "remote_id": item["id"],
                    "expected_remote_id": (
                        expected["remote_id"] if expected else None
                    ),
                    "local_path": expected["local"] if expected else None,
                }
                if args.inventory_only:
                    record["content_status"] = "not_downloaded"
                    task_record["files"].append(record)
                    continue

                destination = temp_root / task / item["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                downloaded = gdown.download(
                    id=item["id"],
                    output=str(destination),
                    quiet=True,
                    use_cookies=False,
                )
                if downloaded is None or not destination.is_file():
                    record["content_status"] = "download_failed"
                    overall_ok = False
                    task_record["files"].append(record)
                    continue
                record.update({
                    "remote_sha256": digest(destination),
                    "remote_bytes": destination.stat().st_size,
                })
                if expected is None:
                    record["content_status"] = "new_remote_file"
                    overall_ok = False
                else:
                    local = ROOT / expected["local"]
                    if not local.is_file():
                        record["content_status"] = "missing_local_file"
                        overall_ok = False
                    else:
                        local_hash = digest(local)
                        record.update({
                            "local_sha256": local_hash,
                            "local_bytes": local.stat().st_size,
                        })
                        if local_hash == record["remote_sha256"]:
                            record["content_status"] = "byte_identical"
                        else:
                            record["content_status"] = "content_mismatch"
                            overall_ok = False
                task_record["files"].append(record)
            report["tasks"][task] = task_record

    report["all_expected_paths_and_ids_present"] = all(
        not any(task["inventory_differences"].values())
        for task in report["tasks"].values()
    )
    report["all_downloaded_files_byte_identical"] = (
        None if args.inventory_only else all(
            item["content_status"] == "byte_identical"
            for task in report["tasks"].values()
            for item in task["files"]
        )
    )
    report["status"] = "pass" if overall_ok else "attention_required"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "all_expected_paths_and_ids_present": report[
            "all_expected_paths_and_ids_present"
        ],
        "all_downloaded_files_byte_identical": report[
            "all_downloaded_files_byte_identical"
        ],
        "output": str(args.output),
    }, indent=2))
    if not overall_ok and not args.allow_differences:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
