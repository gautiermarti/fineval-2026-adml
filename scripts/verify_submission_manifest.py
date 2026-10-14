#!/usr/bin/env python3
"""Fail closed if an ADML submission portfolio changed after it was built."""

from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_count(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pandas as pd

        return int(len(pd.read_parquet(path)))
    if suffix in {".jsonl", ".ndjson"}:
        with path.open() as handle:
            return sum(bool(line.strip()) for line in handle)
    if suffix == ".csv":
        import pandas as pd

        return int(len(pd.read_csv(path)))
    if suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            return len(payload)
        for key in ("data", "rows", "instances", "examples"):
            if isinstance(payload, dict) and isinstance(payload.get(key), list):
                return len(payload[key])
    raise ValueError(f"cannot determine row count for {path}")


def resolve_path(value: str, root: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def verify_record(record: dict, *, root: Path = ROOT) -> None:
    required = {"path", "sha256", "bytes", "rows"}
    missing = required.difference(record)
    if missing:
        raise ValueError(f"manifest record lacks {sorted(missing)}: {record}")
    path = resolve_path(record["path"], root)
    if not path.is_file():
        raise ValueError(f"manifest file is missing: {path}")
    actual_digest = digest(path)
    if actual_digest != record["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path}")
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"byte-count mismatch for {path}")
    if row_count(path) != int(record["rows"]):
        raise ValueError(f"row-count mismatch for {path}")


def verify_manifest(manifest: dict, *, root: Path = ROOT) -> dict:
    if manifest.get("team") != "ADML":
        raise ValueError("manifest team must be ADML")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("manifest has no fingerprinted organizer inputs")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("manifest has no candidates")
    names = [row.get("name") for row in candidates]
    if None in names or len(names) != len(set(names)):
        raise ValueError("candidate names must be present and unique")
    for record in inputs.values():
        verify_record(record, root=root)
    for record in candidates:
        verify_record(record, root=root)
    aliases = manifest.get("deduplicated_candidates", [])
    if not isinstance(aliases, list):
        raise ValueError("deduplicated_candidates must be a list")
    canonical = {
        (int(row["task"]), row["name"]): row for row in candidates
    }
    for record in aliases:
        verify_record(record, root=root)
        target = canonical.get((int(record["task"]), record.get("duplicate_of")))
        if target is None:
            raise ValueError(f"deduplicated alias has no canonical candidate: {record}")
        if record["sha256"] != target["sha256"]:
            raise ValueError(f"deduplicated alias hash differs from canonical: {record}")
    if int(manifest.get("manifest_format_version", 1)) >= 2:
        identities = [(int(row["task"]), row["sha256"]) for row in candidates]
        if len(identities) != len(set(identities)):
            raise ValueError("manifest contains duplicate same-task output hashes")
    return {
        "team": "ADML",
        "inputs_verified": len(inputs),
        "candidates_verified": len(candidates),
        "deduplicated_aliases_verified": len(aliases),
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    sidecar = manifest_path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise SystemExit(f"missing manifest digest sidecar: {sidecar}")
    expected = sidecar.read_text().split()[0]
    actual = digest(manifest_path)
    if expected != actual:
        raise SystemExit("manifest SHA-256 sidecar mismatch")
    result = verify_manifest(
        json.loads(manifest_path.read_text()), root=args.root.resolve(),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
