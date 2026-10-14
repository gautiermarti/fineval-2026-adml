#!/usr/bin/env python3
"""Upgrade a submission manifest to unique same-task output hashes."""

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_submission_portfolio import (  # noqa: E402
    deduplicate_candidate_records,
    digest,
)


PREFERRED_KEYS = (
    "policy_independent_primary",
    "policy_independent_diversity",
    "validation_only_fallback",
    "source_continuity_diversity",
    "public_reuse_candidate",
    "hard_word_limit_public_reuse_candidate",
)


def deduplicate_manifest(manifest: dict) -> dict:
    preferred_names = tuple(
        str(support[key])
        for support in manifest.get("decision_support", {}).values()
        for key in PREFERRED_KEYS
        if key in support
    )
    candidates, new_aliases = deduplicate_candidate_records(
        manifest.get("candidates", []), preferred_names=preferred_names,
    )
    aliases = {
        (row["task"], row["name"], row["sha256"]): row
        for row in manifest.get("deduplicated_candidates", [])
    }
    aliases.update({
        (row["task"], row["name"], row["sha256"]): row
        for row in new_aliases
    })
    return {
        **manifest,
        "manifest_format_version": 2,
        "candidates": candidates,
        "deduplicated_candidates": list(aliases.values()),
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if source == output:
        raise ValueError("refusing to overwrite the source manifest")
    manifest = deduplicate_manifest(json.loads(source.read_text()))
    manifest["deduplication_audit"] = {
        "source_manifest": str(source),
        "source_manifest_sha256": digest(source),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    output.with_suffix(".sha256").write_text(
        f"{digest(output)}  {output.name}\n"
    )
    print(json.dumps({
        "output": str(output),
        "unique_candidates": len(manifest["candidates"]),
        "deduplicated_aliases": len(manifest["deduplicated_candidates"]),
        "sha256": digest(output),
    }, indent=2))


if __name__ == "__main__":
    main()
