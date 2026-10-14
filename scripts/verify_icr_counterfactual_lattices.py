#!/usr/bin/env python3
"""Keep only complete five-label lattices independently recovered by teachers."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.icr_silver import (  # noqa: E402
    LABELS, RUBRIC_VERSION, decide_counterfactual_verification,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    candidates = {
        row["pair_id"]: row
        for row in (
            json.loads(line) for line in args.candidates.read_text().splitlines() if line.strip()
        )
    }
    judgments: dict[str, list[dict]] = defaultdict(list)
    for path in args.judgments:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("rubric_version") != RUBRIC_VERSION:
                raise ValueError(f"rubric version mismatch in {path}")
            if record.get("pair_id") in candidates:
                judgments[record["pair_id"]].append(record)

    approved = []
    rejected = []
    reasons: Counter[str] = Counter()
    for pair_id, candidate in candidates.items():
        decision = decide_counterfactual_verification(
            judgments[pair_id], candidate["response"], candidate["target_label"],
            candidate["generator_id"],
        )
        record = candidate | {"verification": asdict(decision)}
        if decision.accepted:
            approved.append(record)
        else:
            rejected.append(record)
            reasons[decision.reason] += 1

    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in approved:
        by_source[row["source_pair_id"]].append(row)
    complete = []
    for source_pair_id, variants in by_source.items():
        if {row["target_label"] for row in variants} != set(LABELS):
            continue
        variants.sort(key=lambda row: LABELS.index(row["target_label"]))
        complete.append({
            "source_pair_id": source_pair_id,
            "issuer": variants[0].get("issuer"),
            "question": variants[0]["question"],
            "generator_id": variants[0]["generator_id"],
            "variants": variants,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("approved_variants.jsonl", approved),
        ("rejected_variants.jsonl", rejected),
        ("complete_lattices.jsonl", complete),
    ):
        with (args.output_dir / filename).open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "candidate_variants": len(candidates),
        "approved_variants": len(approved),
        "rejected_variants": len(rejected),
        "complete_five_label_lattices": len(complete),
        "approval_rate": len(approved) / len(candidates) if candidates else 0,
        "rejection_reasons": dict(reasons),
        "approved_label_counts": dict(Counter(row["target_label"] for row in approved)),
        "judgment_files": [str(path) for path in args.judgments],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

