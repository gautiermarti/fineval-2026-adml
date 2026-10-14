#!/usr/bin/env python3
"""Build strictly agreed ICR silver labels from independent teacher passes."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.icr_silver import (  # noqa: E402
    RUBRIC_VERSION,
    audit_teacher_independence,
    decide_agreement,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs", type=Path,
        default=ROOT / "data/external/japanese_ir_qa/forward_candidates.jsonl",
    )
    parser.add_argument("--judgments", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    pairs = {
        row["pair_id"]: row
        for row in (
            json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()
        )
    }
    grouped: dict[str, list[dict]] = {pair_id: [] for pair_id in pairs}
    all_judgments = []
    observed = set()
    for path in args.judgments:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("rubric_version") != RUBRIC_VERSION:
                raise ValueError(f"rubric version mismatch in {path}")
            identity = (
                record.get("teacher_id"), record.get("pair_id"), record.get("rubric"),
            )
            if identity in observed:
                raise ValueError(f"duplicate teacher/pair/rubric judgment: {identity}")
            observed.add(identity)
            all_judgments.append(record)
            if record.get("pair_id") in grouped:
                grouped[record["pair_id"]].append(record)

    independence = audit_teacher_independence(all_judgments)
    if not independence["passed"]:
        raise ValueError(f"teacher independence audit failed: {independence}")

    accepted = []
    rejected = []
    reasons: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    issuers: Counter[str] = Counter()
    for pair_id, pair in pairs.items():
        decision = decide_agreement(grouped[pair_id], pair["response"])
        audit = asdict(decision)
        if decision.accepted:
            row = pair | {
                "silver_label": decision.label,
                "agreement_audit": audit,
                "teacher_judgments": grouped[pair_id],
            }
            accepted.append(row)
            labels[str(decision.label)] += 1
            issuers[str(pair.get("issuer", "unknown"))] += 1
        else:
            rejected.append({
                "pair_id": pair_id,
                "issuer": pair.get("issuer"),
                "decision": audit,
            })
            reasons[decision.reason] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "accepted.jsonl").open("w") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_dir / "rejected.jsonl").open("w") as handle:
        for row in rejected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "rubric_version": RUBRIC_VERSION,
        "teacher_independence": independence,
        "pairs": len(pairs),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": len(accepted) / len(pairs) if pairs else 0,
        "labels": dict(labels),
        "accepted_by_issuer": dict(issuers),
        "rejection_reasons": dict(reasons),
        "judgment_files": [str(path) for path in args.judgments],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
