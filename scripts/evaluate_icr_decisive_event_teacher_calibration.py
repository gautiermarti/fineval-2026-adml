#!/usr/bin/env python3
"""Evaluate strict multi-teacher decisive-event calibration for JF-ICR."""

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
    RUBRIC_VERSION, audit_teacher_independence,
    decide_decisive_event_agreement,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queue", type=Path,
        default=ROOT / "artifacts/icr_decisive_event_queue_v1/calibration_queue.jsonl",
    )
    parser.add_argument("--judgments", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    queue = {
        int(row["id"]): row for row in (
            json.loads(line) for line in args.queue.read_text().splitlines() if line.strip()
        )
    }
    grouped: dict[int, list[dict]] = {identifier: [] for identifier in queue}
    all_judgments = []
    for path in args.judgments:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("rubric_version") != RUBRIC_VERSION:
                raise ValueError(f"rubric version mismatch in {path}")
            identifier = int(record["id"])
            if identifier in grouped:
                grouped[identifier].append(record)
                all_judgments.append(record)

    independence = audit_teacher_independence(all_judgments)
    if not independence["passed"]:
        raise ValueError(
            "teacher independence audit failed: "
            + json.dumps(independence, ensure_ascii=False, sort_keys=True)
        )

    accepted = []
    rejected = []
    reasons: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    for identifier, row in queue.items():
        decision = decide_decisive_event_agreement(
            grouped[identifier], row["response"], str(row["gold_label"]),
        )
        audit = asdict(decision)
        if decision.accepted:
            accepted.append({
                "id": identifier,
                "source_fold": int(row["source_fold"]),
                "gold_label": str(row["gold_label"]),
                "question": row["question"],
                "response": row["response"],
                "decisive_span": decision.decisive_span,
                "factors": dict(decision.factors),
                "agreement_audit": audit,
                "teacher_judgments": grouped[identifier],
            })
            labels[str(row["gold_label"])] += 1
        else:
            rejected.append({"id": identifier, "decision": audit})
            reasons[decision.reason] += 1

    rate = len(accepted) / len(queue) if queue else 0.0
    retained_labels = sorted(labels)
    summary = {
        "rubric_version": RUBRIC_VERSION,
        "calibration_rows": len(queue),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "exact_span_factor_agreement_rate": rate,
        "accepted_label_counts": dict(labels),
        "retained_labels": retained_labels,
        "rejection_reasons": dict(reasons),
        "teacher_independence": independence,
        "frozen_gate": {
            "minimum_retained_labels": 4,
            "minimum_agreement_rate": 0.70,
            "passed": len(retained_labels) >= 4 and rate >= 0.70,
        },
        "gold_labels_used_only_for_calibration_anchoring": True,
        "silver_label_claim": False,
        "current_or_hidden_test_used": False,
        "judgment_files": [str(path) for path in args.judgments],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("accepted.jsonl", accepted), ("rejected.jsonl", rejected)):
        (args.output_dir / name).write_text("".join(
            json.dumps(item, ensure_ascii=False) + "\n" for item in rows
        ))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
