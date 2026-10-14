#!/usr/bin/env python3
"""Recompute validity and within-teacher coverage for a bounded ICR pilot."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.icr_silver import (  # noqa: E402
    RUBRIC_VERSION,
    decide_decisive_event_agreement,
    validate_judgment,
)


def summarize(queue: dict[int, dict], records: list[dict]) -> dict:
    grouped: dict[str, dict[int, list[dict]]] = defaultdict(
        lambda: defaultdict(list),
    )
    duplicate_keys = Counter(
        (str(row.get("teacher_id", "")), int(row["id"]), str(row["rubric"]))
        for row in records
    )
    duplicates = [list(key) for key, count in duplicate_keys.items() if count != 1]
    if duplicates:
        raise ValueError(f"duplicate teacher/id/rubric judgments: {duplicates}")

    recomputed = []
    for record in records:
        if record.get("rubric_version") != RUBRIC_VERSION:
            raise ValueError("rubric version mismatch")
        identifier = int(record["id"])
        if identifier not in queue:
            raise ValueError(f"judgment id {identifier} is absent from the queue")
        source = queue[identifier]
        if str(record.get("gold_label")) != str(source["gold_label"]):
            raise ValueError(f"recorded gold label drift for id {identifier}")
        judgment = record.get("judgment", {})
        valid, reason = validate_judgment(judgment, source["response"])
        if valid and str(judgment.get("label")) != str(source["gold_label"]):
            valid, reason = False, "teacher_changed_fixed_gold_label"
        if bool(record.get("valid")) != valid:
            raise ValueError(f"stored/recomputed validity mismatch for id {identifier}")
        if str(record.get("validation_reason")) != reason:
            raise ValueError(f"stored/recomputed reason mismatch for id {identifier}")
        normalized = dict(record)
        normalized["valid"] = valid
        normalized["validation_reason"] = reason
        recomputed.append(normalized)
        grouped[str(record["teacher_id"])][identifier].append(normalized)

    teacher_summaries = {}
    for teacher, by_id in grouped.items():
        teacher_records = [row for rows in by_id.values() for row in rows]
        two_valid = []
        invariant = []
        decision_reasons = Counter()
        for identifier, rows in by_id.items():
            if len({row["rubric"] for row in rows if row["valid"]}) >= 2:
                two_valid.append(identifier)
            decision = decide_decisive_event_agreement(
                rows,
                queue[identifier]["response"],
                str(queue[identifier]["gold_label"]),
                minimum_teachers=1,
            )
            decision_reasons[decision.reason] += 1
            if decision.accepted:
                invariant.append(identifier)
        teacher_summaries[teacher] = {
            "model": sorted({str(row.get("model", "")) for row in teacher_records}),
            "model_revision": sorted({
                str(row.get("model_revision", "")) for row in teacher_records
            }),
            "records": len(teacher_records),
            "valid_records": sum(bool(row["valid"]) for row in teacher_records),
            "validity_rate": (
                sum(bool(row["valid"]) for row in teacher_records)
                / len(teacher_records)
            ),
            "validation_reasons": dict(Counter(
                str(row["validation_reason"]) for row in teacher_records
            )),
            "pilot_ids": sorted(by_id),
            "ids_with_two_valid_rubrics": sorted(two_valid),
            "ids_with_invariant_span_factors": sorted(invariant),
            "within_teacher_decision_reasons": dict(decision_reasons),
        }

    no_rubric_coverage = [
        teacher for teacher, row in teacher_summaries.items()
        if not row["ids_with_two_valid_rubrics"]
    ]
    return {
        "rubric_version": RUBRIC_VERSION,
        "judgment_records": len(recomputed),
        "pilot_ids": sorted({int(row["id"]) for row in recomputed}),
        "pilot_label_counts": dict(Counter(
            str(queue[identifier]["gold_label"])
            for identifier in {int(row["id"]) for row in recomputed}
        )),
        "teachers": teacher_summaries,
        "pilot_futility": {
            "passed": not no_rubric_coverage,
            "teachers_with_zero_two_rubric_rows": no_rubric_coverage,
            "reason": (
                "every teacher retains at least one two-rubric pilot row"
                if not no_rubric_coverage
                else "a second teacher cannot rescue zero within-teacher rubric coverage"
            ),
        },
        "gold_labels_used_only_for_explanation_anchoring": True,
        "current_or_hidden_test_used": False,
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--queue", type=Path,
        default=ROOT / "artifacts/icr_decisive_event_queue_v1/calibration_queue.jsonl",
    )
    parser.add_argument("--judgments", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    queue = {
        int(row["id"]): row for row in (
            json.loads(line) for line in args.queue.read_text().splitlines()
            if line.strip()
        )
    }
    records = [
        json.loads(line) for path in args.judgments
        for line in path.read_text().splitlines() if line.strip()
    ]
    summary = summarize(queue, records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
