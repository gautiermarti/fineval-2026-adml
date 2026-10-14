#!/usr/bin/env python3
"""Build and strictly validate alternatives from verified evidence translations."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.metrics import rouge1_f1  # noqa: E402
from fineval.parsing import parse_polyfiqa_query  # noqa: E402
from fineval.polyfiqa_facts import question_family  # noqa: E402
from fineval.polyfiqa_translation import replace_evidence_translations  # noqa: E402
from fineval.validation import validate_answer  # noqa: E402
from predict_polyfiqa_hybrid import input_grounded_task_id  # noqa: E402
from validate_polyfiqa_predictions import asian_derived_numbers, derived_numbers  # noqa: E402


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path).reset_index(drop=True)
    return pd.DataFrame(
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    )


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def translation_record(row: dict) -> dict:
    return {
        "start": row["source_start"],
        "end": row["source_end"],
        "label": row["source_label"],
        "translation": row["translation"],
    }


def is_valid(answer: str, packet: pd.Series) -> tuple[bool, str]:
    context = parse_polyfiqa_query(packet.query)
    source = "\n".join((
        context.financial_statements, context.english_news, context.chinese_news,
        context.japanese_news, context.spanish_news, context.greek_news,
    ))
    tier, _ = question_family(packet.question)
    routed = packet.copy()
    routed["task_id"] = input_grounded_task_id(packet)
    result = validate_answer(
        answer, source, tier,
        allowed_numbers=(
            derived_numbers(packet.question, context.financial_statements)
            + asian_derived_numbers(routed)
        ),
    )
    return result.valid, "valid" if result.valid else ";".join(result.reasons)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = load_frame(args.packets)
    predictions = load_jsonl(args.predictions)
    if len(frame) != len(predictions):
        raise ValueError("packet and prediction counts differ")
    verified: dict[int, list[dict]] = defaultdict(list)
    for row in load_jsonl(args.verified):
        if row.get("accepted"):
            verified[int(row["row_index"])].append(row)

    all_outputs = []
    lattice_rows = []
    validation_reasons: Counter[str] = Counter()
    scores = []
    for position, (packet, control) in enumerate(
        zip(frame.itertuples(index=False), predictions, strict=True)
    ):
        packet_series = pd.Series(packet._asdict())
        if str(packet_series.task_id) != str(control["task_id"]):
            raise ValueError(f"task ID mismatch at row {position}")
        candidates = {"baseline": control["answer"]}
        for row in verified[position]:
            candidate, reason = replace_evidence_translations(
                control["answer"], [translation_record(row)],
            )
            if candidate is None:
                validation_reasons[reason] += 1
                continue
            valid, reason = is_valid(candidate, packet_series)
            if valid:
                candidates[f"translation:{row['segment_index']}"] = candidate
            else:
                validation_reasons[reason] += 1
        if verified[position]:
            candidate, reason = replace_evidence_translations(
                control["answer"], [translation_record(row) for row in verified[position]],
            )
            if candidate is not None:
                valid, reason = is_valid(candidate, packet_series)
                if valid:
                    candidates["translation:all"] = candidate
                else:
                    validation_reasons[reason] += 1
            else:
                validation_reasons[reason] += 1
        chosen = "translation:all" if "translation:all" in candidates else "baseline"
        all_outputs.append({"task_id": control["task_id"], "answer": candidates[chosen]})
        record = {
            "row_index": position, "task_id": control["task_id"],
            "chosen": chosen, "candidates": candidates,
        }
        if "answer" in frame.columns:
            reference = str(packet_series.answer)
            candidate_scores = {
                name: rouge1_f1([reference], [answer]) for name, answer in candidates.items()
            }
            oracle = max(candidate_scores, key=candidate_scores.get)
            record |= {"scores": candidate_scores, "oracle": oracle}
            scores.append({
                "baseline": candidate_scores["baseline"],
                "all": candidate_scores[chosen],
                "oracle": candidate_scores[oracle],
                "changed": chosen != "baseline",
                "improved": candidate_scores[chosen] > candidate_scores["baseline"],
                "harmed": candidate_scores[chosen] < candidate_scores["baseline"],
            })
        lattice_rows.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "all_verified.jsonl").open("w") as handle:
        for row in all_outputs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_dir / "lattice.jsonl").open("w") as handle:
        for row in lattice_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "rows": len(frame),
        "verified_translation_segments": sum(len(rows) for rows in verified.values()),
        "rows_with_verified_translation": len(verified),
        "rows_changed": sum(row["chosen"] != "baseline" for row in lattice_rows),
        "validation_rejections": dict(validation_reasons),
    }
    if scores:
        values = pd.DataFrame(scores)
        summary |= {
            "baseline_rouge": float(values.baseline.mean()),
            "all_verified_rouge": float(values["all"].mean()),
            "oracle_rouge": float(values.oracle.mean()),
            "improved_rows": int(values.improved.sum()),
            "harmed_rows": int(values.harmed.sum()),
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

