#!/usr/bin/env python3
"""Measure full-answer oracle headroom from strictly valid sparse edits."""

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

from audit_polyfiqa_hidden_overlap import fingerprint  # noqa: E402
from build_polyfiqa_candidate_lattice import candidate_valid  # noqa: E402
from evaluate_polyfiqa_easy_july import family as easy_family  # noqa: E402
from evaluate_polyfiqa_expert_july import family as expert_family  # noqa: E402
from fineval.metrics import rouge1_f1  # noqa: E402
from fineval.polyfiqa_edits import sparse_edits  # noqa: E402
from predict_polyfiqa_hybrid import cap_generated_answer, is_easy  # noqa: E402


REFERENCES = {
    "easy": ROOT / "data/external/polyfiqa_easy_july/test-00000-of-00001.parquet",
    "expert": ROOT / "data/external/polyfiqa_expert_july/test-00000-of-00001.parquet",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def item_score(reference: str, prediction: str) -> float:
    return rouge1_f1([reference], [prediction])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "data/external/finmmeval_task2/task2_test_questions.jsonl",
    )
    parser.add_argument(
        "--lattice", type=Path,
        default=ROOT / "artifacts/polyfiqa_candidate_lattice_v1/candidate_lattice.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_sparse_edit_oracle_v1",
    )
    args = parser.parse_args()

    inputs = load_jsonl(args.input)
    lattice = load_jsonl(args.lattice)
    if len(inputs) != len(lattice):
        raise ValueError("input and lattice row counts differ")
    references = {}
    for tier, path in REFERENCES.items():
        for row in pd.read_parquet(path).itertuples():
            references[(tier, fingerprint(row.query, row.question))] = row.answer

    rows = []
    lattice_outputs = []
    oracle_outputs = []
    for position, (input_row, lattice_row) in enumerate(zip(inputs, lattice, strict=True)):
        if str(input_row["task_id"]) != str(lattice_row["task_id"]):
            raise ValueError(f"lattice alignment mismatch at row {position}")
        tier = "easy" if is_easy(input_row["question"]) else "expert"
        reference = references.get((tier, fingerprint(input_row["query"], input_row["question"])))
        if reference is None:
            continue
        baseline = lattice_row["candidates"]["balanced"]
        alternatives = {
            name: answer for name, answer in lattice_row["candidates"].items()
            if name != "balanced"
        }
        generated = sparse_edits(baseline, alternatives)
        valid = {}
        for name, answer in generated.items():
            answer = cap_generated_answer(answer)
            if candidate_valid(answer, pd.Series(input_row), lattice_row["internal_id"]):
                valid[name] = answer
        if "baseline" not in valid:
            raise AssertionError(f"baseline became invalid at row {position}")
        scores = {name: item_score(reference, answer) for name, answer in valid.items()}
        oracle_name = max(scores, key=lambda name: (scores[name], name == "baseline"))
        family_fn = easy_family if tier == "easy" else expert_family
        family = family_fn(input_row["question"])
        rows.append({
            "row_index": position, "task_id": input_row["task_id"],
            "internal_id": lattice_row["internal_id"], "tier": tier,
            "family": family, "valid_candidates": len(valid),
            "oracle": oracle_name, "baseline_score": scores["baseline"],
            "oracle_score": scores[oracle_name],
            "delta": scores[oracle_name] - scores["baseline"],
        })
        lattice_outputs.append({
            "row_index": position, "task_id": input_row["task_id"],
            "internal_id": lattice_row["internal_id"], "tier": tier,
            "family": family, "reference": reference,
            "candidates": valid, "scores": scores,
        })
        oracle_outputs.append({
            "task_id": input_row["task_id"], "answer": valid[oracle_name],
        })

    audit = pd.DataFrame(rows)
    by_family = {}
    for (tier, family), group in audit.groupby(["tier", "family"], sort=True):
        by_family[f"{tier}:{family}"] = {
            "rows": len(group),
            "baseline": float(group.baseline_score.mean()),
            "oracle": float(group.oracle_score.mean()),
            "delta": float(group.delta.mean()),
            "improved_rows": int((group.delta > 0).sum()),
        }
    summary = {
        "matched_rows": len(audit),
        "baseline_rouge": float(audit.baseline_score.mean()),
        "sparse_edit_oracle_rouge": float(audit.oracle_score.mean()),
        "oracle_delta": float(audit.delta.mean()),
        "oracle_improved_rows": int((audit.delta > 0).sum()),
        "mean_valid_candidates": float(audit.valid_candidates.mean()),
        "oracle_choice_families": dict(Counter(
            name.split(":", 1)[0] for name in audit.oracle
        )),
        "by_family": by_family,
        "selection_claim": False,
        "current_hidden_answers_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_dir / "audit.csv", index=False)
    (args.output_dir / "oracle.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in oracle_outputs
    ))
    (args.output_dir / "lattice.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in lattice_outputs
    ))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
