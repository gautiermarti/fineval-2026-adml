#!/usr/bin/env python3
"""Confirm the Easy-balance evidence-recall edit outside its discovery packets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_polyfiqa_hidden_overlap import fingerprint  # noqa: E402
from build_polyfiqa_candidate_lattice import (  # noqa: E402
    candidate_valid, easy_candidates,
)
from fineval.metrics import rouge1_f1  # noqa: E402
from fineval.parsing import parse_polyfiqa_query  # noqa: E402
from fineval.polyfiqa_edits import sparse_edits  # noqa: E402
from fineval.polyfiqa_family_gate import exact_cluster_bootstrap  # noqa: E402
from fineval.polyfiqa_facts import build_fact_graph  # noqa: E402
from predict_polyfiqa_hybrid import (  # noqa: E402
    ASIAN_NUMBERS, cap_generated_answer, input_grounded_task_id,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"rows": 0}
    return {
        "rows": len(rows),
        "eligible_edits": sum(row["eligible"] for row in rows),
        "baseline_rouge": sum(row["baseline_score"] for row in rows) / len(rows),
        "selected_rouge": sum(row["selected_score"] for row in rows) / len(rows),
        "delta": sum(row["delta"] for row in rows) / len(rows),
        "improved_edits": sum(row["eligible"] and row["delta"] > 0 for row in rows),
        "harmed_edits": sum(row["eligible"] and row["delta"] < 0 for row in rows),
        "company_cluster_bootstrap": exact_cluster_bootstrap(
            rows, cluster_key="company", delta_key="delta",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--historical", type=Path,
        default=ROOT / "data/external/polyfiqa_easy_july/test-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--historical-baseline", type=Path,
        default=ROOT / "artifacts/polyfiqa_easy_historical_inputonly_v15/predictions.jsonl",
    )
    parser.add_argument(
        "--discovery-input", type=Path,
        default=ROOT / "data/external/finmmeval_task2/task2_test_questions.jsonl",
    )
    parser.add_argument(
        "--discovery-lattice", type=Path,
        default=ROOT / "artifacts/polyfiqa_sparse_edit_oracle_v1/lattice.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_evidence_recall_confirmation_v1",
    )
    args = parser.parse_args()

    historical = pd.read_parquet(args.historical)
    baseline = load_jsonl(args.historical_baseline)
    if len(historical) != len(baseline):
        raise ValueError("historical inputs and baseline predictions are not aligned")
    discovery_inputs = load_jsonl(args.discovery_input)
    discovery_rows = load_jsonl(args.discovery_lattice)
    discovery_fingerprints = {
        fingerprint(
            discovery_inputs[int(row["row_index"])]["query"],
            discovery_inputs[int(row["row_index"])]["question"],
        )
        for row in discovery_rows if row["tier"] == "easy"
    }

    audits = []
    for position, ((_, row), prediction) in enumerate(
        zip(historical.iterrows(), baseline, strict=True)
    ):
        if prediction["task_id"] != row.task_id:
            raise ValueError(f"baseline alignment mismatch at row {position}")
        if not row.question.startswith("How does"):
            continue
        internal_id = input_grounded_task_id(row)
        graph = build_fact_graph(
            internal_id, row.query, row.question, ASIAN_NUMBERS.get(internal_id),
        )
        if graph.family != "balance":
            raise AssertionError(f"balance routing failed at row {position}")
        context = parse_polyfiqa_query(row.query)
        recall = easy_candidates(graph, context, internal_id)["recall"]
        candidates = sparse_edits(prediction["answer"], {"recall": recall})
        valid = {
            name: cap_generated_answer(answer)
            for name, answer in candidates.items()
            if candidate_valid(cap_generated_answer(answer), row, internal_id)
        }
        if "baseline" not in valid:
            # This confirmation concerns the edit policy, not historical parser
            # failures. Invalid incumbents cannot provide a fair paired score.
            continue
        chosen = "evidence:recall" if "evidence:recall" in valid else "baseline"
        baseline_score = rouge1_f1([row.answer], [valid["baseline"]])
        selected_score = rouge1_f1([row.answer], [valid[chosen]])
        packet_fingerprint = fingerprint(row.query, row.question)
        audits.append({
            "row_index": position, "task_id": row.task_id,
            "company": str(row.task_id).split("_", 1)[0],
            "partition": (
                "discovery" if packet_fingerprint in discovery_fingerprints
                else "confirmation"
            ),
            "eligible": chosen != "baseline", "selected": chosen,
            "baseline_score": baseline_score, "selected_score": selected_score,
            "delta": selected_score - baseline_score,
        })

    partitions = {
        name: summarize([row for row in audits if row["partition"] == name])
        for name in ("discovery", "confirmation")
    }
    result = {
        "policy": "use evidence:recall when that grounded sparse edit exists on Easy balance rows",
        "all": summarize(audits), "partitions": partitions,
        "confirmation_is_packet_disjoint_from_discovery": True,
        "historical_references_used_for_evaluation_only": True,
        "current_or_hidden_test_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audits).to_csv(args.output_dir / "audit.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
