#!/usr/bin/env python3
"""Confirm an exact Easy-cash evidence expansion outside discovery packets."""

from __future__ import annotations

import argparse
from collections import defaultdict
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


def choose(valid: dict[str, str]) -> str:
    """Frozen exact-operation policy; never aliases extractive to recall."""
    return "evidence:recall" if "evidence:recall" in valid else "baseline"


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"rows": 0, "eligible_edits": 0}
    eligible = [row for row in rows if row["eligible"]]
    by_company = {}
    for company, values in sorted(_group(rows, "company").items()):
        by_company[company] = {
            "rows": len(values),
            "eligible_edits": sum(row["eligible"] for row in values),
            "delta": sum(row["delta"] for row in values) / len(values),
        }
    return {
        "rows": len(rows),
        "eligible_edits": len(eligible),
        "baseline_rouge": sum(row["baseline_score"] for row in rows) / len(rows),
        "selected_rouge": sum(row["selected_score"] for row in rows) / len(rows),
        "delta": sum(row["delta"] for row in rows) / len(rows),
        "eligible_mean_delta": (
            sum(row["delta"] for row in eligible) / len(eligible) if eligible else 0.0
        ),
        "improved_edits": sum(row["delta"] > 0 for row in eligible),
        "harmed_edits": sum(row["delta"] < 0 for row in eligible),
        "by_company": by_company,
        "company_cluster_bootstrap": exact_cluster_bootstrap(
            rows, cluster_key="company", delta_key="delta",
        ),
    }


def _group(rows: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return groups


def promotion_decision(discovery: dict, confirmation: dict, all_rows: dict) -> bool:
    required = {"HON", "JNJ", "MSFT", "UVV"}
    company_deltas = {
        company: values["delta"]
        for company, values in discovery.get("by_company", {}).items()
        if values["eligible_edits"] > 0
    }
    return bool(
        confirmation.get("eligible_edits", 0) >= 2
        and confirmation.get("harmed_edits", 0) == 0
        and confirmation.get("improved_edits", 0) >= 1
        and confirmation.get("eligible_mean_delta", 0.0) > 0
        and set(company_deltas) == required
        and all(delta > 0 for delta in company_deltas.values())
        and all_rows.get("delta", 0.0) > 0
    )


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
        default=ROOT / "artifacts/polyfiqa_cash_evidence_recall_confirmation_v1",
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
        internal_id = input_grounded_task_id(row)
        graph = build_fact_graph(
            internal_id, row.query, row.question, ASIAN_NUMBERS.get(internal_id),
        )
        if graph.family != "cash":
            continue
        context = parse_polyfiqa_query(row.query)
        generated = sparse_edits(
            prediction["answer"], easy_candidates(graph, context, internal_id),
        )
        valid = {
            name: cap_generated_answer(answer)
            for name, answer in generated.items()
            if candidate_valid(cap_generated_answer(answer), row, internal_id)
        }
        if "baseline" not in valid:
            continue
        selected = choose(valid)
        baseline_score = rouge1_f1([row.answer], [valid["baseline"]])
        selected_score = rouge1_f1([row.answer], [valid[selected]])
        audits.append({
            "row_index": position, "task_id": row.task_id,
            "internal_id": internal_id,
            "company": str(internal_id).split("_", 1)[0],
            "partition": (
                "discovery"
                if fingerprint(row.query, row.question) in discovery_fingerprints
                else "confirmation"
            ),
            "eligible": selected != "baseline", "selected": selected,
            "baseline_score": baseline_score, "selected_score": selected_score,
            "delta": selected_score - baseline_score,
        })

    partitions = {
        name: summarize([row for row in audits if row["partition"] == name])
        for name in ("discovery", "confirmation")
    }
    all_summary = summarize(audits)
    result = {
        "policy": (
            "select exact evidence:recall only on Easy cash rows when the "
            "validated recall evidence differs from extractive evidence"
        ),
        "all": all_summary, "partitions": partitions,
        "promotion_eligible": promotion_decision(
            partitions["discovery"], partitions["confirmation"], all_summary,
        ),
        "confirmation_is_packet_disjoint_from_discovery": True,
        "confirmation_reference_scores_read_once_after_preregistration": True,
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
