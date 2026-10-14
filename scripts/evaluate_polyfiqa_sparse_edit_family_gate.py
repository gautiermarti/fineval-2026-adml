#!/usr/bin/env python3
"""Leave-company-out conservative gate for sparse PolyFiQA edits."""

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

from fineval.polyfiqa_family_gate import exact_cluster_bootstrap  # noqa: E402


COMPANIES = ("HON", "JNJ", "MSFT", "UVV")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def company(record: dict) -> str:
    value = str(record["internal_id"]).split("_", 1)[0]
    if value not in COMPANIES:
        raise ValueError(f"unexpected company: {value}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lattice", type=Path,
        default=ROOT / "artifacts/polyfiqa_sparse_edit_oracle_v1/lattice.jsonl",
    )
    parser.add_argument("--minimum-rows-per-company", type=int, default=3)
    parser.add_argument("--minimum-mean-gain", type=float, default=0.005)
    parser.add_argument("--minimum-company-gain", type=float, default=-0.002)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_sparse_edit_family_gate_v1",
    )
    args = parser.parse_args()

    records = load_jsonl(args.lattice)
    output_rows = []
    gates_by_fold = {}
    for held_company in COMPANIES:
        train = [row for row in records if company(row) != held_company]
        held = [row for row in records if company(row) == held_company]
        observations = defaultdict(list)
        for row in train:
            baseline_score = float(row["scores"]["baseline"])
            for name, candidate_score in row["scores"].items():
                if name == "baseline":
                    continue
                key = row["tier"], row["family"], name
                observations[key].append({
                    "company": company(row),
                    "delta": float(candidate_score) - baseline_score,
                })
        gates = {}
        for key, values in observations.items():
            by_company = {}
            for training_company in sorted({row["company"] for row in values}):
                deltas = [
                    row["delta"] for row in values
                    if row["company"] == training_company
                ]
                if len(deltas) >= args.minimum_rows_per_company:
                    by_company[training_company] = sum(deltas) / len(deltas)
            mean_gain = sum(row["delta"] for row in values) / len(values)
            admitted = bool(
                len(by_company) == len(COMPANIES) - 1
                and mean_gain >= args.minimum_mean_gain
                and min(by_company.values()) >= args.minimum_company_gain
            )
            gates[key] = {
                "admitted": admitted, "training_rows": len(values),
                "mean_gain": mean_gain, "company_gains": by_company,
            }
        gates_by_fold[held_company] = {
            "admitted": [
                {"tier": key[0], "family": key[1], "operation": key[2], **value}
                for key, value in gates.items() if value["admitted"]
            ],
        }

        for row in held:
            choices = [(0.0, "baseline")]
            for name in row["candidates"]:
                if name == "baseline":
                    continue
                gate = gates.get((row["tier"], row["family"], name))
                if gate is not None and gate["admitted"]:
                    choices.append((float(gate["mean_gain"]), name))
            estimated_gain, selected = max(
                choices, key=lambda item: (item[0], item[1] == "baseline"),
            )
            baseline_score = float(row["scores"]["baseline"])
            selected_score = float(row["scores"][selected])
            output_rows.append({
                "row_index": int(row["row_index"]), "task_id": row["task_id"],
                "held_company": held_company, "tier": row["tier"],
                "family": row["family"], "selected": selected,
                "estimated_training_gain": estimated_gain,
                "baseline_score": baseline_score,
                "selected_score": selected_score,
                "delta": selected_score - baseline_score,
                "answer": row["candidates"][selected],
            })

    output_rows.sort(key=lambda row: row["row_index"])
    by_company = {}
    for held_company in COMPANIES:
        subset = [row for row in output_rows if row["held_company"] == held_company]
        by_company[held_company] = {
            "rows": len(subset),
            "changed": sum(row["selected"] != "baseline" for row in subset),
            "baseline": sum(row["baseline_score"] for row in subset) / len(subset),
            "selected": sum(row["selected_score"] for row in subset) / len(subset),
            "delta": sum(row["delta"] for row in subset) / len(subset),
        }
    bootstrap = exact_cluster_bootstrap(
        output_rows, cluster_key="held_company", delta_key="delta",
    )
    baseline = sum(row["baseline_score"] for row in output_rows) / len(output_rows)
    selected = sum(row["selected_score"] for row in output_rows) / len(output_rows)
    summary = {
        "evaluation": "leave-company-out complete-answer sparse edit routing",
        "rows": len(output_rows),
        "baseline_rouge": baseline,
        "selected_rouge": selected,
        "delta": selected - baseline,
        "changed_rows": sum(row["selected"] != "baseline" for row in output_rows),
        "improved_rows": sum(row["delta"] > 0 for row in output_rows),
        "harmed_rows": sum(row["delta"] < 0 for row in output_rows),
        "thresholds": {
            "minimum_rows_per_company": args.minimum_rows_per_company,
            "minimum_mean_gain": args.minimum_mean_gain,
            "minimum_company_gain": args.minimum_company_gain,
        },
        "by_company": by_company,
        "company_cluster_bootstrap": bootstrap,
        "promotion_eligible": bool(
            selected > baseline
            and sum(row["delta"] >= 0 for row in by_company.values()) >= 3
            and sum(row["delta"] < -0.002 for row in by_company.values()) <= 1
            and bootstrap["p_gain"] >= 0.90
            and bootstrap["p05"] >= 0.0
        ),
        "current_hidden_answers_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions.jsonl").write_text("".join(
        json.dumps({"task_id": row["task_id"], "answer": row["answer"]}, ensure_ascii=False) + "\n"
        for row in output_rows
    ))
    pd.DataFrame(output_rows).drop(columns=["answer"]).to_csv(
        args.output_dir / "audit.csv", index=False,
    )
    (args.output_dir / "gates.json").write_text(
        json.dumps(gates_by_fold, indent=2) + "\n"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
