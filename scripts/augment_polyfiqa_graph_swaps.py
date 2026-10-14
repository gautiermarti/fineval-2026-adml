#!/usr/bin/env python3
"""Create safe cross-company fact-graph/value-swap realization examples.

Only fact-slot historical templates are swapped.  Oracle copy plans and purely
qualitative templates are excluded.  Donor and target must share tier, family,
and issuer-independent semantic trend state, and every referenced metric/index
must exist in the target graph.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.polyfiqa_delex import (  # noqa: E402
    delexicalize_reference,
    realization_plan,
    restore_reference,
)
from fineval.polyfiqa_facts import build_fact_graph  # noqa: E402
from fineval.polyfiqa_programs import COMPANY_NAMES, semantic_state, task_company  # noqa: E402


def compatible(record, donor_graph, target_graph) -> bool:
    if donor_graph.tier != target_graph.tier or donor_graph.family != target_graph.family:
        return False
    if semantic_state(donor_graph) != semantic_state(target_graph):
        return False
    if any(slot["source"] != "fact" for slot in record["slots"]):
        return False
    if not record["slots"]:
        return False
    return all(
        (fact := target_graph.fact(slot["metric"])) is not None
        and int(slot["index"]) < len(fact.values)
        for slot in record["slots"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", default=[
        ROOT / "data/external/polyfiqa_easy_july/test-00000-of-00001.parquet",
        ROOT / "data/external/polyfiqa_expert_july/test-00000-of-00001.parquet",
    ])
    parser.add_argument("--maximum-targets-per-donor", type=int, default=4)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_graph_to_reference_v1",
    )
    args = parser.parse_args()

    source_rows = []
    for path in args.inputs:
        frame = pd.read_parquet(path)
        for source_index, row in frame.iterrows():
            graph = build_fact_graph(row.task_id, row.query, row.question)
            record, reason = delexicalize_reference(
                row.task_id, row.answer, row.query, graph,
            )
            source_rows.append({
                "source": str(path), "source_index": int(source_index),
                "row": row, "graph": graph, "record": record, "reason": reason,
            })

    augmented = []
    donor_counts = Counter()
    pair_counts = Counter()
    for donor in source_rows:
        record = donor["record"]
        if record is None or not record["slots"]:
            continue
        if any(slot["source"] != "fact" for slot in record["slots"]):
            continue
        donor_company = task_company(donor["row"].task_id)
        candidates = [
            target for target in source_rows
            if task_company(target["row"].task_id) != donor_company
            and compatible(record, donor["graph"], target["graph"])
        ]
        candidates.sort(key=lambda target: (
            task_company(target["row"].task_id), str(target["row"].task_id),
            target["source"], target["source_index"],
        ))
        # Prefer distinct target companies before additional dates.
        selected = []
        seen_companies = set()
        for target in candidates:
            company = task_company(target["row"].task_id)
            if company in seen_companies:
                continue
            selected.append(target)
            seen_companies.add(company)
            if len(selected) == args.maximum_targets_per_donor:
                break
        for target in selected:
            target_company = task_company(target["row"].task_id)
            target_record = dict(record)
            target_record["company_possessive_surface"] = COMPANY_NAMES[target_company] + "'s"
            restored = restore_reference(
                target_record, target["graph"], target["row"].task_id,
            )
            augmented.append({
                "source_kind": "semantic_state_graph_swap",
                "donor_task_id": donor["row"].task_id,
                "donor_company": donor_company,
                "target_task_id": target["row"].task_id,
                "target_company": target_company,
                "family": target["graph"].family,
                "semantic_state": list(semantic_state(target["graph"])),
                "plan": realization_plan(
                    target_record, target["graph"], include_oracle_evidence=False,
                ),
                "target_template": target_record["template"],
                "slots": target_record["slots"],
                "mechanically_restored_target": restored,
            })
            donor_counts[donor_company] += 1
            pair_counts[f"{donor_company}->{target_company}"] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "counterfactual_training.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in augmented
    ))
    summary = {
        "historical_rows_considered": len(source_rows),
        "counterfactual_rows": len(augmented),
        "maximum_targets_per_donor": args.maximum_targets_per_donor,
        "donor_company_counts": dict(donor_counts),
        "company_pair_counts": dict(pair_counts),
        "families": dict(Counter(row["family"] for row in augmented)),
        "oracle_copy_plans_swapped": 0,
        "qualitative_templates_swapped": 0,
        "semantic_state_required": True,
        "public_test_answers_used": False,
    }
    (args.output_dir / "counterfactual_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
