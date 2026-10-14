#!/usr/bin/env python3
"""Build strict graph-to-delexicalized-reference training records.

Historical Easy/Expert references are used.  Current competition outputs and
public-test answers are not read.  Copy slots selected with the historical
reference are explicitly marked as oracle plans; a deployable system must learn
that plan separately before the realizer can be evaluated end to end.
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
from fineval.polyfiqa_programs import task_company  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", default=[
        ROOT / "data/external/polyfiqa_easy_july/test-00000-of-00001.parquet",
        ROOT / "data/external/polyfiqa_expert_july/test-00000-of-00001.parquet",
    ])
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_graph_to_reference_v1",
    )
    args = parser.parse_args()

    accepted = []
    reasons = Counter()
    family_counts = Counter()
    roundtrip_failures = []
    source_rows = 0
    for path in args.inputs:
        frame = pd.read_parquet(path)
        required = {"task_id", "query", "question", "answer"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path}: missing historical reference columns")
        for source_index, row in frame.iterrows():
            source_rows += 1
            graph = build_fact_graph(row.task_id, row.query, row.question)
            record, reason = delexicalize_reference(
                row.task_id, row.answer, row.query, graph,
            )
            reasons[reason] += 1
            if record is None:
                continue
            restored = restore_reference(record, graph, row.task_id)
            if restored != record["core"]:
                roundtrip_failures.append({
                    "source": str(path), "source_index": int(source_index),
                    "task_id": row.task_id,
                })
                continue
            family_counts[graph.family] += 1
            accepted.append({
                "source": str(path), "source_index": int(source_index),
                "company": task_company(row.task_id),
                "plan": realization_plan(record, graph),
                "target_template": record["template"],
                "slots": record["slots"],
                "reference_core": record["core"],
            })
    if roundtrip_failures:
        raise AssertionError(
            f"{len(roundtrip_failures)} accepted records failed exact restoration"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in accepted
    ))
    summary = {
        "source_rows": source_rows,
        "accepted_rows": len(accepted),
        "acceptance_rate": len(accepted) / source_rows if source_rows else 0.0,
        "reasons": dict(reasons),
        "family_counts": dict(family_counts),
        "company_counts": dict(Counter(row["company"] for row in accepted)),
        "fact_only_rows": sum(
            not any(slot["source"] != "fact" for slot in row["slots"])
            for row in accepted
        ),
        "oracle_copy_plan_rows": sum(
            any(slot["source"] != "fact" for slot in row["slots"])
            for row in accepted
        ),
        "query_transform_rows": sum(
            any(slot["source"] == "query_transform" for slot in row["slots"])
            for row in accepted
        ),
        "roundtrip_failures": 0,
        "current_competition_or_public_test_answers_used": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
