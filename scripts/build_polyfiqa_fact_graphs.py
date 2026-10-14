#!/usr/bin/env python3
"""Build and audit typed fact graphs for a PolyFiQA packet."""

from argparse import ArgumentParser
from pathlib import Path
from collections import Counter
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.polyfiqa_facts import build_fact_graph, provenance_is_exact  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from predict_polyfiqa_hybrid import ASIAN_NUMBERS, input_grounded_task_id  # noqa: E402


def load(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.DataFrame(json.loads(line) for line in path.read_text().splitlines() if line.strip())


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "data/external/finmmeval_task2/task2_test_questions.jsonl",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/polyfiqa_fact_graphs/facts.jsonl",
    )
    args = parser.parse_args()
    frame = load(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    family_counts: Counter[str] = Counter()
    numeric_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    exact = 0
    with args.output.open("w") as handle:
        for position, (_, row) in enumerate(frame.iterrows()):
            internal_id = input_grounded_task_id(row)
            graph = build_fact_graph(
                internal_id, row.query, row.question, ASIAN_NUMBERS.get(internal_id),
            )
            valid_provenance = provenance_is_exact(graph, row.query)
            exact += valid_provenance
            family_counts[graph.family] += 1
            numeric_counts[graph.family] += bool(graph.numeric)
            error_counts[graph.family] += bool(graph.errors)
            record = {
                "row_index": position,
                "submission_task_id": str(row.task_id),
                "question": row.question,
                "provenance_exact": valid_provenance,
                **graph.to_dict(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "rows": len(frame),
        "provenance_exact_rows": exact,
        "family_counts": dict(sorted(family_counts.items())),
        "numeric_graph_rows": dict(sorted(numeric_counts.items())),
        "rows_with_errors": dict(sorted(error_counts.items())),
    }
    (args.output.parent / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
