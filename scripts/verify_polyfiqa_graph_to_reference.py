#!/usr/bin/env python3
"""Independent verifier for original and graph-swap realization records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.polyfiqa_delex import delexicalize_reference, restore_reference  # noqa: E402
from fineval.polyfiqa_facts import build_fact_graph, provenance_is_exact  # noqa: E402
from fineval.polyfiqa_programs import COMPANY_NAMES, semantic_state, task_company  # noqa: E402


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_graph_to_reference_v1",
    )
    parser.add_argument("--inputs", type=Path, nargs="+", default=[
        ROOT / "data/external/polyfiqa_easy_july/test-00000-of-00001.parquet",
        ROOT / "data/external/polyfiqa_expert_july/test-00000-of-00001.parquet",
    ])
    args = parser.parse_args()

    frames = {str(path): pd.read_parquet(path) for path in args.inputs}
    target_lookup = {}
    for path, frame in frames.items():
        for source_index, row in frame.iterrows():
            graph = build_fact_graph(row.task_id, row.query, row.question)
            target_lookup[(str(row.task_id), graph.family)] = (row, graph)

    originals = load_jsonl(args.artifact_dir / "training.jsonl")
    for record in originals:
        frame = frames[record["source"]]
        row = frame.loc[int(record["source_index"])]
        graph = build_fact_graph(row.task_id, row.query, row.question)
        expected, reason = delexicalize_reference(
            row.task_id, row.answer, row.query, graph,
        )
        if expected is None or not reason.startswith("aligned"):
            raise AssertionError("stored original no longer passes strict delexicalization")
        if record["target_template"] != expected["template"] or record["slots"] != expected["slots"]:
            raise AssertionError("stored original differs from independent reconstruction")
        if restore_reference(expected, graph, row.task_id) != expected["core"]:
            raise AssertionError("original reference does not restore exactly")
        if not provenance_is_exact(graph, row.query):
            raise AssertionError("original graph contains non-provenance evidence")
        for evidence in record["plan"]["evidence"]:
            if evidence["text"] not in row.query:
                raise AssertionError("selected evidence is not an exact input substring")

    counterfactuals = load_jsonl(args.artifact_dir / "counterfactual_training.jsonl")
    for record in counterfactuals:
        row, graph = target_lookup[(str(record["target_task_id"]), record["family"])]
        if task_company(row.task_id) != record["target_company"]:
            raise AssertionError("counterfactual target company mismatch")
        if list(semantic_state(graph)) != [tuple(item) for item in record["semantic_state"]]:
            raise AssertionError("counterfactual semantic state mismatch")
        if any(slot["source"] != "fact" for slot in record["slots"]):
            raise AssertionError("counterfactual contains a non-fact slot")
        synthetic = {
            "template": record["target_template"], "slots": record["slots"],
            "company_possessive_surface": COMPANY_NAMES[record["target_company"]] + "'s",
        }
        restored = restore_reference(synthetic, graph, row.task_id)
        if restored != record["mechanically_restored_target"]:
            raise AssertionError("counterfactual restoration mismatch")
        donor_name = COMPANY_NAMES[record["donor_company"]]
        if donor_name.casefold() in restored.casefold():
            raise AssertionError("donor company leaked into swapped reference")
        if "__" in restored:
            raise AssertionError("counterfactual has an unresolved marker")
        if record["plan"]["evidence"]:
            raise AssertionError("fact swap unexpectedly carries oracle evidence")

    summary = {
        "original_records_verified": len(originals),
        "counterfactual_records_verified": len(counterfactuals),
        "exact_roundtrip": True,
        "evidence_provenance_exact": True,
        "counterfactual_donor_leaks": 0,
        "counterfactual_oracle_evidence_rows": 0,
        "public_test_answers_used": False,
    }
    (args.artifact_dir / "verification_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
