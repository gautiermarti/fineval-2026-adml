#!/usr/bin/env python3
"""Rebuild the frozen Task 3 sparse-edit lattice on all historical packets."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_polyfiqa_hidden_overlap import fingerprint  # noqa: E402
from build_polyfiqa_candidate_lattice import (  # noqa: E402
    candidate_valid, easy_candidates, expert_candidates,
)
from fineval.metrics import rouge1_f1  # noqa: E402
from fineval.parsing import parse_polyfiqa_query  # noqa: E402
from fineval.polyfiqa_edits import sparse_edits  # noqa: E402
from fineval.polyfiqa_facts import build_fact_graph  # noqa: E402
from predict_polyfiqa_hybrid import (  # noqa: E402
    ASIAN_NUMBERS, cap_generated_answer, input_grounded_task_id,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def item_score(reference: str, prediction: str) -> float:
    return rouge1_f1([reference], [prediction])


def partition_name(packet_fingerprint: str, discovery: set[str]) -> str:
    return "late_discovery" if packet_fingerprint in discovery else "early_confirmation"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--easy", type=Path,
        default=ROOT / "data/external/polyfiqa_easy_july/test-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--expert", type=Path,
        default=ROOT / "data/external/polyfiqa_expert_july/test-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--easy-baseline", type=Path,
        default=ROOT / "artifacts/polyfiqa_easy_historical_inputonly_v15/predictions.jsonl",
    )
    parser.add_argument(
        "--expert-baseline", type=Path,
        default=ROOT / "artifacts/polyfiqa_expert_historical_inputonly_v15/predictions.jsonl",
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
        default=ROOT / "artifacts/polyfiqa_historical_sparse_lattice_v1",
    )
    args = parser.parse_args()

    discovery_input = load_jsonl(args.discovery_input)
    discovery_lattice = load_jsonl(args.discovery_lattice)
    current_by_fingerprint = {}
    for lattice_row in discovery_lattice:
        current_row = discovery_input[int(lattice_row["row_index"])]
        packet_fingerprint = fingerprint(current_row["query"], current_row["question"])
        if packet_fingerprint in current_by_fingerprint:
            raise ValueError("discovery packet fingerprint is not unique")
        current_by_fingerprint[packet_fingerprint] = lattice_row
    discovery_fingerprints = set(current_by_fingerprint)

    output, audit, excluded = [], [], []
    global_index = 0
    aligned = 0
    regenerated_discovery_differences = 0
    for tier, data_path, baseline_path in (
        ("easy", args.easy, args.easy_baseline),
        ("expert", args.expert, args.expert_baseline),
    ):
        frame = pd.read_parquet(data_path)
        baselines = load_jsonl(baseline_path)
        if len(frame) != len(baselines):
            raise ValueError(f"{tier} input and baseline lengths differ")
        for local_index, ((_, row), baseline) in enumerate(
            zip(frame.iterrows(), baselines, strict=True)
        ):
            if str(row.task_id) != str(baseline["task_id"]):
                raise ValueError(f"{tier} baseline mismatch at {local_index}")
            row = row.copy()
            row["tier"] = tier
            internal_id = input_grounded_task_id(row)
            graph = build_fact_graph(
                internal_id, row.query, row.question, ASIAN_NUMBERS.get(internal_id),
            )
            if graph.tier != tier:
                raise AssertionError(f"tier routing failed for {row.task_id}")
            family = "revenue" if graph.family == "revenue_focus" else graph.family
            context = parse_polyfiqa_query(row.query)
            alternatives = (
                easy_candidates(graph, context, internal_id)
                if tier == "easy"
                else expert_candidates(graph, context, internal_id)
            )
            generated = sparse_edits(baseline["answer"], alternatives)
            valid = {}
            for name, answer in generated.items():
                capped = cap_generated_answer(answer)
                if candidate_valid(capped, row, internal_id):
                    valid[name] = capped
            packet_fingerprint = fingerprint(row.query, row.question)
            partition = partition_name(packet_fingerprint, discovery_fingerprints)
            if partition == "late_discovery":
                current = current_by_fingerprint[packet_fingerprint]
                if str(current["internal_id"]).split("_", 1)[0] != internal_id.split("_", 1)[0]:
                    raise AssertionError(
                        "discovery issuer mismatch: "
                        f"{tier}:{local_index} historical={internal_id} "
                        f"current={current['internal_id']}"
                    )
                if current["reference"] != str(row.answer):
                    raise AssertionError("discovery reference mismatch")
                if current["tier"] != tier or current["family"] != family:
                    raise AssertionError("discovery route mismatch")
                if current["candidates"] != valid:
                    # Historical source IDs expose the exact report date, while
                    # the released discovery packet uses anonymized IDs and the
                    # frozen input-only year heuristic.  Preserve the actual
                    # deployed late lattice and use the richer-ID regeneration
                    # only for genuinely earlier packets.
                    regenerated_discovery_differences += 1
                valid = dict(current["candidates"])
                scores = {
                    name: float(value) for name, value in current["scores"].items()
                }
                internal_id = str(current["internal_id"])
                aligned += 1
            else:
                scores = {
                    name: item_score(str(row.answer), answer)
                    for name, answer in valid.items()
                }
            if "baseline" not in valid:
                excluded.append({
                    "global_index": global_index, "tier": tier,
                    "local_index": local_index, "task_id": row.task_id,
                    "internal_id": internal_id, "family": family,
                    "partition": partition, "reason": "invalid_historical_baseline",
                })
                global_index += 1
                continue
            oracle = max(scores, key=lambda name: (scores[name], name == "baseline"))
            record = {
                "row_index": global_index, "tier_row_index": local_index,
                "task_id": str(row.task_id), "internal_id": internal_id,
                "tier": tier, "family": family, "partition": partition,
                "packet_fingerprint": packet_fingerprint,
                "question": str(row.question), "query": str(row.query),
                "reference": str(row.answer), "candidates": valid, "scores": scores,
            }
            output.append(record)
            baseline_score = scores["baseline"]
            audit.append({
                "row_index": global_index, "tier": tier, "task_id": row.task_id,
                "internal_id": internal_id, "family": family,
                "partition": partition, "valid_candidates": len(valid),
                "baseline_score": baseline_score, "oracle": oracle,
                "oracle_score": scores[oracle],
                "oracle_delta": scores[oracle] - baseline_score,
            })
            global_index += 1

    audit_frame = pd.DataFrame(audit)
    by_partition = {}
    for partition, values in audit_frame.groupby("partition", sort=True):
        by_partition[partition] = {
            "rows": len(values),
            "baseline_rouge": float(values.baseline_score.mean()),
            "oracle_rouge": float(values.oracle_score.mean()),
            "oracle_delta": float(values.oracle_delta.mean()),
            "oracle_improved_rows": int((values.oracle_delta > 0).sum()),
        }
    summary = {
        "source_rows": global_index, "valid_rows": len(output),
        "excluded_rows": len(excluded),
        "partition_counts": dict(Counter(row["partition"] for row in output)),
        "discovery_rows_exactly_aligned_to_frozen_lattice": aligned,
        "discovery_rows_whose_richer_historical_id_changes_regeneration": (
            regenerated_discovery_differences
        ),
        "baseline_rouge": float(audit_frame.baseline_score.mean()),
        "oracle_rouge": float(audit_frame.oracle_score.mean()),
        "oracle_delta": float(audit_frame.oracle_delta.mean()),
        "by_partition": by_partition,
        "selection_claim": False,
        "current_or_hidden_test_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "lattice.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in output
    ))
    audit_frame.to_csv(args.output_dir / "audit.csv", index=False)
    pd.DataFrame(excluded).to_csv(args.output_dir / "excluded.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
