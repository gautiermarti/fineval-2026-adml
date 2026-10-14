#!/usr/bin/env python3
"""Nested evaluation of family-specific executable PolyFiQA compilers.

The historical target reference supplies the fact-slot signature, so this is a
realizer diagnostic rather than an end-to-end competition score.  Template
selection for each outer held company is input-only.  A family-specific ranker
is enabled only by inner leave-company-out evidence from the remaining
companies; otherwise the state-conditioned MBR medoid is retained.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_polyfiqa_reference_compiler import (  # noqa: E402
    fact_signature,
    item_score,
    load_jsonl,
    pair_text,
    restore_donor,
    select_mbr,
    semantic_state,
)
from evaluate_polyfiqa_numeric_planner import (  # noqa: E402
    build_question_lookup,
    company as planner_company,
    fact_only as planner_fact_only,
    fact_threshold_prediction,
    fit_planner,
    score_plan_record,
)
from fineval.polyfiqa_facts import build_fact_graph  # noqa: E402
from fineval.polyfiqa_family_gate import (  # noqa: E402
    choose_family_route,
    exact_cluster_bootstrap,
)


COMPANIES = ("HON", "JNJ", "MSFT", "UVV")


class GraphStore:
    def __init__(self) -> None:
        self.frames: dict[str, pd.DataFrame] = {}

    def get(self, record: dict):
        source = str(record["source"])
        if source not in self.frames:
            self.frames[source] = pd.read_parquet(source)
        row = self.frames[source].iloc[int(record["source_index"])]
        return build_fact_graph(row.task_id, row.query, row.question)


def compatible_donors(target: dict, training: list[dict]) -> list[dict]:
    signature = fact_signature(target)
    if signature is None:
        return []
    return [
        donor for donor in training
        if donor["plan"]["family"] == target["plan"]["family"]
        and fact_signature(donor) == signature
    ]


def pair_signature(record: dict) -> frozenset[tuple[str, int]] | None:
    signature = fact_signature(record)
    if signature is None:
        return None
    return frozenset((str(metric), int(index)) for metric, index, _kind in signature)


def planned_donors(
    target: dict,
    training: list[dict],
    planned_signature: frozenset[tuple[str, int]],
) -> list[dict]:
    """Return templates compatible with an input-only unordered fact plan."""
    return [
        donor for donor in training
        if donor["plan"]["family"] == target["plan"]["family"]
        and pair_signature(donor) == planned_signature
    ]


def state_medoid(target: dict, donors: list[dict]) -> dict | None:
    exact_state = [
        donor for donor in donors
        if semantic_state(donor) == semantic_state(target)
    ]
    state_companies = {donor["company"] for donor in exact_state}
    cohort = (
        exact_state
        if len(exact_state) >= 3 and len(state_companies) >= 2
        else donors
    )
    return select_mbr(cohort)


def fit_family_rankers(training: list[dict], graphs: GraphStore) -> dict[str, dict]:
    """Fit one group-balanced utility model per family."""
    examples: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    fact_rows = [row for row in training if fact_signature(row) is not None]
    for target in fact_rows:
        family = str(target["plan"]["family"])
        donors = [
            donor for donor in fact_rows
            if donor["company"] != target["company"]
            and donor["plan"]["family"] == family
            and fact_signature(donor) == fact_signature(target)
        ]
        if not donors:
            continue
        graph = graphs.get(target)
        valid = []
        for donor in donors:
            try:
                prediction = restore_donor(donor, target, graph)
            except (KeyError, ValueError):
                continue
            valid.append((donor, item_score(target["reference_core"], prediction)))
        if not valid:
            continue
        group_weight = 1.0 / len(valid)
        examples[family].extend(
            (pair_text(target, donor), score, group_weight)
            for donor, score in valid
        )

    fitted = {}
    for family, rows in examples.items():
        if len(rows) < 8:
            continue
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            max_features=30_000, sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform([row[0] for row in rows])
        model = Ridge(alpha=5.0).fit(
            matrix,
            np.asarray([row[1] for row in rows]),
            sample_weight=np.asarray([row[2] for row in rows]),
        )
        fitted[family] = {
            "vectorizer": vectorizer,
            "model": model,
            "training_pairs": len(rows),
        }
    return fitted


def evaluate_record(
    target: dict,
    training: list[dict],
    rankers: dict[str, dict],
    graphs: GraphStore,
) -> dict | None:
    donors = compatible_donors(target, training)
    if not donors:
        return None
    graph = graphs.get(target)
    baseline_donor = state_medoid(target, donors)
    if baseline_donor is None:
        return None
    baseline_prediction = restore_donor(baseline_donor, target, graph)
    baseline_score = item_score(target["reference_core"], baseline_prediction)

    family = str(target["plan"]["family"])
    ranked_donor = baseline_donor
    bundle = rankers.get(family)
    if bundle is not None:
        utilities = bundle["model"].predict(bundle["vectorizer"].transform([
            pair_text(target, donor) for donor in donors
        ]))
        ranked_donor = donors[int(utilities.argmax())]
    ranker_prediction = restore_donor(ranked_donor, target, graph)
    ranker_score = item_score(target["reference_core"], ranker_prediction)
    return {
        "task_id": target["plan"]["task_id"],
        "company": target["company"],
        "family": family,
        "compatible_templates": len(donors),
        "baseline_template": baseline_donor["target_template"],
        "baseline_prediction": baseline_prediction,
        "baseline_score": baseline_score,
        "ranker_template": ranked_donor["target_template"],
        "ranker_prediction": ranker_prediction,
        "ranker_score": ranker_score,
        "reference": target["reference_core"],
    }


def evaluate_planned_record(
    target: dict,
    training: list[dict],
    rankers: dict[str, dict],
    graphs: GraphStore,
    planned_signature: frozenset[tuple[str, int]],
) -> dict | None:
    """Compile using only the planner's predicted fact set."""
    donors = planned_donors(target, training, planned_signature)
    if not donors:
        return None
    graph = graphs.get(target)
    baseline_donor = state_medoid(target, donors)
    if baseline_donor is None:
        return None
    baseline_prediction = restore_donor(baseline_donor, target, graph)
    family = str(target["plan"]["family"])
    ranked_donor = baseline_donor
    bundle = rankers.get(family)
    if bundle is not None:
        utilities = bundle["model"].predict(bundle["vectorizer"].transform([
            pair_text(target, donor) for donor in donors
        ]))
        ranked_donor = donors[int(utilities.argmax())]
    ranker_prediction = restore_donor(ranked_donor, target, graph)
    return {
        "task_id": target["plan"]["task_id"],
        "company": target["company"],
        "family": family,
        "planned_signature": sorted([list(value) for value in planned_signature]),
        "compatible_templates": len(donors),
        "baseline_prediction": baseline_prediction,
        "baseline_score": item_score(target["reference_core"], baseline_prediction),
        "ranker_prediction": ranker_prediction,
        "ranker_score": item_score(target["reference_core"], ranker_prediction),
        "reference": target["reference_core"],
    }


def inner_evidence(
    outer_training: list[dict], graphs: GraphStore,
) -> list[dict]:
    evidence = []
    available_companies = sorted({row["company"] for row in outer_training})
    for inner_company in available_companies:
        inner_training = [
            row for row in outer_training if row["company"] != inner_company
        ]
        inner_evaluation = [
            row for row in outer_training
            if row["company"] == inner_company and fact_signature(row) is not None
        ]
        rankers = fit_family_rankers(inner_training, graphs)
        for target in inner_evaluation:
            result = evaluate_record(target, inner_training, rankers, graphs)
            if result is not None:
                evidence.append(result)
    return evidence


def inner_planned_evidence(
    held_company: str,
    compiler_outer_training: list[dict],
    planner_corpus_dir: Path,
    question_lookup: dict,
    graphs: GraphStore,
) -> list[dict]:
    """Evaluate ranker routes behind a fully input-only inner planner."""
    planner_outer_training = [
        row for row in load_jsonl(
            planner_corpus_dir / "loco" / held_company / "train.jsonl"
        )
        if planner_fact_only(row)
    ]
    evidence = []
    available_companies = sorted({
        planner_company(row) for row in planner_outer_training
        if row.get("source_kind") == "historical"
    })
    for inner_company in available_companies:
        planner_inner_evaluation = [
            row for row in planner_outer_training
            if row.get("source_kind") == "historical"
            and planner_company(row) == inner_company
        ]
        planner_inner_training = []
        for row in planner_outer_training:
            if row.get("source_kind") == "historical":
                keep = planner_company(row) != inner_company
            else:
                keep = (
                    row.get("donor_company") != inner_company
                    and row.get("target_company") != inner_company
                )
            if keep:
                planner_inner_training.append(row)
        if not planner_inner_training or not planner_inner_evaluation:
            continue

        planner_bundle = fit_planner(planner_inner_training, question_lookup)
        compiler_inner_training = [
            row for row in compiler_outer_training
            if row["company"] != inner_company
        ]
        rankers = fit_family_rankers(compiler_inner_training, graphs)
        for target in planner_inner_evaluation:
            scored = score_plan_record(target, planner_bundle, question_lookup)
            planned_signature = fact_threshold_prediction(scored, 0.5)
            if scored["numeric_answer_probability"] < 0.5:
                planned_signature = set()
            result = evaluate_planned_record(
                target,
                compiler_inner_training,
                rankers,
                graphs,
                frozenset(planned_signature),
            )
            if result is not None:
                evidence.append(result)
    return evidence


def mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_graph_to_reference_v5",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_family_compiler_v1",
    )
    parser.add_argument(
        "--planner-predictions", type=Path,
        default=ROOT / "artifacts/polyfiqa_numeric_planner_v2/predictions.jsonl",
    )
    parser.add_argument(
        "--planner-corpus-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_graph_to_reference_v2",
    )
    args = parser.parse_args()
    graphs = GraphStore()
    planner_lookup = {
        (row["held_company"], row["task_id"], row["family"]): row
        for row in load_jsonl(args.planner_predictions)
    }
    question_lookup = build_question_lookup(
        load_jsonl(args.planner_corpus_dir / "training.jsonl")
    )
    all_rows = []
    all_planned_rows = []
    fact_only_evaluation_rows = 0
    fold_summaries = {}

    for held_company in COMPANIES:
        split = args.corpus_dir / "loco" / held_company
        training = [row for row in load_jsonl(split / "train.jsonl") if "source" in row]
        evaluation = load_jsonl(split / "evaluation.jsonl")
        if any(row["company"] == held_company for row in training):
            raise AssertionError(f"outer company leaked into {held_company} training")

        inner = inner_evidence(training, graphs)
        gates = {
            family: choose_family_route([
                row for row in inner if row["family"] == family
            ])
            for family in sorted({row["family"] for row in inner})
        }
        inner_planned = inner_planned_evidence(
            held_company,
            training,
            args.planner_corpus_dir,
            question_lookup,
            graphs,
        )
        planned_gates = {
            family: choose_family_route([
                row for row in inner_planned if row["family"] == family
            ])
            for family in sorted({row["family"] for row in inner_planned})
        }
        rankers = fit_family_rankers(training, graphs)
        fold_rows = []
        planned_fold_rows = []
        for target in evaluation:
            if fact_signature(target) is None:
                continue
            fact_only_evaluation_rows += 1
            result = evaluate_record(target, training, rankers, graphs)
            if result is not None:
                route = gates.get(result["family"], {"route": "state_medoid"})["route"]
                result["held_company"] = held_company
                result["nested_route"] = route
                result["nested_prediction"] = (
                    result["ranker_prediction"]
                    if route == "family_ranker" else result["baseline_prediction"]
                )
                result["nested_score"] = (
                    result["ranker_score"]
                    if route == "family_ranker" else result["baseline_score"]
                )
                fold_rows.append(result)
                all_rows.append(result)

            key = (
                held_company, target["plan"]["task_id"],
                target["plan"]["family"],
            )
            planner = planner_lookup.get(key)
            if planner is not None:
                planned_signature = frozenset(
                    (str(metric), int(index))
                    for metric, index in planner["two_stage"]
                )
                planned = evaluate_planned_record(
                    target, training, rankers, graphs, planned_signature,
                )
                if planned is not None:
                    route = planned_gates.get(
                        planned["family"], {"route": "state_medoid"}
                    )["route"]
                    planned["held_company"] = held_company
                    planned["planner_signature_exact"] = (
                        planned_signature == pair_signature(target)
                    )
                    planned["nested_route"] = route
                    planned["nested_prediction"] = (
                        planned["ranker_prediction"]
                        if route == "family_ranker"
                        else planned["baseline_prediction"]
                    )
                    planned["nested_score"] = (
                        planned["ranker_score"]
                        if route == "family_ranker" else planned["baseline_score"]
                    )
                    planned_fold_rows.append(planned)
                    all_planned_rows.append(planned)
        fold_summaries[held_company] = {
            "covered_rows": len(fold_rows),
            "baseline_rouge": mean(fold_rows, "baseline_score"),
            "ranker_rouge": mean(fold_rows, "ranker_score"),
            "nested_rouge": mean(fold_rows, "nested_score"),
            "planned_compiler_covered_rows": len(planned_fold_rows),
            "planned_nested_rouge_covered": mean(
                planned_fold_rows, "nested_score"
            ),
            "gates": gates,
            "input_only_planner_gates": planned_gates,
        }

    family_summaries = {}
    for family in sorted({row["family"] for row in all_rows}):
        rows = [row for row in all_rows if row["family"] == family]
        family_summaries[family] = {
            "covered_rows": len(rows),
            "baseline_rouge": mean(rows, "baseline_score"),
            "ranker_rouge": mean(rows, "ranker_score"),
            "nested_rouge": mean(rows, "nested_score"),
            "nested_ranker_rows": sum(
                row["nested_route"] == "family_ranker" for row in rows
            ),
            "company_mean_nested_gains": {
                company: mean([
                    {
                        "delta": row["nested_score"] - row["baseline_score"]
                    }
                    for row in rows if row["held_company"] == company
                ], "delta")
                for company in sorted({row["held_company"] for row in rows})
            },
        }

    planned_family_summaries = {}
    for family in sorted({row["family"] for row in all_planned_rows}):
        rows = [row for row in all_planned_rows if row["family"] == family]
        planned_family_summaries[family] = {
            "covered_rows": len(rows),
            "planner_signature_exact_rows": sum(
                row["planner_signature_exact"] for row in rows
            ),
            "planner_signature_exact_rate": sum(
                row["planner_signature_exact"] for row in rows
            ) / len(rows),
            "baseline_rouge": mean(rows, "baseline_score"),
            "ranker_rouge": mean(rows, "ranker_score"),
            "nested_rouge": mean(rows, "nested_score"),
        }

    summary = {
        "evaluation": "nested leave-one-company-out historical originals",
        "selection": "family-specific utility ranker with inner-company conservative gate",
        "covered_fact_only_rows": len(all_rows),
        "baseline_rouge": mean(all_rows, "baseline_score"),
        "ungated_family_ranker_rouge": mean(all_rows, "ranker_score"),
        "nested_family_hybrid_rouge": mean(all_rows, "nested_score"),
        "nested_gain": mean(all_rows, "nested_score") - mean(all_rows, "baseline_score"),
        "company_cluster_bootstrap": exact_cluster_bootstrap([
            {
                "company": row["held_company"],
                "delta": row["nested_score"] - row["baseline_score"],
            }
            for row in all_rows
        ], cluster_key="company", delta_key="delta"),
        "folds": fold_summaries,
        "by_family": family_summaries,
        "input_only_planner_replay": {
            "planner_artifact": str(args.planner_predictions),
            "historical_fact_only_rows": fact_only_evaluation_rows,
            "compiled_rows": len(all_planned_rows),
            "coverage": len(all_planned_rows) / fact_only_evaluation_rows,
            "planner_signature_exact_rows_on_compiled": sum(
                row["planner_signature_exact"] for row in all_planned_rows
            ),
            "planner_signature_exact_rate_on_compiled": sum(
                row["planner_signature_exact"] for row in all_planned_rows
            ) / len(all_planned_rows),
            "nested_rouge_covered": mean(all_planned_rows, "nested_score"),
            "baseline_rouge_covered": mean(all_planned_rows, "baseline_score"),
            "nested_gain_covered": (
                mean(all_planned_rows, "nested_score")
                - mean(all_planned_rows, "baseline_score")
            ),
            "company_cluster_bootstrap": exact_cluster_bootstrap([
                {
                    "company": row["held_company"],
                    "delta": row["nested_score"] - row["baseline_score"],
                }
                for row in all_planned_rows
            ], cluster_key="company", delta_key="delta"),
            "nested_rouge_all_fact_only_zero_for_abstentions": (
                sum(row["nested_score"] for row in all_planned_rows)
                / fact_only_evaluation_rows
            ),
            "by_family": planned_family_summaries,
            "historical_references_used_at_inference": False,
        },
        "restoration_errors": dict(Counter()),
        "realizer_only_oracle_slot_signature": True,
        "end_to_end_competition_claim": False,
        "current_or_public_test_answers_used": False,
        "counterfactual_swaps_used_for_ranker": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows
    ))
    (args.output_dir / "planned_predictions.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in all_planned_rows
    ))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
