#!/usr/bin/env python3
"""Evaluate a leave-company-out executable reference-template compiler.

This is deliberately a realizer diagnostic.  Evaluation records contain the
historical reference's requested fact-slot signature, so the result is not an
end-to-end competition score.  Template selection itself is labels-free for
the held company: it maximizes expected delexicalized ROUGE only against
compatible templates from the other companies, then restores target facts
mechanically.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.metrics import rouge1_f1  # noqa: E402
from fineval.polyfiqa_delex import restore_reference  # noqa: E402
from fineval.polyfiqa_facts import build_fact_graph  # noqa: E402
from fineval.polyfiqa_programs import COMPANY_NAMES  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fact_signature(record: dict) -> tuple | None:
    """Return the ordered executable fact-slot signature, or ``None``."""
    signature = []
    for slot in record["slots"]:
        if slot["source"] != "fact":
            return None
        signature.append((slot["metric"], int(slot["index"]), slot["kind"]))
    return tuple(signature)


def semantic_state(record: dict) -> tuple:
    """Represent only input-derived sign and trend state, never reference text."""
    state = []
    for fact in record["plan"]["numeric"]:
        values = [float(item["value"]) for item in fact["values"]]
        if not values:
            continue
        sign = "positive" if values[0] > 0 else "negative" if values[0] < 0 else "zero"
        trend = "single"
        if len(values) >= 2:
            trend = "up" if values[0] > values[1] else "down" if values[0] < values[1] else "flat"
        state.append((fact["metric"], sign, trend))
    return tuple(state)


def item_score(reference: str, prediction: str) -> float:
    return rouge1_f1([reference], [prediction])


def expected_utility(candidate: dict, cohort: list[dict]) -> float:
    """Mean template ROUGE against independent compatible training records."""
    peers = [row for row in cohort if row is not candidate]
    if not peers:
        return 0.0
    return sum(
        item_score(row["target_template"], candidate["target_template"])
        for row in peers
    ) / len(peers)


def select_mbr(cohort: list[dict]) -> dict | None:
    if not cohort:
        return None
    return max(
        cohort,
        key=lambda row: (
            expected_utility(row, cohort),
            -len(row["target_template"]),
            row["target_template"],
        ),
    )


def pair_text(target: dict, donor: dict) -> str:
    """Input-only target/candidate representation for utility prediction."""
    target_state = semantic_state(target)
    donor_state = semantic_state(donor)
    fields = [
        f"family={target['plan']['family']}",
        f"tier={target['plan']['tier']}",
        f"state_match={int(target_state == donor_state)}",
    ]
    fields.extend(
        f"target_state={metric}:{sign}:{trend}"
        for metric, sign, trend in target_state
    )
    fields.extend(
        f"donor_state={metric}:{sign}:{trend}"
        for metric, sign, trend in donor_state
    )
    for fact in target["plan"]["numeric"]:
        values = [float(item["value"]) for item in fact["values"]]
        if values:
            magnitude = int(math.floor(math.log10(max(abs(values[0]), 1e-12))))
            fields.append(f"target_magnitude={fact['metric']}:{magnitude}")
    for slot in donor["slots"]:
        style = slot.get("style") or {}
        fields.append(
            "donor_slot="
            f"{slot.get('metric')}:{slot.get('kind')}:{style.get('suffix', '')}:"
            f"{style.get('decimals', '')}"
        )
    return " ".join(fields) + " TEMPLATE " + donor["target_template"]


def restore_donor(donor: dict, target: dict, graph) -> str:
    return restore_reference(
        {
            "template": donor["target_template"],
            "slots": donor["slots"],
            "company_possessive_surface": COMPANY_NAMES[target["company"]] + "'s",
        },
        graph,
        target["plan"]["task_id"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_graph_to_reference_v5",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_reference_compiler_v1",
    )
    args = parser.parse_args()

    rows = []
    frame_cache: dict[str, pd.DataFrame] = {}

    def graph_for(record: dict):
        source = record["source"]
        if source not in frame_cache:
            frame_cache[source] = pd.read_parquet(source)
        source_row = frame_cache[source].iloc[int(record["source_index"])]
        return build_fact_graph(source_row.task_id, source_row.query, source_row.question)

    for held_company in ("HON", "JNJ", "MSFT", "UVV"):
        split = args.corpus_dir / "loco" / held_company
        # Counterfactual swaps duplicate donor language and would give popular
        # templates artificial votes.  Use historical originals only.
        training = [row for row in load_jsonl(split / "train.jsonl") if "source" in row]
        evaluation = load_jsonl(split / "evaluation.jsonl")

        ranker_texts = []
        ranker_targets = []
        ranker_weights = []
        for pseudo_target in training:
            signature = fact_signature(pseudo_target)
            if signature is None:
                continue
            pseudo_donors = [
                row for row in training
                if row["company"] != pseudo_target["company"]
                and row["plan"]["family"] == pseudo_target["plan"]["family"]
                and fact_signature(row) == signature
            ]
            if not pseudo_donors:
                continue
            pseudo_graph = graph_for(pseudo_target)
            weight = 1.0 / len(pseudo_donors)
            for donor in pseudo_donors:
                try:
                    prediction = restore_donor(donor, pseudo_target, pseudo_graph)
                except (KeyError, ValueError):
                    continue
                ranker_texts.append(pair_text(pseudo_target, donor))
                ranker_targets.append(item_score(pseudo_target["reference_core"], prediction))
                ranker_weights.append(weight)
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            max_features=30_000, sublinear_tf=True,
        )
        ranker = Ridge(alpha=5.0)
        ranker_matrix = vectorizer.fit_transform(ranker_texts)
        ranker.fit(ranker_matrix, ranker_targets, sample_weight=ranker_weights)

        for target in evaluation:
            signature = fact_signature(target)
            target_state = semantic_state(target)
            compatible = [
                row for row in training
                if row["plan"]["family"] == target["plan"]["family"]
                and fact_signature(row) == signature
            ] if signature is not None else []
            exact_state = [row for row in compatible if semantic_state(row) == target_state]
            # State conditioning is allowed only when it has independent
            # support; otherwise the broader signature medoid is lower variance.
            state_companies = {row["company"] for row in exact_state}
            cohort = exact_state if len(exact_state) >= 3 and len(state_companies) >= 2 else compatible
            selected = select_mbr(cohort)

            graph = graph_for(target)

            prediction = ""
            selected_score = 0.0
            selected_template = None
            restoration_error = None
            if selected is not None:
                try:
                    prediction = restore_donor(selected, target, graph)
                    selected_score = item_score(target["reference_core"], prediction)
                    selected_template = selected["target_template"]
                except (KeyError, ValueError) as exc:
                    restoration_error = str(exc)

            ranked = None
            ranker_prediction = ""
            ranker_selected_score = 0.0
            if compatible:
                utilities = ranker.predict(vectorizer.transform([
                    pair_text(target, donor) for donor in compatible
                ]))
                ranked = compatible[int(utilities.argmax())]
                try:
                    ranker_prediction = restore_donor(ranked, target, graph)
                    ranker_selected_score = item_score(
                        target["reference_core"], ranker_prediction,
                    )
                except (KeyError, ValueError):
                    ranked = None

            oracle_score = 0.0
            oracle_template = None
            for donor in compatible:
                try:
                    candidate = restore_donor(donor, target, graph)
                except (KeyError, ValueError):
                    continue
                score = item_score(target["reference_core"], candidate)
                if score > oracle_score:
                    oracle_score = score
                    oracle_template = donor["target_template"]

            rows.append({
                "task_id": target["plan"]["task_id"],
                "held_company": held_company,
                "family": target["plan"]["family"],
                "fact_only": signature is not None,
                "semantic_state": target_state,
                "compatible_templates": len(compatible),
                "state_templates": len(exact_state),
                "state_conditioned": cohort is exact_state,
                "selected_template": selected_template,
                "prediction": prediction,
                "reference": target["reference_core"],
                "selected_score": selected_score,
                "ranker_template": ranked["target_template"] if ranked else None,
                "ranker_prediction": ranker_prediction,
                "ranker_selected_score": ranker_selected_score,
                "oracle_score": oracle_score,
                "oracle_template": oracle_template,
                "restoration_error": restoration_error,
            })

    fact_rows = [row for row in rows if row["fact_only"]]
    covered = [row for row in fact_rows if row["compatible_templates"] > 0]
    family_summary = {}
    for family in sorted({row["family"] for row in fact_rows}):
        subset = [row for row in fact_rows if row["family"] == family]
        subset_covered = [row for row in subset if row["compatible_templates"] > 0]
        family_summary[family] = {
            "rows": len(subset),
            "covered": len(subset_covered),
            "coverage": len(subset_covered) / len(subset) if subset else 0.0,
            "selected_rouge_all": sum(row["selected_score"] for row in subset) / len(subset),
            "selected_rouge_covered": (
                sum(row["selected_score"] for row in subset_covered) / len(subset_covered)
                if subset_covered else 0.0
            ),
            "ranker_rouge_all": sum(row["ranker_selected_score"] for row in subset) / len(subset),
            "ranker_rouge_covered": (
                sum(row["ranker_selected_score"] for row in subset_covered) / len(subset_covered)
                if subset_covered else 0.0
            ),
            "oracle_rouge_all": sum(row["oracle_score"] for row in subset) / len(subset),
        }
    summary = {
        "evaluation": "leave-one-company-out historical originals",
        "selection": "input-state-conditioned expected delexicalized ROUGE medoid",
        "historical_evaluation_rows": len(rows),
        "fact_only_rows": len(fact_rows),
        "covered_fact_only_rows": len(covered),
        "fact_only_coverage": len(covered) / len(fact_rows),
        "selected_rouge_all_fact_only": sum(row["selected_score"] for row in fact_rows) / len(fact_rows),
        "selected_rouge_covered_fact_only": sum(row["selected_score"] for row in covered) / len(covered),
        "ranker_rouge_all_fact_only": sum(row["ranker_selected_score"] for row in fact_rows) / len(fact_rows),
        "ranker_rouge_covered_fact_only": sum(row["ranker_selected_score"] for row in covered) / len(covered),
        "oracle_rouge_all_fact_only": sum(row["oracle_score"] for row in fact_rows) / len(fact_rows),
        "state_conditioned_rows": sum(row["state_conditioned"] for row in fact_rows),
        "restoration_errors": dict(Counter(
            row["restoration_error"] for row in rows if row["restoration_error"]
        )),
        "by_family": family_summary,
        "realizer_only_oracle_slot_signature": True,
        "end_to_end_competition_claim": False,
        "public_test_answers_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    ))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
