#!/usr/bin/env python3
"""Evaluate evidence-entailment routing for Task 3 sparse candidates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.polyfiqa_family_gate import exact_cluster_bootstrap  # noqa: E402


COMPANIES = ("HON", "JNJ", "MSFT", "UVV")
ALPHAS = (10.0, 100.0, 1000.0, 10000.0)
GAIN_THRESHOLDS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04)
FIXED_MARGINS = (0.0, 0.01, 0.025, 0.05, 0.10, 0.15, 0.20)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def company(internal_id: str) -> str:
    value = str(internal_id).split("_", 1)[0]
    if value not in COMPANIES:
        raise ValueError(f"unexpected company: {value}")
    return value


def categorical(row: dict) -> dict[str, float]:
    operation, _, variant = row["operation"].partition(":")
    return {
        f"tier={row['tier']}": 1.0,
        f"family={row['family']}": 1.0,
        f"operation={operation}": 1.0,
        f"variant={variant or 'none'}": 1.0,
        f"operation_variant={row['operation']}": 1.0,
    }


def feature_block(
    examples: list[dict], entailment: np.ndarray,
    source: np.ndarray | None,
) -> np.ndarray:
    rows = []
    for row in examples:
        index = row["feature_index"]
        baseline = row["baseline_feature_index"]
        blocks = [entailment[index], entailment[index] - entailment[baseline]]
        if source is not None:
            # The first 22 source-MBR columns are the auditable scalar signals;
            # exclude the 768 dense embedding dimensions that failed previously.
            blocks.extend([source[index, :22], source[index, :22] - source[baseline, :22]])
        rows.append(np.concatenate(blocks))
    return np.stack(rows)


def fit_predict(
    train: list[dict], test: list[dict], entailment: np.ndarray,
    source: np.ndarray | None, alpha: float,
) -> np.ndarray:
    vectorizer = DictVectorizer(sparse=False)
    train_cat = vectorizer.fit_transform([categorical(row) for row in train])
    test_cat = vectorizer.transform([categorical(row) for row in test])
    train_matrix = np.concatenate([
        feature_block(train, entailment, source), train_cat,
    ], axis=1)
    test_matrix = np.concatenate([
        feature_block(test, entailment, source), test_cat,
    ], axis=1)
    row_counts = Counter((row["company"], row["row_index"]) for row in train)
    rows_per_company = Counter(key[0] for key in row_counts)
    weights = np.asarray([
        1.0 / (
            rows_per_company[row["company"]]
            * row_counts[(row["company"], row["row_index"])]
        )
        for row in train
    ])
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(
        train_matrix, np.asarray([row["delta"] for row in train]),
        ridge__sample_weight=weights,
    )
    return model.predict(test_matrix)


def selection_record(row: dict, operation: str, predicted_gain: float) -> dict:
    baseline = row["scores"]["baseline"]
    selected = row["scores"][operation]
    return {
        "row_index": row["row_index"], "task_id": row["task_id"],
        "held_company": row["company"], "tier": row["tier"],
        "family": row["family"], "selected": operation,
        "predicted_gain": float(predicted_gain),
        "baseline_score": float(baseline), "selected_score": float(selected),
        "delta": float(selected - baseline), "answer": row["answers"][operation],
    }


def select_predicted(
    row_records: list[dict], examples: list[dict], predicted: np.ndarray,
    threshold: float,
) -> list[dict]:
    by_row = defaultdict(list)
    for example, gain in zip(examples, predicted, strict=True):
        by_row[example["row_index"]].append((float(gain), example))
    selected = []
    for row in row_records:
        choices = [item for item in by_row[row["row_index"]] if item[0] >= threshold]
        if choices:
            predicted_gain, example = max(
                choices, key=lambda item: (item[0], -len(item[1]["answer"])),
            )
            operation = example["operation"]
        else:
            predicted_gain, operation = 0.0, "baseline"
        selected.append(selection_record(row, operation, predicted_gain))
    return selected


def select_fixed(row_records: list[dict], margin: float) -> list[dict]:
    selected = []
    for row in row_records:
        baseline_score = row["grounding"]["baseline"]
        operation = max(
            row["grounding"],
            key=lambda name: (
                row["grounding"][name], name == "baseline",
                -len(row["answers"][name]),
            ),
        )
        gain = row["grounding"][operation] - baseline_score
        if operation != "baseline" and gain < margin:
            operation, gain = "baseline", 0.0
        selected.append(selection_record(row, operation, gain))
    return selected


def mean_score(rows: list[dict]) -> float:
    return float(np.mean([row["selected_score"] for row in rows]))


def summarize(name: str, selected: list[dict], fold_rows: list[dict]) -> dict:
    baseline = float(np.mean([row["baseline_score"] for row in selected]))
    score = mean_score(selected)
    bootstrap = exact_cluster_bootstrap(
        selected, cluster_key="held_company", delta_key="delta",
    )
    result = {
        "evaluation": name, "rows": len(selected),
        "baseline_rouge": baseline, "selected_rouge": score,
        "delta": score - baseline,
        "changed_rows": sum(row["selected"] != "baseline" for row in selected),
        "improved_rows": sum(row["delta"] > 0 for row in selected),
        "harmed_rows": sum(row["delta"] < 0 for row in selected),
        "folds": fold_rows, "company_cluster_bootstrap": bootstrap,
        "current_or_hidden_test_used": False,
    }
    result["promotion_eligible"] = bool(
        result["delta"] > 0
        and sum(row["delta"] >= 0 for row in fold_rows) >= 3
        and sum(row["delta"] < -0.002 for row in fold_rows) <= 1
        and bootstrap["p_gain"] >= 0.90 and bootstrap["p05"] >= 0.0
    )
    return result


def fixed_variant(row_records: list[dict]) -> tuple[list[dict], list[dict]]:
    selected, folds = [], []
    for held in COMPANIES:
        training_rows = [row for row in row_records if row["company"] != held]
        held_rows = [row for row in row_records if row["company"] == held]
        choices = []
        for margin in FIXED_MARGINS:
            inner = select_fixed(training_rows, margin)
            choices.append((mean_score(inner), margin, inner))
        best_score = max(item[0] for item in choices)
        _, margin, inner = max(
            (item for item in choices if item[0] == best_score),
            key=lambda item: item[1],
        )
        outer = select_fixed(held_rows, margin)
        selected.extend(outer)
        folds.append({
            "held_company": held, "selected_margin": margin,
            "inner_score": best_score,
            "inner_delta": float(np.mean([row["delta"] for row in inner])),
            "inner_changed": sum(row["selected"] != "baseline" for row in inner),
            "rows": len(outer),
            "changed": sum(row["selected"] != "baseline" for row in outer),
            "baseline": float(np.mean([row["baseline_score"] for row in outer])),
            "selected": mean_score(outer),
            "delta": float(np.mean([row["delta"] for row in outer])),
        })
    return selected, folds


def nested_variant(
    row_records: list[dict], examples: list[dict], entailment: np.ndarray,
    source: np.ndarray | None,
) -> tuple[list[dict], list[dict]]:
    selected, folds = [], []
    for held in COMPANIES:
        outer_training_records = [row for row in row_records if row["company"] != held]
        outer_test_records = [row for row in row_records if row["company"] == held]
        options = []
        for alpha in ALPHAS:
            inner_examples, inner_predictions = [], []
            for inner_held in sorted(set(COMPANIES) - {held}):
                train = [
                    row for row in examples if row["company"] not in {held, inner_held}
                ]
                test = [row for row in examples if row["company"] == inner_held]
                inner_examples.extend(test)
                inner_predictions.extend(
                    fit_predict(train, test, entailment, source, alpha)
                )
            inner_predictions = np.asarray(inner_predictions)
            for threshold in GAIN_THRESHOLDS:
                inner = select_predicted(
                    outer_training_records, inner_examples, inner_predictions, threshold,
                )
                options.append({
                    "alpha": alpha, "threshold": threshold,
                    "score": mean_score(inner),
                    "delta": float(np.mean([row["delta"] for row in inner])),
                    "changed": sum(row["selected"] != "baseline" for row in inner),
                })
        best_score = max(row["score"] for row in options)
        chosen = max(
            (row for row in options if row["score"] == best_score),
            key=lambda row: (row["threshold"], row["alpha"]),
        )
        train = [row for row in examples if row["company"] != held]
        test = [row for row in examples if row["company"] == held]
        prediction = fit_predict(
            train, test, entailment, source, chosen["alpha"],
        )
        outer = select_predicted(
            outer_test_records, test, prediction, chosen["threshold"],
        )
        selected.extend(outer)
        folds.append({
            "held_company": held, "selected_alpha": chosen["alpha"],
            "selected_threshold": chosen["threshold"],
            "inner_score": chosen["score"], "inner_delta": chosen["delta"],
            "inner_changed": chosen["changed"], "rows": len(outer),
            "changed": sum(row["selected"] != "baseline" for row in outer),
            "baseline": float(np.mean([row["baseline_score"] for row in outer])),
            "selected": mean_score(outer),
            "delta": float(np.mean([row["delta"] for row in outer])),
        })
    return selected, folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features", type=Path,
        default=ROOT / "artifacts/polyfiqa_entailment_features_v1/features.npy",
    )
    parser.add_argument(
        "--metadata", type=Path,
        default=ROOT / "artifacts/polyfiqa_entailment_features_v1/metadata.jsonl",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "artifacts/polyfiqa_entailment_features_v1/manifest.json",
    )
    parser.add_argument(
        "--source-features", type=Path,
        default=ROOT / "artifacts/polyfiqa_source_mbr_features_v1/features.npy",
    )
    parser.add_argument(
        "--source-metadata", type=Path,
        default=ROOT / "artifacts/polyfiqa_source_mbr_features_v1/metadata.jsonl",
    )
    parser.add_argument(
        "--lattice", type=Path,
        default=ROOT / "artifacts/polyfiqa_sparse_edit_oracle_v1/lattice.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_entailment_v1",
    )
    args = parser.parse_args()

    entailment = np.load(args.features)
    metadata = load_jsonl(args.metadata)
    source = np.load(args.source_features)
    source_metadata = load_jsonl(args.source_metadata)
    lattice = load_jsonl(args.lattice)
    manifest = json.loads(args.manifest.read_text())
    keys = ("row_index", "task_id", "internal_id", "tier", "family", "operation")
    if [tuple(row[key] for key in keys) for row in metadata] != [
        tuple(row[key] for key in keys) for row in source_metadata
    ]:
        raise ValueError("entailment and source-MBR metadata are not aligned")
    if len(entailment) != len(metadata) or len(source) != len(metadata):
        raise ValueError("feature and metadata row counts differ")

    feature_names = manifest["feature_names"]
    feature_lookup = {name: index for index, name in enumerate(feature_names)}
    diagnostic = {int(row["row_index"]): row for row in lattice}
    by_row = defaultdict(list)
    examples = []
    for feature_index, item in enumerate(metadata):
        row = {**item, "feature_index": feature_index}
        source_row = diagnostic[int(row["row_index"])]
        row["company"] = company(row["internal_id"])
        row["delta"] = (
            float(source_row["scores"][row["operation"]])
            - float(source_row["scores"]["baseline"])
        )
        by_row[row["row_index"]].append(row)

    row_records = []
    for row_index in sorted(by_row):
        candidates = by_row[row_index]
        source_row = diagnostic[row_index]
        baseline_index = next(
            row["feature_index"] for row in candidates if row["operation"] == "baseline"
        )
        for row in candidates:
            row["baseline_feature_index"] = baseline_index
            if row["operation"] != "baseline":
                examples.append(row)
        grounding = {}
        for row in candidates:
            vector = entailment[row["feature_index"]]
            get = lambda name: float(vector[feature_lookup[name]])
            grounding[row["operation"]] = (
                0.25 * get("nli_entail_mean")
                + 0.25 * get("nli_entail_min")
                + 0.20 * get("nli_margin_mean")
                + 0.10 * get("nli_margin_min")
                + 0.10 * get("quote_exact_support")
                + 0.10 * get("core_numeric_support")
            )
        first = candidates[0]
        row_records.append({
            "row_index": row_index, "task_id": first["task_id"],
            "internal_id": first["internal_id"], "company": first["company"],
            "tier": first["tier"], "family": first["family"],
            "scores": {
                name: float(value) for name, value in source_row["scores"].items()
            },
            "answers": {row["operation"]: row["answer"] for row in candidates},
            "grounding": grounding,
        })

    zero = select_fixed(row_records, 0.0)
    zero_folds = []
    for held in COMPANIES:
        held_rows = [row for row in zero if row["held_company"] == held]
        zero_folds.append({
            "held_company": held, "rows": len(held_rows),
            "changed": sum(row["selected"] != "baseline" for row in held_rows),
            "baseline": float(np.mean([row["baseline_score"] for row in held_rows])),
            "selected": mean_score(held_rows),
            "delta": float(np.mean([row["delta"] for row in held_rows])),
        })
    fixed, fixed_folds = fixed_variant(row_records)
    nli, nli_folds = nested_variant(row_records, examples, entailment, None)
    combined, combined_folds = nested_variant(
        row_records, examples, entailment, source,
    )

    variants = {
        "zero_shot_entailment": summarize(
            "zero-shot evidence-entailment score", zero, zero_folds,
        ),
        "company_heldout_margin": summarize(
            "leave-company-out entailment margin", fixed, fixed_folds,
        ),
        "nested_entailment_ridge": summarize(
            "fully nested leave-company-out entailment ridge", nli, nli_folds,
        ),
        "nested_entailment_source_ridge": summarize(
            "fully nested entailment plus source-scalar ridge",
            combined, combined_folds,
        ),
    }
    result = {
        "candidate_rows": len(row_records), "candidate_examples": len(metadata),
        "baseline_rouge": variants["zero_shot_entailment"]["baseline_rouge"],
        "features_are_label_free": True,
        "selection_evaluation_uses_historical_references": True,
        "nested_company_heldout_evaluation": True,
        "current_or_hidden_test_used": False, "variants": variants,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    for name, selected in (
        ("zero_shot", zero), ("margin", fixed), ("nested_nli", nli),
        ("nested_combined", combined),
    ):
        pd.DataFrame(selected).drop(columns=["answer"]).sort_values("row_index").to_csv(
            args.output_dir / f"audit_{name}.csv", index=False,
        )
        (args.output_dir / f"predictions_{name}.jsonl").write_text("".join(
            json.dumps(
                {"task_id": row["task_id"], "answer": row["answer"]},
                ensure_ascii=False,
            ) + "\n"
            for row in sorted(selected, key=lambda item: item["row_index"])
        ))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
