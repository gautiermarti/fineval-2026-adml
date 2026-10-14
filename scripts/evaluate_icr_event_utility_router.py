#!/usr/bin/env python3
"""Nested source-disjoint event-conditioned utility routing for JF-ICR.

Each frozen expert supplies an OOF ordinal prediction.  On rows where an
alternative disagrees with the incumbent, a ridge model predicts the *relative
utility* of switching: +1 for a rescue, -1 for a harm, and 0 when neither
prediction is correct.  The router remains on the incumbent unless the best
predicted utility clears a nested-selected abstention margin.
"""

from __future__ import annotations

from collections import Counter
import itertools
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
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_icr_adapted import probabilities, source_bootstrap_delta  # noqa: E402
from evaluate_icr_dynamic_expert_selection import OOF_PATHS, decode_experts  # noqa: E402
from fineval.icr_zunda import zunda_signature  # noqa: E402
from predict_icr import rule_prediction  # noqa: E402


ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
MARGINS = (0.0, 0.03, 0.06, 0.10, 0.20, 0.35)


def source_weights(sources: np.ndarray, mask: np.ndarray) -> np.ndarray:
    counts = Counter(sources[mask])
    weights = np.asarray([1.0 / counts[source] for source in sources[mask]])
    return weights / weights.mean()


def fit_utility_models(
    design: np.ndarray,
    predictions: np.ndarray,
    gold: np.ndarray,
    sources: np.ndarray,
    train: np.ndarray,
    alpha: float,
) -> list[object | None]:
    incumbent_correct = predictions[:, 0] == gold
    models: list[object | None] = [None]
    for expert in range(1, predictions.shape[1]):
        disagrees = predictions[:, expert] != predictions[:, 0]
        eligible = train & disagrees
        if not np.any(eligible):
            models.append(None)
            continue
        utility = (
            (predictions[:, expert] == gold).astype(float)
            - incumbent_correct.astype(float)
        )
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(
            design[eligible], utility[eligible],
            ridge__sample_weight=source_weights(sources, eligible),
        )
        models.append(model)
    return models


def route(
    models: list[object | None], design: np.ndarray,
    predictions: np.ndarray, evaluation: np.ndarray, margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.flatnonzero(evaluation)
    chosen = np.zeros(len(positions), dtype=int)
    best_utility = np.full(len(positions), -np.inf, dtype=float)
    for expert, model in enumerate(models[1:], start=1):
        if model is None:
            continue
        disagrees = predictions[positions, expert] != predictions[positions, 0]
        if not np.any(disagrees):
            continue
        utility = np.full(len(positions), -np.inf, dtype=float)
        utility[disagrees] = model.predict(design[positions[disagrees]])
        replace = utility > best_utility
        chosen[replace] = expert
        best_utility[replace] = utility[replace]
    accepted = best_utility > margin
    chosen = np.where(accepted, chosen, 0)
    output = predictions[positions, chosen]
    return output, chosen, best_utility


def candidate_prediction(
    design: np.ndarray, predictions: np.ndarray, gold: np.ndarray,
    sources: np.ndarray, train: np.ndarray, evaluation: np.ndarray,
    alpha: float, margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    models = fit_utility_models(
        design, predictions, gold, sources, train, alpha,
    )
    return route(models, design, predictions, evaluation, margin)


def nested_router(
    design: np.ndarray, predictions: np.ndarray, gold: np.ndarray,
    folds: np.ndarray, sources: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    incumbent = predictions[:, 0]
    output = incumbent.copy()
    selected_expert = np.zeros(len(gold), dtype=int)
    selected_utility = np.full(len(gold), np.nan)
    fold_rows = []
    for outer_fold in sorted(np.unique(folds)):
        outer_train = folds != outer_fold
        outer_test = ~outer_train
        options = [{
            "parameters": None,
            "inner_accuracy": float(np.mean(incumbent[outer_train] == gold[outer_train])),
            "minimum_inner_fold_delta": 0.0,
            "inner_fold_balance": 0,
            "inner_changed_fraction": 0.0,
        }]
        for alpha, margin in itertools.product(ALPHAS, MARGINS):
            inner_prediction = incumbent.copy()
            deltas = []
            for inner_fold in sorted(np.unique(folds[outer_train])):
                inner_train = outer_train & (folds != inner_fold)
                inner_valid = outer_train & (folds == inner_fold)
                guesses, _, _ = candidate_prediction(
                    design, predictions, gold, sources,
                    inner_train, inner_valid, alpha, margin,
                )
                inner_prediction[inner_valid] = guesses
                deltas.append(float(
                    np.mean(guesses == gold[inner_valid])
                    - np.mean(incumbent[inner_valid] == gold[inner_valid])
                ))
            options.append({
                "parameters": (alpha, margin),
                "inner_accuracy": float(np.mean(
                    inner_prediction[outer_train] == gold[outer_train]
                )),
                "minimum_inner_fold_delta": min(deltas),
                "inner_fold_balance": (
                    sum(delta > 0 for delta in deltas)
                    - sum(delta < 0 for delta in deltas)
                ),
                "inner_changed_fraction": float(np.mean(
                    inner_prediction[outer_train] != incumbent[outer_train]
                )),
            })
        # Accuracy is primary. Robustness and abstention break ties; the
        # incumbent wins an exact tie so a switch needs affirmative evidence.
        selected = max(options, key=lambda row: (
            row["inner_accuracy"], row["inner_fold_balance"],
            row["minimum_inner_fold_delta"], -row["inner_changed_fraction"],
            row["parameters"] is None,
        ))
        if selected["parameters"] is not None:
            guesses, experts, utilities = candidate_prediction(
                design, predictions, gold, sources,
                outer_train, outer_test, *selected["parameters"],
            )
            output[outer_test] = guesses
            selected_expert[outer_test] = experts
            selected_utility[outer_test] = utilities
        fold_rows.append({
            "held_out_fold": int(outer_fold),
            **{key: value for key, value in selected.items() if key != "parameters"},
            "selected_parameters": selected["parameters"],
            "incumbent_accuracy": float(np.mean(
                incumbent[outer_test] == gold[outer_test]
            )),
            "accuracy": float(np.mean(output[outer_test] == gold[outer_test])),
            "changed_rows": int(np.sum(output[outer_test] != incumbent[outer_test])),
        })
    return output, selected_expert, selected_utility, fold_rows


def evaluate(
    candidate: np.ndarray, incumbent: np.ndarray, gold: np.ndarray,
    folds: np.ndarray, sources: np.ndarray,
) -> dict:
    fold_candidate = np.asarray([
        np.mean(candidate[folds == fold] == gold[folds == fold])
        for fold in sorted(np.unique(folds))
    ])
    fold_incumbent = np.asarray([
        np.mean(incumbent[folds == fold] == gold[folds == fold])
        for fold in sorted(np.unique(folds))
    ])
    result = {
        "accuracy": float(np.mean(candidate == gold)),
        "rescues": int(np.sum((candidate == gold) & (incumbent != gold))),
        "harms": int(np.sum((candidate != gold) & (incumbent == gold))),
        "changed_rows": int(np.sum(candidate != incumbent)),
        "folds_improved": int(np.sum(fold_candidate > fold_incumbent)),
        "folds_tied": int(np.sum(fold_candidate == fold_incumbent)),
        "folds_worsened": int(np.sum(fold_candidate < fold_incumbent)),
        **source_bootstrap_delta(
            candidate.astype(str), incumbent.astype(str), gold.astype(str), sources,
        ),
    }
    result["promotion_eligible"] = bool(
        result["accuracy"] >= 0.72
        and result["folds_improved"] >= 3
        and result["folds_worsened"] <= 1
        and result["bootstrap_probability_positive"] >= 0.90
        and result["bootstrap_p05_delta"] >= 0.0
    )
    return result


def main() -> None:
    output_dir = ROOT / "artifacts/icr_event_utility_router_v1"
    event_path = ROOT / "artifacts/icr_zunda_semantic_events_v1/events.jsonl"
    records = [json.loads(line) for line in event_path.read_text().splitlines()]
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    ids = frame.id.to_numpy()
    if [int(row["id"]) for row in records] != ids.tolist():
        raise ValueError("Zunda artifact and official rows are not aligned")
    gold = frame.answer.astype(int).to_numpy()
    metadata = pd.read_csv(
        ROOT / "data/derived/icr_source_folds.csv"
    ).set_index("id").loc[ids]
    folds = metadata.source_fold.to_numpy()
    sources = metadata.source_group.to_numpy()
    rules = np.asarray([
        int(rule_prediction(query, "safe_rules")) for query in frame["query"]
    ])
    cube = np.stack([probabilities(path, ids) for path in OOF_PATHS.values()], axis=1)
    expert_predictions, expert_names = decode_experts(cube, rules)
    incumbent = expert_predictions[:, 0]

    probability_design = cube.reshape(len(frame), -1)
    prediction_design = np.concatenate([
        (expert_predictions == level).astype(float)
        for level in range(-2, 3)
    ], axis=1)
    base = np.concatenate([probability_design, prediction_design], axis=1)
    raw_signatures = [zunda_signature([
        {key: value for key, value in candidate.items() if key != "ruri_similarity"}
        for candidate in row["candidates"]
    ]) for row in records]
    semantic_signatures = [zunda_signature(row["candidates"]) for row in records]
    raw = DictVectorizer(sparse=False).fit_transform(raw_signatures)
    semantic = DictVectorizer(sparse=False).fit_transform(semantic_signatures)
    designs = {
        "probability_control": base,
        "raw_events": np.concatenate([base, raw], axis=1),
        "semantic_events": np.concatenate([base, semantic], axis=1),
        "raw_plus_semantic_events": np.concatenate([base, raw, semantic], axis=1),
    }

    predictions_frame: dict[str, object] = {
        "id": ids, "gold": gold, "source_fold": folds,
        "source_group": sources, "incumbent": incumbent,
    }
    summaries = {}
    all_fold_rows = []
    for name, design in designs.items():
        candidate, chosen, utility, fold_rows = nested_router(
            design, expert_predictions, gold, folds, sources,
        )
        summaries[name] = evaluate(
            candidate, incumbent, gold, folds, sources,
        ) | {
            "feature_count": int(design.shape[1]),
            "selection_counts": dict(Counter(
                expert_names[index] for index in chosen
            )),
        }
        predictions_frame[name] = candidate
        predictions_frame[f"{name}_expert"] = [expert_names[index] for index in chosen]
        predictions_frame[f"{name}_utility"] = utility
        all_fold_rows.extend({"variant": name, **row} for row in fold_rows)

    summary = {
        "incumbent_accuracy": float(np.mean(incumbent == gold)),
        "expert_oracle_accuracy": float(np.mean(
            np.any(expert_predictions == gold[:, None], axis=1)
        )),
        "evaluation": "fully nested five-fold reconstructed-source-disjoint",
        "router_target": "alternative correctness minus incumbent correctness",
        "event_features": "input-only Zunda and Ruri-ranked Zunda signatures",
        "current_or_hidden_test_used": False,
        "variants": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions_frame).to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(all_fold_rows).to_csv(output_dir / "folds.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
