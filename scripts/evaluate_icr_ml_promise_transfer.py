#!/usr/bin/env python3
"""Source-disjoint evaluation of ML-Promise transfer signals on JF-ICR."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_icr_adapted import probabilities, source_bootstrap_delta  # noqa: E402
from evaluate_icr_dynamic_expert_selection import (  # noqa: E402
    OOF_PATHS as EXPERT_PATHS, decode_experts,
)
from evaluate_icr_event_utility_router import (  # noqa: E402
    evaluate as evaluate_router, nested_router,
)
from evaluate_icr_event_views import accuracy_summary, source_equal_weights  # noqa: E402
from evaluate_icr_tsutsuji_semantics import (  # noqa: E402
    LEVELS, OOF_PATHS as INCUMBENT_PATHS, nested_prediction,
)
from evaluate_icr_zunda_residual import (  # noqa: E402
    evaluate as evaluate_ordinal, nested_residual,
)
from fineval.icr_events import event_candidates  # noqa: E402
from fineval.parsing import parse_icr_query  # noqa: E402
from predict_icr import rule_prediction  # noqa: E402
from predict_icr_japanese_ensemble import decode_probabilities  # noqa: E402


TRANSFER_COLUMNS = (
    "external_row_promise_probability",
    "external_max_event_promise_probability",
    "external_mean_top3_event_probability",
    "external_tail_event_probability",
    "external_high_probability_event_count",
    "external_top_event_relative_position",
    "external_top_joint_score",
    "event_candidate_count",
)


def event_view_predictions(
    frame: pd.DataFrame, metadata: pd.DataFrame, features: pd.DataFrame,
) -> tuple[dict, dict[str, np.ndarray]]:
    parsed = [parse_icr_query(query) for query in frame["query"]]
    views = {
        "full_response_control": [
            item.question + " [Q-A] " + item.response for item in parsed
        ],
        "input_only_top1_control": [
            item.question + " [Q-A] " + event_candidates(item.question, item.response)[0].text
            for item in parsed
        ],
        "ml_promise_top1": [
            item.question + " [Q-A] " + text
            for item, text in zip(parsed, features.external_top_promise_event, strict=True)
        ],
        "ml_promise_joint_top1": [
            item.question + " [Q-A] " + text
            for item, text in zip(parsed, features.external_top_joint_event, strict=True)
        ],
    }
    gold = frame.answer.astype(str).tolist()
    folds = metadata.source_fold.astype(int).to_numpy()
    summaries, predictions = {}, {}
    for name, texts in views.items():
        predicted = np.empty(len(frame), dtype=object)
        for fold in sorted(set(folds)):
            train = folds != fold
            valid = ~train
            vectorizer = TfidfVectorizer(
                analyzer="char", ngram_range=(2, 5), min_df=2,
                max_features=60_000, sublinear_tf=True,
            )
            train_matrix = vectorizer.fit_transform([
                text for text, keep in zip(texts, train, strict=True) if keep
            ])
            valid_matrix = vectorizer.transform([
                text for text, keep in zip(texts, valid, strict=True) if keep
            ])
            model = LogisticRegression(
                C=2.0, class_weight="balanced", max_iter=3000,
                solver="lbfgs", random_state=7,
            )
            model.fit(
                train_matrix, frame.loc[train, "answer"],
                sample_weight=source_equal_weights(metadata.loc[train, "source_group"]),
            )
            predicted[valid] = model.predict(valid_matrix)
        predictions[name] = predicted
        summaries[name] = accuracy_summary(
            gold, predicted.tolist(), folds.tolist(),
        )
    return summaries, predictions


def main() -> None:
    output_dir = ROOT / "artifacts/icr_ml_promise_transfer_v1"
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    ids = frame.id.to_numpy()
    gold = frame.answer.astype(int).to_numpy()
    metadata = pd.read_csv(
        ROOT / "data/derived/icr_source_folds.csv"
    ).set_index("id").loc[ids].reset_index()
    folds = metadata.source_fold.to_numpy()
    sources = metadata.source_group.to_numpy()
    rules = np.asarray([
        int(rule_prediction(query, "safe_rules")) for query in frame["query"]
    ])
    features = pd.read_csv(
        ROOT / "artifacts/ml_promise_japanese_transfer_features_v1/jf_icr_features.csv"
    ).set_index("id").loc[ids].reset_index()
    transfer = features.loc[:, TRANSFER_COLUMNS].to_numpy(dtype=float)

    event_summaries, event_predictions = event_view_predictions(
        frame, metadata, features,
    )

    incumbent_cube = np.stack([
        probabilities(path, ids) for path in INCUMBENT_PATHS
    ], axis=1)
    incumbent_probabilities = incumbent_cube.mean(axis=1)
    score = incumbent_probabilities @ LEVELS
    incumbent = decode_probabilities(
        incumbent_probabilities, ordinal_mean=True,
    ).astype(int)
    incumbent = np.where(rules != 1, rules, incumbent)
    base = np.concatenate([
        incumbent_probabilities,
        score[:, None],
        (-(incumbent_probabilities * np.log(
            incumbent_probabilities + 1e-12
        )).sum(axis=1))[:, None],
    ], axis=1)

    ordinal_variants = {}
    ordinal_predictions = {}
    all_folds = []
    direct, direct_folds = nested_prediction(
        np.concatenate([base, transfer], axis=1),
        gold, folds, sources, rules,
    )
    ordinal_variants["incumbent_plus_transfer_direct"] = evaluate_ordinal(
        direct, incumbent, gold, folds, sources,
    )
    ordinal_predictions["incumbent_plus_transfer_direct"] = direct
    all_folds.extend({"variant": "incumbent_plus_transfer_direct", **row}
                     for row in direct_folds)
    residual, residual_folds = nested_residual(
        np.concatenate([base, transfer], axis=1), score, gold,
        folds, sources, rules, incumbent,
    )
    ordinal_variants["incumbent_plus_transfer_residual"] = evaluate_ordinal(
        residual, incumbent, gold, folds, sources,
    )
    ordinal_predictions["incumbent_plus_transfer_residual"] = residual
    all_folds.extend({"variant": "incumbent_plus_transfer_residual", **row}
                     for row in residual_folds)

    expert_cube = np.stack([
        probabilities(path, ids) for path in EXPERT_PATHS.values()
    ], axis=1)
    expert_predictions, expert_names = decode_experts(expert_cube, rules)
    probability_design = expert_cube.reshape(len(frame), -1)
    prediction_design = np.concatenate([
        (expert_predictions == level).astype(float) for level in range(-2, 3)
    ], axis=1)
    router_design = np.concatenate([
        probability_design, prediction_design, transfer,
    ], axis=1)
    routed, chosen, utility, router_folds = nested_router(
        router_design, expert_predictions, gold, folds, sources,
    )
    router_summary = evaluate_router(
        routed, incumbent, gold, folds, sources,
    ) | {
        "selection_counts": dict(Counter(expert_names[index] for index in chosen)),
        "feature_count": int(router_design.shape[1]),
    }
    all_folds.extend({"variant": "transfer_expert_utility_router", **row}
                     for row in router_folds)

    output = pd.DataFrame({
        "id": ids,
        "gold": gold,
        "source_fold": folds,
        "source_group": sources,
        "incumbent": incumbent,
        **ordinal_predictions,
        "transfer_expert_utility_router": routed,
        "transfer_selected_expert": [expert_names[index] for index in chosen],
        "transfer_predicted_utility": utility,
        **{f"event_view_{name}": prediction for name, prediction in event_predictions.items()},
    })
    summary = {
        "incumbent_accuracy": float(np.mean(incumbent == gold)),
        "external_feature_source": "357 benchmark-disjoint ML-Promise Japanese train rows",
        "evaluation": "fully nested five-fold reconstructed-source-disjoint",
        "event_view_diagnostics": event_summaries,
        "ordinal_variants": ordinal_variants,
        "expert_utility_router": router_summary,
        "target_labels_used_only_inside_nested_evaluation": True,
        "ml_promise_test_used": False,
        "current_or_hidden_test_used": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(all_folds).to_csv(output_dir / "folds.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
