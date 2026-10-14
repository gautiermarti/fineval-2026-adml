#!/usr/bin/env python3
"""Source-nested ordinal stacking over frozen JF-ICR OOF probabilities.

This is a diagnostic of recoverable ensemble complementarity, not a deployment
shortcut.  Every meta-model is trained on other source folds, and its ridge
penalty is selected in an inner source-fold loop.
"""

from collections import Counter
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_icr_adapted import probabilities, source_bootstrap_delta  # noqa: E402
from predict_icr import rule_prediction  # noqa: E402
from predict_icr_japanese_ensemble import decode_probabilities  # noqa: E402


OOF_PATHS = {
    "general_seed7": ROOT / "artifacts/task2_japanese_cv/seed7/oof_predictions.csv",
    "general_seed19": ROOT / "artifacts/task2_japanese_cv/seed19/oof_predictions.csv",
    "edinet_seed19": ROOT / "artifacts/task2_edinet_cv/seed19/oof_predictions.csv",
    "rinna_seed7": ROOT / "artifacts/task2_rinna_roberta_cv/seed7_full/oof_predictions.csv",
    "finance_seed7": ROOT / "artifacts/task2_japanese_finance_cv/bert_base_additional_seed7/oof_predictions.csv",
}
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
LEVELS = np.arange(-2, 3)


def source_weights(sources: np.ndarray, mask: np.ndarray) -> np.ndarray:
    counts = Counter(sources[mask])
    return np.asarray([1.0 / counts[source] for source in sources[mask]])


def features(probability_cube: np.ndarray) -> np.ndarray:
    expected = probability_cube @ LEVELS
    entropy = -(probability_cube * np.log(probability_cube + 1e-12)).sum(axis=2)
    ordered = np.sort(probability_cube, axis=2)
    margin = ordered[:, :, -1] - ordered[:, :, -2]
    return np.concatenate(
        [
            probability_cube.reshape(len(probability_cube), -1),
            expected,
            entropy,
            margin,
            expected.std(axis=1, keepdims=True),
        ],
        axis=1,
    )


def decode(values: np.ndarray, rules: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ordinal = np.clip(np.rint(values), -2, 2).astype(int)
    return np.where(rules[mask] != 1, rules[mask], ordinal)


def main() -> None:
    output = ROOT / "artifacts/icr_nested_residual_stacker"
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    metadata = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id")
    metadata = metadata.loc[frame.id]
    ids = frame.id.to_numpy()
    gold = frame.answer.astype(int).to_numpy()
    folds = metadata.source_fold.to_numpy()
    sources = metadata.source_group.to_numpy()
    rules = np.asarray([int(rule_prediction(query, "safe_rules")) for query in frame["query"]])

    cube = np.stack([probabilities(path, ids) for path in OOF_PATHS.values()], axis=1)
    design = features(cube)
    nested = np.zeros(len(frame), dtype=int)
    fold_rows = []
    for outer_fold in sorted(np.unique(folds)):
        outer_train = folds != outer_fold
        outer_test = ~outer_train
        candidates = []
        for alpha in ALPHAS:
            correct = []
            for inner_fold in sorted(np.unique(folds[outer_train])):
                inner_train = outer_train & (folds != inner_fold)
                inner_valid = outer_train & (folds == inner_fold)
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                model.fit(
                    design[inner_train], gold[inner_train],
                    ridge__sample_weight=source_weights(sources, inner_train),
                )
                predicted = decode(model.predict(design[inner_valid]), rules, inner_valid)
                correct.extend(predicted == gold[inner_valid])
            candidates.append((float(np.mean(correct)), alpha))
        best_inner = max(score for score, _ in candidates)
        # Prefer stronger regularization when inner accuracy ties.
        alpha = max(value for score, value in candidates if score == best_inner)
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(
            design[outer_train], gold[outer_train],
            ridge__sample_weight=source_weights(sources, outer_train),
        )
        nested[outer_test] = decode(model.predict(design[outer_test]), rules, outer_test)
        fold_rows.append({
            "held_out_fold": int(outer_fold),
            "selected_alpha": alpha,
            "inner_accuracy": best_inner,
            "nested_accuracy": float(np.mean(nested[outer_test] == gold[outer_test])),
            "rows": int(outer_test.sum()),
        })

    incumbent_probs = cube[:, :2].mean(axis=1)
    incumbent = decode_probabilities(incumbent_probs, ordinal_mean=True).astype(int)
    incumbent = np.where(rules != 1, rules, incumbent)
    base_predictions = np.stack(
        [decode(cube[:, index] @ LEVELS, rules, np.ones(len(frame), dtype=bool))
         for index in range(cube.shape[1])],
        axis=1,
    )
    bootstrap = source_bootstrap_delta(
        nested.astype(str), incumbent.astype(str), gold.astype(str), sources,
    )
    summary = {
        "incumbent_accuracy": float(np.mean(incumbent == gold)),
        "nested_accuracy": float(np.mean(nested == gold)),
        "rescues": int(np.sum((nested == gold) & (incumbent != gold))),
        "harms": int(np.sum((nested != gold) & (incumbent == gold))),
        "oracle_any_base_accuracy": float(np.mean(np.any(base_predictions == gold[:, None], axis=1))),
        "promotion_eligible": False,
        "decision": "diagnostic_only_unstable_across_sources",
        **bootstrap,
    }
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(output / "folds.csv", index=False)
    pd.DataFrame({
        "id": ids, "gold": gold, "source_fold": folds,
        "incumbent": incumbent, "nested_stacker": nested,
    }).to_csv(output / "predictions.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(pd.DataFrame(fold_rows).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
