#!/usr/bin/env python3
"""Source-nested diagnostic for Tsutsuji semantic functional expressions.

This asks a narrow question: do explicit Japanese modality/speech-act
signatures add cross-source information beyond frozen model probabilities?
Hyperparameters are selected inside each outer source fold.  The result is a
diagnostic and is promoted only through the same gate as other Task 2 models.
"""

from __future__ import annotations

from collections import Counter
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
from fineval.icr_functional_expressions import (  # noqa: E402
    informative_entries, load_tsutsuji, semantic_signature,
)
from fineval.parsing import parse_icr_query  # noqa: E402
from predict_icr import rule_prediction  # noqa: E402
from predict_icr_japanese_ensemble import decode_probabilities  # noqa: E402


OOF_PATHS = (
    ROOT / "artifacts/task2_japanese_cv/seed7/oof_predictions.csv",
    ROOT / "artifacts/task2_japanese_cv/seed19/oof_predictions.csv",
)
LEXICON = ROOT / "data/external/tsutsuji/tsutsuji-1.1u/tsutsuji1.1.xml"
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
LEVELS = np.arange(-2, 3)


def source_weights(sources: np.ndarray, mask: np.ndarray) -> np.ndarray:
    counts = Counter(sources[mask])
    return np.asarray([1.0 / counts[source] for source in sources[mask]])


def decode(values: np.ndarray, rules: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ordinal = np.clip(np.rint(values), -2, 2).astype(int)
    return np.where(rules[mask] != 1, rules[mask], ordinal)


def nested_prediction(
    design: np.ndarray, gold: np.ndarray, folds: np.ndarray,
    sources: np.ndarray, rules: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    prediction = np.zeros(len(gold), dtype=int)
    fold_rows: list[dict] = []
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
                candidate = decode(model.predict(design[inner_valid]), rules, inner_valid)
                correct.extend(candidate == gold[inner_valid])
            candidates.append((float(np.mean(correct)), alpha))
        best_inner = max(score for score, _ in candidates)
        # Conservative tie-break: stronger regularization.
        alpha = max(value for score, value in candidates if score == best_inner)
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(
            design[outer_train], gold[outer_train],
            ridge__sample_weight=source_weights(sources, outer_train),
        )
        prediction[outer_test] = decode(
            model.predict(design[outer_test]), rules, outer_test,
        )
        fold_rows.append({
            "held_out_fold": int(outer_fold), "selected_alpha": alpha,
            "inner_accuracy": best_inner,
            "accuracy": float(np.mean(prediction[outer_test] == gold[outer_test])),
            "rows": int(outer_test.sum()),
        })
    return prediction, fold_rows


def main() -> None:
    output = ROOT / "artifacts/icr_tsutsuji_semantics_v1"
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    ids = frame.id.to_numpy()
    gold = frame.answer.astype(int).to_numpy()
    metadata = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id").loc[ids]
    folds = metadata.source_fold.to_numpy()
    sources = metadata.source_group.to_numpy()
    rules = np.asarray([int(rule_prediction(query, "safe_rules")) for query in frame["query"]])
    responses = [parse_icr_query(query).response for query in frame["query"]]

    entries = informative_entries(load_tsutsuji(LEXICON))
    signatures = [semantic_signature(response, entries) for response in responses]
    vectorizer = DictVectorizer(sparse=False)
    semantic = vectorizer.fit_transform(signatures)
    cube = np.stack([probabilities(path, ids) for path in OOF_PATHS], axis=1)
    incumbent_probs = cube.mean(axis=1)
    incumbent = decode_probabilities(incumbent_probs, ordinal_mean=True).astype(int)
    incumbent = np.where(rules != 1, rules, incumbent)
    base = np.concatenate([
        incumbent_probs,
        (incumbent_probs @ LEVELS)[:, None],
        (-(incumbent_probs * np.log(incumbent_probs + 1e-12)).sum(axis=1))[:, None],
    ], axis=1)

    variants = {
        "semantic_only": semantic,
        "incumbent_plus_semantic": np.concatenate([base, semantic], axis=1),
    }
    predictions = {"id": ids, "gold": gold, "source_fold": folds,
                   "source_group": sources, "incumbent": incumbent}
    summaries = {}
    all_folds = []
    for name, design in variants.items():
        predicted, fold_rows = nested_prediction(
            design, gold, folds, sources, rules,
        )
        bootstrap = source_bootstrap_delta(
            predicted.astype(str), incumbent.astype(str), gold.astype(str), sources,
        )
        fold_incumbent = [
            float(np.mean(incumbent[folds == fold] == gold[folds == fold]))
            for fold in sorted(np.unique(folds))
        ]
        fold_candidate = [row["accuracy"] for row in fold_rows]
        summary = {
            "accuracy": float(np.mean(predicted == gold)),
            "rescues": int(np.sum((predicted == gold) & (incumbent != gold))),
            "harms": int(np.sum((predicted != gold) & (incumbent == gold))),
            "folds_improved": int(np.sum(np.asarray(fold_candidate) > fold_incumbent)),
            "folds_tied": int(np.sum(np.asarray(fold_candidate) == fold_incumbent)),
            "folds_worsened": int(np.sum(np.asarray(fold_candidate) < fold_incumbent)),
            **bootstrap,
        }
        summary["promotion_eligible"] = bool(
            summary["accuracy"] >= 0.72 and summary["folds_improved"] >= 3
            and summary["folds_worsened"] <= 1
            and summary["bootstrap_probability_positive"] >= 0.90
            and summary["bootstrap_p05_delta"] >= 0.0
        )
        summaries[name] = summary
        predictions[name] = predicted
        all_folds.extend({"variant": name, **row} for row in fold_rows)

    result = {
        "incumbent_accuracy": float(np.mean(incumbent == gold)),
        "lexicon_entries_total": len(load_tsutsuji(LEXICON)),
        "lexicon_entries_used": len(entries),
        "semantic_features": int(semantic.shape[1]),
        "matched_rows": int(sum(row["global:matches"] > 0 for row in signatures)),
        "evaluation": "nested five-fold source-disjoint",
        "public_test_used": False,
        "variants": summaries,
    }
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(all_folds).to_csv(output / "folds.csv", index=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "feature_names.txt").write_text(
        "\n".join(vectorizer.get_feature_names_out()) + "\n",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
