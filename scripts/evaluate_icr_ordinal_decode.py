#!/usr/bin/env python3
"""Compare categorical and ordinal decoding on frozen source-OOF predictions."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from predict_icr import rule_prediction  # noqa: E402
from predict_icr_japanese_ensemble import decode_probabilities  # noqa: E402
from train_icr_japanese_cv import LABELS  # noqa: E402


def load_probabilities(seed: int) -> np.ndarray:
    frame = pd.read_csv(
        ROOT / f"artifacts/task2_japanese_cv/seed{seed}/oof_predictions.csv"
    )
    logits = frame[[f"logit_{label}" for label in LABELS]].to_numpy()
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    metadata = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id")
    metadata = metadata.loc[frame.id]
    gold = frame.answer.astype(str).to_numpy()
    probabilities = (load_probabilities(7) + load_probabilities(19)) / 2
    rules = np.asarray([rule_prediction(query, "safe_rules") for query in frame["query"]])

    rows = []
    predictions = {}
    for decoder, ordinal in (("argmax", False), ("ordinal_mean", True)):
        raw = decode_probabilities(probabilities, ordinal_mean=ordinal)
        prediction = np.where(rules != "+1", rules, raw)
        predictions[decoder] = prediction
        rows.append({
            "decoder": decoder,
            "all_accuracy": float(np.mean(prediction == gold)),
            "core_accuracy": float(np.mean(prediction[:94] == gold[:94])),
            "expanded_accuracy": float(np.mean(prediction[94:] == gold[94:])),
            **{
                f"fold_{fold}_accuracy": float(np.mean(
                    prediction[metadata.source_fold.to_numpy() == fold]
                    == gold[metadata.source_fold.to_numpy() == fold]
                ))
                for fold in sorted(metadata.source_fold.unique())
            },
        })

    source = metadata.source_group.to_numpy()
    groups = np.unique(source)
    group_indices = {group: np.flatnonzero(source == group) for group in groups}
    rng = np.random.default_rng(20260804)
    deltas = []
    for _ in range(100_000):
        sampled = rng.choice(groups, len(groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled])
        deltas.append(
            np.mean(predictions["ordinal_mean"][indices] == gold[indices])
            - np.mean(predictions["argmax"][indices] == gold[indices])
        )
    deltas = np.asarray(deltas)
    result = pd.DataFrame(rows)
    output = ROOT / "artifacts/icr_gates/ordinal_decode_comparison.csv"
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(
        "paired_cluster_bootstrap "
        f"p05={np.quantile(deltas, .05):.12f} "
        f"median={np.quantile(deltas, .50):.12f} "
        f"p95={np.quantile(deltas, .95):.12f} "
        f"p_ordinal_beats={np.mean(deltas > 0):.12f}"
    )
    print(f"wrote={output}")


if __name__ == "__main__":
    main()
