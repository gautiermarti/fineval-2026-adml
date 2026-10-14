#!/usr/bin/env python3
"""Cross-check ICR minority-label gates across independent fold schemes.

Hyperparameters and thresholds are selected on one OOF scheme, then evaluated
unchanged on the other.  This is deliberately stricter than reporting the best
point from each frontier independently.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/icr_gates"


def score(gold: np.ndarray, probability: np.ndarray, target: str, threshold: float) -> dict:
    selected = probability >= threshold
    prediction = np.full(len(gold), "+1", dtype="<U2")
    prediction[selected] = target
    correct = int(np.sum(selected & (gold == target)))
    return {
        "accuracy": float(np.mean(prediction == gold)),
        "selected": int(selected.sum()),
        "correct_overrides": correct,
        "override_precision": correct / selected.sum() if selected.any() else 0.0,
        "net_gain": int(np.sum(prediction == gold) - np.sum(gold == "+1")),
    }


def main() -> None:
    probabilities = pd.read_csv(
        ARTIFACTS / "oof_probabilities.csv", dtype={"gold": str},
    )
    frontiers = pd.read_csv(
        ARTIFACTS / "binary_gate_frontiers.csv", dtype={"target": str},
    )
    gold = probabilities["gold"].astype(str).to_numpy()
    ids = probabilities["id"].to_numpy()
    rows = []
    consensus_rows = []
    for selection_scheme, test_scheme in (("source", "semantic"), ("semantic", "source")):
        candidates = frontiers[
            (frontiers.scheme == selection_scheme)
            & (frontiers.split == "all")
            & (frontiers.selected >= 4)
            & (frontiers.override_precision >= 0.60)
            & (frontiers.net_gain > 0)
        ].copy()
        # One threshold per model, selected exclusively using the first scheme.
        candidates = candidates.sort_values(
            ["net_gain", "override_precision", "selected"],
            ascending=[False, False, True],
        ).drop_duplicates(["view", "C", "target"])
        for _, chosen in candidates.iterrows():
            stem = f"{chosen['view']}__c{chosen['C']:g}__{chosen['target']}"
            select_p = probabilities[f"{selection_scheme}__{stem}"].to_numpy()
            test_p = probabilities[f"{test_scheme}__{stem}"].to_numpy()
            threshold = float(chosen.threshold)
            result = score(gold, test_p, str(chosen.target), threshold)
            rows.append({
                "selected_on": selection_scheme,
                "tested_on": test_scheme,
                "view": chosen["view"], "C": chosen["C"],
                "target": chosen.target, "threshold": threshold,
                "selection_accuracy": chosen.accuracy,
                "selection_net_gain": int(chosen.net_gain),
                "selection_precision": chosen.override_precision,
                **{f"test_{key}": value for key, value in result.items()},
            })

            # Agreement is a useful conservative deployment policy: both OOF
            # constructions must independently select the same example.
            agreed = (select_p >= threshold) & (test_p >= threshold)
            agreed_p = np.where(agreed, 1.0, 0.0)
            agreement_score = score(gold, agreed_p, str(chosen.target), 0.5)
            consensus_rows.append({
                "selected_on": selection_scheme, "view": chosen["view"],
                "C": chosen["C"], "target": chosen.target,
                "threshold": threshold,
                **agreement_score,
                "ids": " ".join(map(str, ids[agreed])),
            })

    cross = pd.DataFrame(rows).sort_values(
        ["test_net_gain", "test_override_precision"], ascending=False,
    )
    consensus = pd.DataFrame(consensus_rows).sort_values(
        ["net_gain", "override_precision"], ascending=False,
    )
    cross.to_csv(ARTIFACTS / "cross_scheme_gate_validation.csv", index=False)
    consensus.to_csv(ARTIFACTS / "consensus_gate_validation.csv", index=False)
    columns = [
        "selected_on", "tested_on", "view", "C", "target", "threshold",
        "selection_net_gain", "test_net_gain", "test_selected",
        "test_override_precision", "test_accuracy",
    ]
    print("CROSS-SCHEME LEADERS")
    print(cross[columns].head(20).to_string(index=False))
    print("\nCONSENSUS LEADERS")
    print(consensus.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
