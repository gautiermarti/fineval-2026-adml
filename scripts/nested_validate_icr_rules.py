#!/usr/bin/env python3
"""Nested source-fold validation of the interpretable-rule mining process."""

from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.parsing import parse_icr_query  # noqa: E402
from mine_icr_rules import features  # noqa: E402


TARGETS = ("+2", "0", "-1", "-2")


def candidate_masks(atoms: dict[str, np.ndarray]):
    names = sorted(atoms)
    for width in (1, 2, 3):
        for selected in combinations(names, width):
            if any(f"not_{name}" in selected and name in selected for name in selected):
                continue
            yield " AND ".join(selected), np.logical_and.reduce([atoms[name] for name in selected])


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    parsed = frame["query"].map(parse_icr_query)
    frame = frame.assign(
        question=parsed.map(lambda x: x.question), response=parsed.map(lambda x: x.response),
    ).merge(
        pd.read_csv(ROOT / "data/derived/icr_source_folds.csv")[["id", "source_group", "source_fold"]],
        on="id", how="left",
    )
    atoms = features(frame)
    masks = list(candidate_masks(atoms))
    gold = frame.answer.astype(str).to_numpy()
    predictions = np.full(len(frame), "+1", dtype="<U2")
    choices = []
    for fold in sorted(frame.source_fold.unique()):
        train = frame.source_fold.to_numpy() != fold
        test = ~train
        for target in TARGETS:
            viable = []
            for rule, mask in masks:
                selected = mask & train
                n = int(selected.sum())
                if n < 4:
                    continue
                k = int(np.sum(selected & (gold == target)))
                precision = k / n
                positive_sources = frame.loc[selected & (gold == target), "source_group"].nunique()
                if precision < 0.75 or positive_sources < 2:
                    continue
                viable.append((2 * k - n, precision, positive_sources, -len(rule), rule, mask))
            if not viable:
                continue
            _, precision, positive_sources, _, rule, mask = max(viable)
            # Earlier target rules have priority if two rules overlap.
            selected_test = mask & test & (predictions == "+1")
            predictions[selected_test] = target
            choices.append({
                "fold": fold, "target": target, "rule": rule,
                "train_precision": precision, "train_positive_sources": positive_sources,
                "test_selected": int(selected_test.sum()),
                "test_correct": int(np.sum(selected_test & (gold == target))),
            })
    rows = []
    for split, mask in (
        ("all", np.ones(len(frame), dtype=bool)),
        ("core", frame.id.to_numpy() <= 93),
        ("expanded", frame.id.to_numpy() > 93),
    ):
        rows.append({
            "split": split, "rows": int(mask.sum()),
            "majority_accuracy": float(np.mean(gold[mask] == "+1")),
            "nested_rule_accuracy": float(np.mean(predictions[mask] == gold[mask])),
            "overrides": int(np.sum(predictions[mask] != "+1")),
        })
    out = ROOT / "artifacts/icr_gates"
    pd.DataFrame(choices).to_csv(out / "nested_rule_choices.csv", index=False)
    pd.DataFrame({"id": frame.id, "gold": gold, "prediction": predictions}).to_csv(
        out / "nested_rule_oof_predictions.csv", index=False,
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "nested_rule_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nFOLD CHOICES")
    print(pd.DataFrame(choices).to_string(index=False))


if __name__ == "__main__":
    main()
