#!/usr/bin/env python3
"""Source-cluster bootstrap comparison of fixed Task 2 deployment profiles."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PORTFOLIO = ROOT / "artifacts/submission_portfolio_task2_public"
PROFILES = ("majority", "short_neutral", "broad_short_neutral", "safe_rules")


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    source = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id").loc[
        frame.id, "source_group"
    ].to_numpy()
    gold = frame.answer.astype(str).to_numpy()
    predictions = {}
    for profile in PROFILES:
        path = PORTFOLIO / f"task2_{profile}.jsonl"
        predictions[profile] = np.asarray([
            json.loads(line)["answer"] for line in path.read_text().splitlines()
        ])
    groups = np.unique(source)
    group_indices = {group: np.flatnonzero(source == group) for group in groups}
    rng = np.random.default_rng(20260801)
    draws = {profile: [] for profile in PROFILES}
    for _ in range(100_000):
        sampled = rng.choice(groups, len(groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled])
        for profile in PROFILES:
            draws[profile].append(float(np.mean(predictions[profile][indices] == gold[indices])))
    rows = []
    baseline = np.asarray(draws["majority"])
    broad = np.asarray(draws["broad_short_neutral"])
    for profile in PROFILES:
        values = np.asarray(draws[profile])
        delta = values - baseline
        rows.append({
            "profile": profile, "point_accuracy": float(np.mean(predictions[profile] == gold)),
            "bootstrap_accuracy_p05": float(np.quantile(values, .05)),
            "bootstrap_accuracy_p50": float(np.quantile(values, .50)),
            "bootstrap_accuracy_p95": float(np.quantile(values, .95)),
            "p_beats_majority": float(np.mean(delta > 0)),
            "delta_vs_majority_p05": float(np.quantile(delta, .05)),
            "delta_vs_majority_p95": float(np.quantile(delta, .95)),
            "p_beats_broad_short": float(np.mean(values > broad)),
        })
    result = pd.DataFrame(rows)
    output = ROOT / "artifacts/icr_gates/profile_cluster_bootstrap.csv"
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
