#!/usr/bin/env python3
"""Evaluate pre-specified interpretable ICR overrides by source and bootstrap."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.parsing import parse_icr_query  # noqa: E402
from mine_icr_rules import features  # noqa: E402


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    parsed = frame["query"].map(parse_icr_query)
    frame = frame.assign(
        question=parsed.map(lambda item: item.question),
        response=parsed.map(lambda item: item.response),
    )
    sources = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv")[["id", "source_group"]]
    frame = frame.merge(sources, on="id", how="left")
    atoms = features(frame)
    rules = {
        "short_neutral": ("0", atoms["a_len_lt_80"] & atoms["not_a_withhold"] & atoms["not_q_action"]),
        "explicit_target": ("+2", atoms["a_explicit_target"]),
    }
    rng = np.random.default_rng(20260801)
    groups = frame["source_group"].drop_duplicates().to_numpy()
    rows = []
    source_rows = []
    for name, (target, selected) in rules.items():
        for split, split_mask in (
            ("all", np.ones(len(frame), dtype=bool)),
            ("core", frame.id.to_numpy() <= 93),
            ("expanded", frame.id.to_numpy() > 93),
        ):
            mask = selected & split_mask
            correct = mask & (frame.answer.to_numpy() == target)
            n, k = int(mask.sum()), int(correct.sum())
            lower, upper = wilson(k, n)
            rows.append({
                "rule": name, "target": target, "split": split,
                "selected": n, "correct": k, "precision": k / n if n else 0,
                "precision_wilson_low": lower, "precision_wilson_high": upper,
                "net_gain": 2 * k - n,
                "positive_sources": frame.loc[correct, "source_group"].nunique(),
                "selected_sources": frame.loc[mask, "source_group"].nunique(),
            })
        for source, group in frame.groupby("source_group", sort=True):
            idx = group.index.to_numpy()
            chosen = selected[idx]
            if not chosen.any():
                continue
            k = int(np.sum(chosen & (group.answer.to_numpy() == target)))
            source_rows.append({
                "rule": name, "target": target, "source_group": source,
                "selected": int(chosen.sum()), "correct": k,
                "net_gain": 2 * k - int(chosen.sum()),
            })

        # Cluster bootstrap preserves within-document dependence. Report the
        # chance the fixed rule has positive gain on a resampled source mix.
        gains = []
        for _ in range(20_000):
            sampled = rng.choice(groups, len(groups), replace=True)
            gain = 0
            for source in sampled:
                idx = frame.source_group.to_numpy() == source
                chosen = selected & idx
                k = np.sum(chosen & (frame.answer.to_numpy() == target))
                gain += 2 * int(k) - int(chosen.sum())
            gains.append(gain)
        rows[-3]["bootstrap_positive_gain_probability"] = float(np.mean(np.asarray(gains) > 0))
        rows[-3]["bootstrap_net_gain_p05"] = float(np.quantile(gains, 0.05))
        rows[-3]["bootstrap_net_gain_p95"] = float(np.quantile(gains, 0.95))

    summary = pd.DataFrame(rows)
    by_source = pd.DataFrame(source_rows)
    out = ROOT / "artifacts/icr_gates"
    summary.to_csv(out / "fixed_rule_reliability.csv", index=False)
    by_source.to_csv(out / "fixed_rule_by_source.csv", index=False)
    print(summary.to_string(index=False))
    print("\nBY SOURCE")
    print(by_source.to_string(index=False))


if __name__ == "__main__":
    main()
