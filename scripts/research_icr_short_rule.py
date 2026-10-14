#!/usr/bin/env python3
"""Nested source-group research for short neutral-response overrides."""

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.parsing import parse_icr_query  # noqa: E402


WITHHOLD = r"差し控|非開示|ノーコメント|お答えでき|申し上げられ"
EXCLUSIONS = {
    "none": r"(?!)",
    "action": r"実施|増配|自社株買い|投資|削減|還元|上場|MBO",
    "action_outlook": r"実施|増配|自社株買い|投資|削減|還元|上場|MBO|見通し|今後|来期|将来",
    "broad": r"実施|増配|自社株買い|投資|削減|還元|上場|MBO|方針|戦略|目標|見通し|今後|来期|将来",
}


def mask(frame: pd.DataFrame, cutoff: int, exclusion: str) -> np.ndarray:
    return (
        (frame.response.str.len().to_numpy() < cutoff)
        & ~frame.response.str.contains(WITHHOLD, regex=True, na=False).to_numpy()
        & ~frame.question.str.contains(EXCLUSIONS[exclusion], regex=True, na=False).to_numpy()
    )


def statistics(selected: np.ndarray, gold: np.ndarray, sources: np.ndarray) -> dict:
    n = int(selected.sum())
    k = int(np.sum(selected & (gold == "0")))
    return {
        "selected": n, "correct": k, "precision": k / n if n else 0.0,
        "net_gain": 2 * k - n,
        "positive_sources": len(set(sources[selected & (gold == "0")])),
    }


def main() -> None:
    data = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    parsed = data["query"].map(parse_icr_query)
    data = data.assign(
        question=parsed.map(lambda x: x.question), response=parsed.map(lambda x: x.response),
    ).merge(
        pd.read_csv(ROOT / "data/derived/icr_source_folds.csv")[["id", "source_group"]],
        on="id", how="left",
    )
    gold = data.answer.astype(str).to_numpy()
    sources = data.source_group.to_numpy()
    candidates = []
    for cutoff in range(30, 251, 10):
        for exclusion in EXCLUSIONS:
            selected = mask(data, cutoff, exclusion)
            candidates.append({"cutoff": cutoff, "exclusion": exclusion,
                               **statistics(selected, gold, sources)})
    frontier = pd.DataFrame(candidates)

    predictions = np.full(len(data), "+1", dtype="<U2")
    choices = []
    for held_out in sorted(set(sources)):
        train, test = sources != held_out, sources == held_out
        viable = []
        for row in candidates:
            selected = mask(data, row["cutoff"], row["exclusion"])
            stat = statistics(selected & train, gold, sources)
            if stat["selected"] >= 6 and stat["precision"] >= 0.70 and stat["positive_sources"] >= 3:
                viable.append((stat["net_gain"], stat["precision"], -stat["selected"],
                               -row["cutoff"], row["exclusion"], selected, stat))
        if not viable:
            continue
        _, _, _, neg_cutoff, exclusion, selected, train_stat = max(viable)
        chosen = selected & test
        predictions[chosen] = "0"
        choices.append({
            "held_out_source": held_out, "cutoff": -neg_cutoff, "exclusion": exclusion,
            **{f"train_{k}": v for k, v in train_stat.items()},
            **{f"test_{k}": v for k, v in statistics(chosen, gold, sources).items()},
        })
    out = ROOT / "artifacts/icr_short_rule"
    out.mkdir(parents=True, exist_ok=True)
    frontier.sort_values(["net_gain", "precision"], ascending=False).to_csv(
        out / "frontier.csv", index=False,
    )
    pd.DataFrame(choices).to_csv(out / "nested_source_choices.csv", index=False)
    pd.DataFrame({"id": data.id, "gold": gold, "prediction": predictions}).to_csv(
        out / "nested_source_predictions.csv", index=False,
    )
    summary = {
        "rows": len(data), "overrides": int(np.sum(predictions != "+1")),
        "majority_accuracy": float(np.mean(gold == "+1")),
        "nested_accuracy": float(np.mean(predictions == gold)),
    }
    (out / "summary.json").write_text(pd.Series(summary).to_json(indent=2) + "\n")
    print(pd.Series(summary).to_string())
    print("\nFIXED FRONTIER")
    print(frontier.sort_values(["net_gain", "precision"], ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
