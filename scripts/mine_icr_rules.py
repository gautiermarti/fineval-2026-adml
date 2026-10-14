#!/usr/bin/env python3
"""Mine interpretable high-precision JF-ICR override rules."""

from itertools import combinations
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.parsing import parse_icr_query  # noqa: E402


def contains(values: pd.Series, pattern: str) -> np.ndarray:
    return values.str.contains(pattern, regex=True, na=False).to_numpy()


def features(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    question, response = frame["question"], frame["response"]
    response_length = response.str.len().to_numpy()
    atoms = {
        "a_len_lt_80": response_length < 80,
        "a_len_lt_160": response_length < 160,
        "a_len_gt_300": response_length > 300,
        "a_len_gt_600": response_length > 600,
        "a_withhold": contains(response, r"差し控|非開示|ノーコメント|お答えでき|申し上げられ"),
        "a_undecided": contains(response, r"未定|検討中|決定.{0,6}ない|決断.{0,6}ない|議論.{0,8}最中"),
        "a_no_plan": contains(response, r"予定.{0,6}ない|計画.{0,6}ない|方針.{0,6}ない"),
        "a_no_forecast": contains(response, r"見通.{0,8}(?:困難|でき|せない)|何とも言えない"),
        "a_explicit_target": contains(response, r"目指します|必ず.{0,8}(?:達成|やり切)|決議.{0,8}いたしました|維持する方針"),
        "a_future_action": contains(response, r"取り組んでいく|進めていく|実施する|実行する|強化していく|拡大していく"),
        "a_soft_intent": contains(response, r"していきたい|と考えている|と思っている|見込んでいる|予定している"),
        "a_acknowledge": contains(response, r"認識している|理解している|評価している"),
        "a_temporary": contains(response, r"現時点|現状では|今のところ|当面|しばらく"),
        "a_numbers": contains(response, r"\d"),
        "q_outlook": contains(question, r"見通し|見通す|今後|来期|将来|予想"),
        "q_policy": contains(question, r"方針|考え方|考えを|戦略|目標"),
        "q_fact": contains(question, r"教えて|どの程度|いくら|内訳|理由|背景|影響"),
        "q_confirmation": contains(question, r"理解で良い|ということか|想定しているのか|考えているか|良いか"),
        "q_action": contains(question, r"実施|増配|自社株買い|投資|削減|還元|上場|MBO"),
    }
    # Useful absences are explicit atoms, keeping mined rules readable.
    for name in (
        "a_withhold", "a_undecided", "a_explicit_target", "a_future_action",
        "a_soft_intent", "a_numbers", "q_outlook", "q_action",
    ):
        atoms[f"not_{name}"] = ~atoms[name]
    return atoms


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
    names = sorted(atoms)
    rows = []
    for width in (1, 2, 3):
        for selected_names in combinations(names, width):
            # Skip direct contradictions such as feature AND not_feature.
            if any(
                f"not_{name}" in selected_names and name in selected_names
                for name in selected_names
            ):
                continue
            selected = np.logical_and.reduce([atoms[name] for name in selected_names])
            for split, split_mask in (
                ("all", np.ones(len(frame), dtype=bool)),
                ("core", frame["id"].to_numpy() <= 93),
                ("expanded", frame["id"].to_numpy() > 93),
            ):
                mask = selected & split_mask
                coverage = int(mask.sum())
                if coverage < 2:
                    continue
                for target in ("0", "+2", "-1", "-2"):
                    correct = mask & (frame["answer"].to_numpy() == target)
                    true_positive = int(correct.sum())
                    precision = true_positive / coverage
                    net_gain = true_positive - (coverage - true_positive)
                    positive_sources = frame.loc[correct, "source_group"].nunique()
                    if net_gain <= 0 or precision < 0.60:
                        continue
                    rows.append({
                        "split": split, "target": target,
                        "rule": " AND ".join(selected_names), "width": width,
                        "coverage": coverage, "correct": true_positive,
                        "precision": precision, "net_gain": net_gain,
                        "positive_sources": positive_sources,
                        "selected_ids": " ".join(map(str, frame.loc[mask, "id"].tolist())),
                    })
    results = pd.DataFrame(rows).drop_duplicates(
        ["split", "target", "coverage", "correct", "selected_ids"]
    ).sort_values(
        ["split", "net_gain", "precision", "positive_sources", "width"],
        ascending=[True, False, False, False, True],
    )
    output = ROOT / "artifacts/icr_gates/interpretable_rule_candidates.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    robust = results[(results["positive_sources"] >= 2) & (results["coverage"] >= 3)]
    print(robust.groupby(["split", "target"], group_keys=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
