#!/usr/bin/env python3
"""Evaluate the preregistered external-corpus-derived JF-ICR refusal rule."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from evaluate_icr_adapted import source_bootstrap_delta  # noqa: E402
from evaluate_icr_ordinal_decode import load_probabilities  # noqa: E402
from fineval.parsing import parse_icr_query  # noqa: E402
from predict_icr import rule_prediction  # noqa: E402
from predict_icr_japanese_ensemble import decode_probabilities  # noqa: E402


PREREGISTRATION = ROOT / "docs/icr_external_refusal_rule_preregistration.md"
PREREGISTRATION_SHA256 = (
    "5f8391c13ecba3871b9f8d151cffdeadc3bc2d4c9858821c56cdef1115ef0809"
)
EXTERNAL = ROOT / "data/external/japanese_ir_qa/forward_candidates.jsonl"
EXTERNAL_SHA256 = (
    "e84a94e2088cf66ba5d4723b3835d894b9fe45aab4b90ec09865e8041e897f02"
)

WITHHOLDING = re.compile(
    r"(?:回答|言及|コメント|開示|説明|公表|詳しく|詳細|具体的)"
    r".{0,20}"
    r"(?:差し控え|お答えでき(?:ない|ません)|回答でき(?:ない|ません)|"
    r"コメントでき(?:ない|ません)|申し上げられ(?:ない|ません)|"
    r"非開示|開示しない)"
)
ACTION_REJECTION = re.compile(
    r"(?:予定|計画|方針|考え|意向).{0,12}(?:ない|ありません|ございません)|"
    r"(?:考えて|想定して|検討して|実施して|行って)(?:いない|おりません)"
)
TEMPORARY = re.compile(r"現時点|現在|当面|今のところ|現状")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refusal_rule(response: str) -> tuple[str | None, str | None, str | None]:
    normalized = unicodedata.normalize("NFKC", str(response))
    match = WITHHOLDING.search(normalized)
    if match:
        return "-1", "withholding", match.group(0)
    match = ACTION_REJECTION.search(normalized)
    if match:
        label = "-1" if TEMPORARY.search(normalized) else "-2"
        return label, "temporary_action_rejection" if label == "-1" else "action_rejection", match.group(0)
    return None, None, None


def gate(metrics: dict) -> dict[str, bool]:
    checks = {
        "incumbent_exactly_175_correct": metrics["incumbent_correct"] == 175,
        "candidate_at_least_179_correct": metrics["candidate_correct"] >= 179,
        "at_least_four_rescues": metrics["rescues"] >= 4,
        "at_most_one_harm": metrics["harms"] <= 1,
        "changed_beneficial_fraction_at_least_0.80": (
            metrics["changed_beneficial_fraction"] >= 0.80
        ),
        "at_least_four_refusals_correct": metrics["refusal_correct"] >= 4,
        "at_least_three_folds_improved": metrics["folds_improved"] >= 3,
        "at_most_one_fold_worsened": metrics["folds_worsened"] <= 1,
        "bootstrap_probability_positive_at_least_0.90": (
            metrics["bootstrap_probability_positive"] >= 0.90
        ),
        "bootstrap_p05_nonnegative": metrics["bootstrap_p05_delta"] >= 0.0,
        "no_current_or_hidden_test_used": True,
    }
    return checks | {"passed": all(checks.values())}


def main() -> None:
    if digest(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("refusal-rule preregistration hash mismatch")
    if digest(EXTERNAL) != EXTERNAL_SHA256:
        raise ValueError("external discovery corpus hash mismatch")

    benchmark_path = ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet"
    benchmark = pd.read_parquet(benchmark_path)
    metadata = pd.read_csv(
        ROOT / "data/derived/icr_source_folds.csv"
    ).set_index("id").loc[benchmark.id]
    ids = benchmark.id.to_numpy()
    gold = benchmark.answer.astype(str).to_numpy()
    folds = metadata.source_fold.to_numpy()
    sources = metadata.source_group.to_numpy()

    probability = (load_probabilities(7) + load_probabilities(19)) / 2
    rules = np.asarray([
        rule_prediction(query, "safe_rules") for query in benchmark["query"]
    ])
    incumbent = np.where(
        rules != "+1", rules,
        decode_probabilities(probability, ordinal_mean=True),
    )
    if int(np.sum(incumbent == gold)) != 175:
        raise ValueError("frozen incumbent no longer reproduces 175/253")

    candidate = incumbent.copy()
    families: list[str | None] = []
    matches: list[str | None] = []
    for index, query in enumerate(benchmark["query"]):
        response = parse_icr_query(query).response
        label, family, match = refusal_rule(response)
        if label is not None:
            candidate[index] = label
        families.append(family)
        matches.append(match)

    correct = candidate == gold
    incumbent_correct = incumbent == gold
    changed = candidate != incumbent
    rescues = int(np.sum(correct & ~incumbent_correct))
    harms = int(np.sum(~correct & incumbent_correct))
    fold_values = sorted(np.unique(folds))
    incumbent_fold = np.asarray([
        np.mean(incumbent[folds == fold] == gold[folds == fold])
        for fold in fold_values
    ])
    candidate_fold = np.asarray([
        np.mean(candidate[folds == fold] == gold[folds == fold])
        for fold in fold_values
    ])
    bootstrap = source_bootstrap_delta(
        candidate, incumbent, gold, sources, draws=100_000, seed=20260806,
    )
    refusal = np.isin(gold, ["-2", "-1"])
    changed_count = int(changed.sum())
    metrics = {
        "rows": len(gold),
        "incumbent_correct": int(np.sum(incumbent_correct)),
        "incumbent_accuracy": float(np.mean(incumbent_correct)),
        "candidate_correct": int(np.sum(correct)),
        "candidate_accuracy": float(np.mean(correct)),
        "changed": changed_count,
        "rescues": rescues,
        "harms": harms,
        "neutral_changes": changed_count - rescues - harms,
        "changed_beneficial_fraction": (
            rescues / changed_count if changed_count else 0.0
        ),
        "incumbent_refusal_correct": int(np.sum(incumbent[refusal] == gold[refusal])),
        "refusal_correct": int(np.sum(candidate[refusal] == gold[refusal])),
        "folds_improved": int(np.sum(candidate_fold > incumbent_fold)),
        "folds_tied": int(np.sum(candidate_fold == incumbent_fold)),
        "folds_worsened": int(np.sum(candidate_fold < incumbent_fold)),
        "fold_accuracy": {
            str(fold): float(value)
            for fold, value in zip(fold_values, candidate_fold, strict=True)
        },
        "family_counts": {
            str(family): families.count(family)
            for family in sorted({family for family in families if family})
        },
        **bootstrap,
    }
    promotion = gate(metrics)
    output = ROOT / "artifacts/icr_external_refusal_rule_v1"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "id": ids,
        "gold": gold,
        "incumbent": incumbent,
        "candidate": candidate,
        "changed": changed,
        "family": families,
        "matched_span": matches,
    }).to_csv(output / "predictions.csv", index=False)
    summary = {
        "experiment": "icr_external_corpus_refusal_rule",
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "external_corpus_sha256": EXTERNAL_SHA256,
        "benchmark_sha256": digest(benchmark_path),
        "metrics": metrics,
        "promotion_gate": promotion,
        "decision": "promote" if promotion["passed"] else "stop",
        "current_or_hidden_test_used": False,
        "post_hoc_tuning_allowed": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
