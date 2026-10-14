#!/usr/bin/env python3
"""Run the preregistered full JF-ICR Nemotron refusal confirmation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from evaluate_icr_adapted import source_bootstrap_delta  # noqa: E402
from evaluate_icr_nemotron_japanese_pilot import (  # noqa: E402
    EXPECTED_IDS as PILOT_IDS,
    MODEL,
    MODEL_REVISION,
    call_row,
    score_records,
    sha256,
    validate_prediction,
)
from evaluate_icr_ordinal_decode import load_probabilities  # noqa: E402
from fineval.parsing import parse_icr_query  # noqa: E402
from pilot_icr_judges import guideline_prompt  # noqa: E402
from predict_icr import rule_prediction  # noqa: E402
from predict_icr_japanese_ensemble import decode_probabilities  # noqa: E402
from train_icr_japanese_cv import LABELS  # noqa: E402


PREREGISTRATION = (
    ROOT / "docs/icr_nemotron_japanese_refusal_confirmation_preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "3a26c0a68c9980957f09a4e662a044d783d3e773c3539f2390b24987dc39b535"
)
PILOT_DIR = ROOT / "artifacts/icr_nemotron_japanese_pilot_v1"
DEFAULT_OUTPUT = (
    ROOT / "artifacts/icr_nemotron_japanese_refusal_confirmation_v1"
)
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value)))


def grounded_refusal_override(
    incumbent: str,
    judgment: dict[str, Any] | None,
    response: str,
    *,
    valid: bool,
) -> tuple[str, bool, str]:
    """Apply the single frozen negative-label override."""
    if not valid or not isinstance(judgment, dict):
        return incumbent, False, "invalid_judgment"
    label = judgment.get("label")
    if label not in {"-2", "-1"}:
        return incumbent, False, "not_refusal_label"
    if judgment.get("direction") != "reject":
        return incumbent, False, "direction_not_reject"
    confidence = judgment.get("confidence")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or float(confidence) < 0.80
    ):
        return incumbent, False, "confidence_below_0.80"
    span = normalized_text(judgment.get("decisive_span", ""))
    if len(span) < 2:
        return incumbent, False, "decisive_span_too_short"
    if span not in normalized_text(response):
        return incumbent, False, "decisive_span_not_grounded"
    return str(label), True, "grounded_refusal"


def promotion_gate(metrics: dict[str, Any]) -> dict[str, bool]:
    checks = {
        "pilot_passed": bool(metrics["pilot_passed"]),
        "all_253_schema_valid": metrics["valid_judgments"] == 253,
        "incumbent_exactly_175_correct": metrics["incumbent_correct"] == 175,
        "candidate_at_least_179_correct": metrics["candidate_correct"] >= 179,
        "at_least_four_rescues": metrics["rescues"] >= 4,
        "at_most_two_harms": metrics["harms"] <= 2,
        "changed_beneficial_fraction_at_least_0.75": (
            metrics["changed_beneficial_fraction"] >= 0.75
        ),
        "at_least_7_of_13_refusals_correct": metrics["refusal_correct"] >= 7,
        "at_least_three_folds_improved": metrics["folds_improved"] >= 3,
        "at_most_one_fold_worsened": metrics["folds_worsened"] <= 1,
        "bootstrap_probability_positive_at_least_0.90": (
            metrics["bootstrap_probability_positive"] >= 0.90
        ),
        "bootstrap_p05_nonnegative": metrics["bootstrap_p05_delta"] >= 0.0,
        "no_current_or_hidden_test_used": True,
    }
    return checks | {"passed": all(checks.values())}


def load_and_validate_pilot(
    benchmark: pd.DataFrame,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    summary_path = PILOT_DIR / "summary.json"
    judgments_path = PILOT_DIR / "judgments.jsonl"
    if not summary_path.is_file() or not judgments_path.is_file():
        raise ValueError("the frozen 22-row pilot has not completed")
    summary = json.loads(summary_path.read_text())
    if (
        summary.get("decision") != "advance"
        or not summary.get("metrics", {}).get("advance_gate", {}).get("passed")
        or summary.get("model") != MODEL
        or summary.get("model_revision") != MODEL_REVISION
        or summary.get("current_or_hidden_test_used") is not False
    ):
        raise ValueError("the frozen 22-row pilot did not pass unchanged")
    records: dict[int, dict[str, Any]] = {}
    for line in judgments_path.read_text().splitlines():
        record = json.loads(line)
        identifier = int(record["id"])
        if identifier in records:
            raise ValueError("duplicate pilot judgment ID")
        records[identifier] = record
    if sorted(records) != PILOT_IDS:
        raise ValueError("pilot judgments do not cover the frozen IDs")
    indexed = benchmark.set_index("id")
    for identifier, record in records.items():
        row = indexed.loc[identifier]
        example = parse_icr_query(str(row["query"]))
        prompt = guideline_prompt(example.question, example.response)
        if (
            not record.get("valid")
            or record.get("model") != MODEL
            or record.get("model_revision") != MODEL_REVISION
            or record.get("gold") != str(row["answer"])
            or record.get("prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest()
        ):
            raise ValueError(f"pilot judgment changed at ID {identifier}")
        schema_valid, reason = validate_prediction(record.get("prediction"))
        if not schema_valid:
            raise ValueError(
                f"pilot judgment schema drift at ID {identifier}: {reason}"
            )
    recomputed_metrics = score_records(
        [records[identifier] for identifier in PILOT_IDS]
    )
    if recomputed_metrics != summary.get("metrics"):
        raise ValueError("pilot summary metrics do not match frozen judgments")
    return summary, records


def profile_metrics(
    candidate: np.ndarray,
    incumbent: np.ndarray,
    gold: np.ndarray,
    folds: np.ndarray,
    sources: np.ndarray,
) -> dict[str, Any]:
    correct = candidate == gold
    incumbent_correct = incumbent == gold
    fold_values = sorted(np.unique(folds))
    incumbent_fold = np.asarray([
        np.mean(incumbent[folds == fold] == gold[folds == fold])
        for fold in fold_values
    ])
    candidate_fold = np.asarray([
        np.mean(candidate[folds == fold] == gold[folds == fold])
        for fold in fold_values
    ])
    changed = candidate != incumbent
    rescues = int(np.sum(correct & ~incumbent_correct))
    harms = int(np.sum(~correct & incumbent_correct))
    changed_count = int(changed.sum())
    return {
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "changed": changed_count,
        "rescues": rescues,
        "harms": harms,
        "changed_beneficial_fraction": (
            rescues / changed_count if changed_count else 0.0
        ),
        "folds_improved": int(np.sum(candidate_fold > incumbent_fold)),
        "folds_tied": int(np.sum(candidate_fold == incumbent_fold)),
        "folds_worsened": int(np.sum(candidate_fold < incumbent_fold)),
        "fold_accuracy": {
            str(fold): float(value)
            for fold, value in zip(fold_values, candidate_fold, strict=True)
        },
        **source_bootstrap_delta(
            candidate, incumbent, gold, sources,
            draws=100_000, seed=20260806,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    if sha256(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("refusal confirmation preregistration hash mismatch")

    benchmark_path = ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet"
    benchmark = pd.read_parquet(benchmark_path)
    if benchmark.id.astype(int).tolist() != list(range(253)):
        raise ValueError("public benchmark ID sequence changed")
    pilot_summary, pilot_records = load_and_validate_pilot(benchmark)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = args.output_dir / "judgments.jsonl"
    completed: dict[int, dict[str, Any]] = {}
    if judgments_path.exists():
        for line in judgments_path.read_text().splitlines():
            record = json.loads(line)
            identifier = int(record["id"])
            if identifier in completed:
                raise ValueError("duplicate full-confirmation judgment ID")
            completed[identifier] = record
    unexpected_ids = sorted(set(completed) - set(range(253)))
    if unexpected_ids:
        raise ValueError(f"unexpected full-confirmation IDs: {unexpected_ids}")
    for identifier in PILOT_IDS:
        if identifier in completed and completed[identifier] != pilot_records[identifier]:
            raise ValueError(f"full output changed frozen pilot ID {identifier}")
    missing_pilot = [identifier for identifier in PILOT_IDS if identifier not in completed]
    if missing_pilot:
        with judgments_path.open("a") as handle:
            for identifier in missing_pilot:
                record = pilot_records[identifier]
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                completed[identifier] = record

    pending = [
        row.to_dict() for _, row in benchmark.iterrows()
        if int(row.id) not in completed
    ]
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        generated = executor.map(
            lambda row: call_row(
                row, args.base_url, args.api_key, args.timeout
            ),
            pending,
        )
        for record in generated:
            with judgments_path.open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            completed[int(record["id"])] = record
            print(
                f"id={record['id']} valid={record['valid']} "
                f"label={record.get('prediction', {}).get('label')}",
                flush=True,
            )

    if sorted(completed) != list(range(253)) or len(completed) != 253:
        raise ValueError("full judgments do not exactly cover IDs 0..252")
    records = [completed[identifier] for identifier in range(253)]
    ids = benchmark.id.to_numpy()
    gold = benchmark.answer.astype(str).to_numpy()
    metadata = pd.read_csv(
        ROOT / "data/derived/icr_source_folds.csv"
    ).set_index("id").loc[benchmark.id]
    folds = metadata.source_fold.to_numpy()
    sources = metadata.source_group.to_numpy()

    general7 = load_probabilities(7)
    general19 = load_probabilities(19)
    incumbent_matrix = (general7 + general19) / 2
    rules = np.asarray([
        rule_prediction(query, "safe_rules") for query in benchmark["query"]
    ])
    incumbent = np.where(
        rules != "+1",
        rules,
        decode_probabilities(incumbent_matrix, ordinal_mean=True),
    )
    if int(np.sum(incumbent == gold)) != 175:
        raise ValueError("frozen incumbent no longer reproduces 175/253")

    candidate = incumbent.copy()
    nemotron = np.full(253, "INVALID", dtype=object)
    switch = np.zeros(253, dtype=bool)
    reasons = []
    one_hot = np.full((253, len(LABELS)), 1 / len(LABELS), dtype=float)
    rows = []
    for index, record in enumerate(records):
        example = parse_icr_query(str(benchmark.iloc[index]["query"]))
        expected_prompt_sha = hashlib.sha256(
            guideline_prompt(example.question, example.response).encode()
        ).hexdigest()
        record_valid = bool(record.get("valid"))
        if (
            int(record["id"]) != index
            or record.get("model") != MODEL
            or record.get("model_revision") != MODEL_REVISION
            or record.get("gold") != str(gold[index])
            or record.get("prompt_sha256") != expected_prompt_sha
        ):
            raise ValueError(f"judgment provenance mismatch at ID {index}")
        judgment = record.get("prediction")
        if record_valid and judgment.get("label") in LABEL_TO_INDEX:
            label = str(judgment["label"])
            nemotron[index] = label
            one_hot[index] = 0.0
            one_hot[index, LABEL_TO_INDEX[label]] = 1.0
        replacement, eligible, reason = grounded_refusal_override(
            str(incumbent[index]), judgment, example.response,
            valid=record_valid,
        )
        candidate[index] = replacement
        switch[index] = eligible
        reasons.append(reason)
        rows.append({
            "id": index,
            "gold": str(gold[index]),
            "incumbent": str(incumbent[index]),
            "nemotron": str(nemotron[index]),
            "grounded_refusal": str(candidate[index]),
            "switched": bool(eligible),
            "switch_reason": reason,
            "judgment_valid": record_valid,
            "direction": judgment.get("direction") if isinstance(judgment, dict) else None,
            "confidence": judgment.get("confidence") if isinstance(judgment, dict) else None,
            "decisive_span": judgment.get("decisive_span") if isinstance(judgment, dict) else None,
        })

    candidate_profile = profile_metrics(
        candidate, incumbent, gold, folds, sources,
    )
    refusal = np.isin(gold, ["-2", "-1"])
    valid_count = sum(bool(record.get("valid")) for record in records)
    metrics = {
        "pilot_passed": bool(
            pilot_summary["metrics"]["advance_gate"]["passed"]
        ),
        "valid_judgments": valid_count,
        "incumbent_correct": int(np.sum(incumbent == gold)),
        "incumbent_accuracy": float(np.mean(incumbent == gold)),
        "incumbent_refusal_correct": int(np.sum(incumbent[refusal] == gold[refusal])),
        "candidate_correct": candidate_profile["correct"],
        "candidate_accuracy": candidate_profile["accuracy"],
        "refusal_correct": int(np.sum(candidate[refusal] == gold[refusal])),
        **{key: candidate_profile[key] for key in (
            "changed", "rescues", "harms", "changed_beneficial_fraction",
            "folds_improved", "folds_tied", "folds_worsened",
            "fold_accuracy", "bootstrap_p05_delta", "bootstrap_median_delta",
            "bootstrap_p95_delta", "bootstrap_probability_positive",
        )},
    }
    gate = promotion_gate(metrics)

    standalone_correct = int(np.sum(nemotron == gold))
    blend_matrix = (general7 + general19 + one_hot) / 3
    blend = np.where(
        rules != "+1", rules,
        decode_probabilities(blend_matrix, ordinal_mean=True),
    )
    diagnostics = {
        "nemotron_standalone": {
            "correct": standalone_correct,
            "accuracy": standalone_correct / 253,
            "not_promotion_eligible": True,
        },
        "general2_plus_nemotron_onehot": {
            **profile_metrics(blend, incumbent, gold, folds, sources),
            "not_promotion_eligible": True,
        },
    }
    pd.DataFrame(rows).to_csv(args.output_dir / "predictions.csv", index=False)
    summary = {
        "experiment": "icr_nemotron_japanese_grounded_refusal_confirmation",
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "pilot_summary_sha256": sha256(PILOT_DIR / "summary.json"),
        "benchmark_sha256": sha256(benchmark_path),
        "judgments_sha256": sha256(judgments_path),
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "rows": 253,
        "metrics": metrics,
        "promotion_gate": gate,
        "diagnostics": diagnostics,
        "decision": "promote_grounded_refusal_override" if gate["passed"] else "retain_incumbent",
        "current_or_hidden_test_used": False,
        "post_hoc_tuning_allowed": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
