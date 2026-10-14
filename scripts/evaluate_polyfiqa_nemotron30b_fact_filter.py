#!/usr/bin/env python3
"""Evaluate frozen Nemotron 30B as a conservative Task 3 fact filter."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fineval.openai_compatible import parse_json_content  # noqa: E402
from scripts.evaluate_polyfiqa_large_content_planner import (  # noqa: E402
    COMPANIES,
    FAMILIES,
    SYSTEM,
    baseline_ids,
    candidate_facts,
    fact_only,
    load_jsonl,
    ordered_lcs_f1,
    prompt_for,
    record_key,
    select_demonstrations,
    set_f1,
    summarize,
    target_ids,
)


PREREGISTRATION = (
    ROOT / "docs/polyfiqa_nemotron30b_fact_filter_preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "d34a234fb053c7c729fd5227023e7ca28405e7f3591f3410b1419e5f3d2dbe95"
)
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_REVISION = "2d59de1cbd51c0adf384eb906b766d1aee0e0517"
EXPECTED_FOLD_COUNTS = {"HON": 38, "JNJ": 30, "MSFT": 44, "UVV": 42}
SCHEMA = {
    "type": "object",
    "properties": {
        "selected": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["selected"],
    "additionalProperties": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["held_company"]),
        str(row["task_id"]),
        str(row["tier"]),
        str(row["family"]),
    )


def parse_selection(
    value: Any, offered: list[str],
) -> tuple[list[str] | None, str]:
    if not isinstance(value, dict) or set(value) != {"selected"}:
        return None, "invalid_schema"
    selected = value["selected"]
    if not isinstance(selected, list) or not all(
        isinstance(item, str) for item in selected
    ):
        return None, "invalid_selected_type"
    if len(selected) != len(set(selected)):
        return None, "duplicate_id"
    if any(item not in offered for item in selected):
        return None, "unknown_id"
    return selected, "valid"


def request_payload(prompt: str) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 2026,
        "max_tokens": 192,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "polyfiqa_fact_selection",
                "strict": True,
                "schema": SCHEMA,
            },
        },
    }


def call_row(
    row: dict[str, Any], base_url: str, api_key: str, timeout: int,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    started = time.time()
    base = {
        "held_company": row["held_company"],
        "task_id": row["task_id"],
        "tier": row["tier"],
        "family": row["family"],
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "prompt_sha256": row["prompt_sha256"],
    }
    try:
        response = requests.post(
            base_url.rstrip("/") + "/chat/completions",
            headers=headers,
            json=request_payload(row["prompt"]),
            timeout=timeout,
        )
        response.raise_for_status()
        raw = response.json()
        parsed = parse_json_content(raw["choices"][0]["message"]["content"])
        prediction, reason = parse_selection(parsed, row["offered"])
        return base | {
            "prediction": prediction,
            "valid": prediction is not None,
            "validation_reason": reason,
            "seconds": time.time() - started,
            "usage": raw.get("usage", {}),
        }
    except requests.RequestException as error:
        return base | {
            "valid": False,
            "validation_reason": "transport_error",
            "error": repr(error),
            "seconds": time.time() - started,
        }
    except Exception as error:
        return base | {
            "valid": False,
            "validation_reason": "completed_generation_error",
            "error": repr(error),
            "seconds": time.time() - started,
        }


def prepare_rows(corpus_dir: Path, baseline_path: Path) -> list[dict[str, Any]]:
    training = load_jsonl(corpus_dir / "training.jsonl")
    baseline_rows = load_jsonl(baseline_path)
    baseline_map: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in baseline_rows:
        key = identity(row)
        if key in baseline_map:
            raise ValueError(f"duplicate baseline identity: {key}")
        baseline_map[key] = row

    prepared = []
    question_cache: dict[str, pd.DataFrame] = {}
    seen: set[tuple[str, str, str, str]] = set()
    for held in COMPANIES:
        records = load_jsonl(corpus_dir / "loco" / held / "evaluation.jsonl")
        eligible = [record for record in records if fact_only(record)]
        if len(eligible) != EXPECTED_FOLD_COUNTS[held]:
            raise ValueError(
                f"unexpected {held} fact-only count: {len(eligible)}"
            )
        for record_index, record in enumerate(eligible):
            key = (held, *record_key(record))
            if key in seen:
                raise ValueError(f"duplicate evaluation identity: {key}")
            seen.add(key)
            if key not in baseline_map:
                raise ValueError(f"missing baseline identity: {key}")
            demos = select_demonstrations(
                training, held, str(record["plan"]["family"]),
            )
            prompt = prompt_for(record, demos, question_cache)
            offered = [fact["id"] for fact in candidate_facts(record)]
            incumbent = baseline_ids(baseline_map[key]["two_stage"])
            if len(offered) != len(set(offered)):
                raise ValueError(f"duplicate offered ID: {key}")
            if len(incumbent) != len(set(incumbent)) or any(
                item not in offered for item in incumbent
            ):
                raise ValueError(f"invalid incumbent selection: {key}")
            prepared.append({
                "held_company": held,
                "record_index": record_index,
                "task_id": str(record["plan"]["task_id"]),
                "tier": str(record["plan"]["tier"]),
                "family": str(record["plan"]["family"]),
                "source": str(record["source"]),
                "source_index": int(record["source_index"]),
                "target": target_ids(record),
                "offered": offered,
                "incumbent": incumbent,
                "demonstrations": [
                    {
                        "company": demo["company"],
                        "task_id": demo["plan"]["task_id"],
                        "source_index": demo["source_index"],
                    }
                    for demo in demos
                ],
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            })
    if len(prepared) != 154 or set(seen) != set(baseline_map):
        missing = sorted(set(baseline_map) - seen)
        extra = sorted(seen - set(baseline_map))
        raise ValueError(
            f"evaluation/baseline population drift: rows={len(prepared)} "
            f"missing={missing[:3]} extra={extra[:3]}"
        )
    return prepared


def validate_resume(record: dict[str, Any], expected: dict[str, Any]) -> None:
    if (
        identity(record) != identity(expected)
        or record.get("model") != MODEL
        or record.get("model_revision") != MODEL_REVISION
        or record.get("prompt_sha256") != expected["prompt_sha256"]
    ):
        raise ValueError(f"resume provenance mismatch: {identity(expected)}")
    if record.get("valid"):
        prediction, reason = parse_selection(
            {"selected": record.get("prediction")}, expected["offered"],
        )
        if prediction is None:
            raise ValueError(
                f"valid resume record failed schema: {identity(expected)} {reason}"
            )


def grounded_intersection(
    incumbent: list[str], prediction: list[str] | None, valid: bool,
) -> list[str]:
    if not valid or prediction is None:
        return list(incumbent)
    selected = set(prediction)
    return [item for item in incumbent if item in selected]


def hierarchical_bootstrap(
    rows: list[dict[str, Any]], replicates: int = 10_000,
    seed: int = 20_260_806,
) -> dict[str, float]:
    by_company = {
        company: np.asarray([
            set_f1(row["target"], row["nemotron_intersection"])
            - set_f1(row["target"], row["incumbent"])
            for row in rows if row["held_company"] == company
        ], dtype=float)
        for company in COMPANIES
    }
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=float)
    companies = np.asarray(COMPANIES)
    for index in range(replicates):
        drawn = rng.choice(companies, size=len(companies), replace=True)
        deltas = []
        for company in drawn:
            values = by_company[str(company)]
            deltas.extend(
                rng.choice(values, size=len(values), replace=True).tolist()
            )
        samples[index] = float(np.mean(deltas))
    return {
        "replicates": replicates,
        "seed": seed,
        "mean_delta": float(samples.mean()),
        "p05_delta": float(np.quantile(samples, 0.05)),
        "p95_delta": float(np.quantile(samples, 0.95)),
        "probability_gain_gt_zero": float(np.mean(samples > 0.0)),
    }


def group_deltas(
    rows: list[dict[str, Any]], field: str,
) -> dict[str, float]:
    result = {}
    for value in sorted({str(row[field]) for row in rows}):
        subset = [row for row in rows if str(row[field]) == value]
        result[value] = (
            summarize(subset, "nemotron_intersection")["macro_set_f1"]
            - summarize(subset, "incumbent")["macro_set_f1"]
        )
    return result


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        candidate = row["nemotron_intersection"]
        if any(item not in row["offered"] for item in candidate):
            raise ValueError(f"ungrounded candidate: {identity(row)}")
        expected = [
            item for item in row["incumbent"]
            if item in set(row["model_prediction"] or [])
        ] if row["model_valid"] else row["incumbent"]
        if candidate != expected:
            raise ValueError(f"intersection/order drift: {identity(row)}")

    incumbent = summarize(rows, "incumbent")
    direct = summarize(rows, "model_prediction")
    candidate = summarize(rows, "nemotron_intersection")
    deltas = [
        set_f1(row["target"], row["nemotron_intersection"])
        - set_f1(row["target"], row["incumbent"])
        for row in rows
    ]
    company_deltas = group_deltas(rows, "held_company")
    family_deltas = group_deltas(rows, "family")
    bootstrap = hierarchical_bootstrap(rows)
    valid = sum(bool(row["model_valid"]) for row in rows)
    gate = {
        "at_least_152_valid": valid >= 152,
        "macro_set_f1_gain_at_least_0_010": (
            candidate["macro_set_f1"] >= incumbent["macro_set_f1"] + 0.010
        ),
        "exact_set_accuracy_non_decreasing": (
            candidate["exact_set_accuracy"] >= incumbent["exact_set_accuracy"]
        ),
        "all_company_deltas_at_least_minus_0_010": all(
            delta >= -0.010 for delta in company_deltas.values()
        ),
        "at_least_three_company_deltas_non_negative": sum(
            delta >= 0.0 for delta in company_deltas.values()
        ) >= 3,
        "at_least_six_family_deltas_non_negative": sum(
            delta >= 0.0 for delta in family_deltas.values()
        ) >= 6,
        "all_family_deltas_at_least_minus_0_025": all(
            delta >= -0.025 for delta in family_deltas.values()
        ),
        "bootstrap_probability_at_least_0_95": (
            bootstrap["probability_gain_gt_zero"] >= 0.95
        ),
        "bootstrap_p05_non_negative": bootstrap["p05_delta"] >= 0.0,
        "grounded_order_preserving_subset_invariants": True,
        "no_current_or_hidden_test_answer_used": True,
    }
    return {
        "rows": len(rows),
        "valid_outputs": valid,
        "incumbent": incumbent,
        "direct_nemotron_diagnostic": direct,
        "nemotron_intersection": candidate,
        "delta_macro_set_f1": (
            candidate["macro_set_f1"] - incumbent["macro_set_f1"]
        ),
        "paired_changes": {
            "rescues": sum(delta > 1e-12 for delta in deltas),
            "harms": sum(delta < -1e-12 for delta in deltas),
            "ties": sum(abs(delta) <= 1e-12 for delta in deltas),
        },
        "company_deltas": company_deltas,
        "family_deltas": family_deltas,
        "hierarchical_bootstrap": bootstrap,
        "advance_gate": gate | {"passed": all(gate.values())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--corpus-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_graph_to_reference_v5",
    )
    parser.add_argument(
        "--baseline", type=Path,
        default=ROOT / "artifacts/polyfiqa_numeric_planner_v5/predictions.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_nemotron30b_fact_filter_v1",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    if sha256(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("preregistration hash mismatch")

    prepared = prepare_rows(args.corpus_dir, args.baseline)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "total_parameters": 31_577_940_288,
        "baseline_sha256": sha256(args.baseline),
        "training_sha256": sha256(args.corpus_dir / "training.jsonl"),
        "evaluation_sha256": {
            held: sha256(
                args.corpus_dir / "loco" / held / "evaluation.jsonl"
            ) for held in COMPANIES
        },
        "inference": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 2026,
            "max_tokens": 192,
            "thinking": False,
        },
        "rows": [
            {k: v for k, v in row.items() if k not in {"prompt", "target"}}
            for row in prepared
        ],
        "target_reference_exposed_to_model": False,
        "current_or_hidden_test_answer_used": False,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if manifest_path.exists() and manifest_path.read_text() != manifest_text:
        raise ValueError("existing manifest drift")
    manifest_path.write_text(manifest_text)
    if args.prepare_only:
        print(json.dumps({
            "rows": len(prepared),
            "fold_counts": EXPECTED_FOLD_COUNTS,
            "manifest_sha256": sha256(manifest_path),
            "preregistration_sha256": PREREGISTRATION_SHA256,
        }, indent=2))
        return

    expected = {identity(row): row for row in prepared}
    output_path = args.output_dir / "judgments.jsonl"
    completed: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            record = json.loads(line)
            key = identity(record)
            if key in completed:
                raise ValueError(f"duplicate resumed identity: {key}")
            if key not in expected:
                raise ValueError(f"unexpected resumed identity: {key}")
            validate_resume(record, expected[key])
            completed[key] = record

    pending = [row for row in prepared if identity(row) not in completed]
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        generated = executor.map(
            lambda row: call_row(
                row, args.base_url, args.api_key, args.timeout,
            ),
            pending,
        )
        for record in generated:
            with output_path.open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            completed[identity(record)] = record
            print(
                f"{record['held_company']} {record['family']} "
                f"valid={record['valid']} "
                f"selected={record.get('prediction')}",
                flush=True,
            )

    if set(completed) != set(expected) or len(completed) != 154:
        raise ValueError("judgments are not an exact cover of 154 frozen rows")
    scored = []
    for row in prepared:
        judgment = completed[identity(row)]
        prediction = judgment.get("prediction") if judgment.get("valid") else None
        candidate = grounded_intersection(
            row["incumbent"], prediction, bool(judgment.get("valid")),
        )
        scored.append({
            **{k: v for k, v in row.items() if k != "prompt"},
            "model_prediction": prediction,
            "model_valid": bool(judgment.get("valid")),
            "nemotron_intersection": candidate,
        })
    metrics = score_rows(scored)
    summary = {
        "experiment": "polyfiqa_nemotron30b_conservative_fact_filter",
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "manifest_sha256": sha256(manifest_path),
        "judgments_sha256": sha256(output_path),
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "total_parameters": 31_577_940_288,
        "metrics": metrics,
        "decision": (
            "advance_to_end_to_end_confirmation"
            if metrics["advance_gate"]["passed"] else "stop"
        ),
        "end_to_end_answer_claim": False,
        "target_reference_exposed_to_model": False,
        "current_or_hidden_test_answer_used": False,
        "post_hoc_tuning_allowed": False,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
