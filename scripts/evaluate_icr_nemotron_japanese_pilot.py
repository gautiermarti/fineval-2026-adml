#!/usr/bin/env python3
"""Run and score the preregistered Japanese Nemotron Task 2 pilot."""

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

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fineval.openai_compatible import parse_json_content  # noqa: E402
from scripts.pilot_icr_judges import (  # noqa: E402
    STRUCTURED_SCHEMA,
    guideline_prompt,
    select_pilot,
)
from fineval.parsing import parse_icr_query  # noqa: E402


PREREGISTRATION = ROOT / "docs/icr_nemotron_japanese_pilot_preregistration.md"
PREREGISTRATION_SHA256 = (
    "d9429e5a5d5a7e4c5ec93ea4b10df1ad2f18d50a5a8d8429187248a9748cc315"
)
MODEL = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese"
MODEL_REVISION = "3979dd16634988c34cc3bd911583c51e6a731d10"
EXPECTED_IDS = [
    25, 61, 86, 87, 90, 91, 95, 97, 107, 108, 109, 110, 117, 118,
    126, 129, 157, 176, 189, 196, 209, 243,
]
LABELS = ["-2", "-1", "0", "+1", "+2"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_prediction(value: Any) -> tuple[bool, str]:
    if not isinstance(value, dict):
        return False, "not_object"
    required = set(STRUCTURED_SCHEMA["required"])
    missing = sorted(required - set(value))
    if missing:
        return False, "missing:" + ",".join(missing)
    # The frozen request schema predates an explicit `additionalProperties`
    # declaration. Enforce its exact property set locally so a provider cannot
    # turn an extra free-form field into a silently "valid" judgment.
    unexpected = sorted(set(value) - set(STRUCTURED_SCHEMA["properties"]))
    if unexpected:
        return False, "unexpected:" + ",".join(unexpected)
    for name, specification in STRUCTURED_SCHEMA["properties"].items():
        item = value[name]
        if specification["type"] == "string" and not isinstance(item, str):
            return False, f"wrong_type:{name}"
        if specification["type"] == "number" and (
            not isinstance(item, (int, float)) or isinstance(item, bool)
        ):
            return False, f"wrong_type:{name}"
        if specification["type"] == "number" and not math.isfinite(float(item)):
            return False, f"non_finite:{name}"
        if "enum" in specification and item not in specification["enum"]:
            return False, f"bad_enum:{name}"
        if specification["type"] == "number" and not (
            float(specification.get("minimum", item))
            <= float(item)
            <= float(specification.get("maximum", item))
        ):
            return False, f"out_of_range:{name}"
    return True, "valid"


def validate_record_provenance(
    record: dict[str, Any], row: dict[str, Any],
) -> None:
    """Fail closed when a resumed record came from another run or input."""
    example = parse_icr_query(str(row["query"]))
    prompt = guideline_prompt(example.question, example.response)
    identifier = int(row["id"])
    if (
        int(record.get("id", -1)) != identifier
        or record.get("gold") != str(row["answer"])
        or record.get("model") != MODEL
        or record.get("model_revision") != MODEL_REVISION
        or record.get("prompt_sha256")
        != hashlib.sha256(prompt.encode()).hexdigest()
    ):
        raise ValueError(f"pilot record provenance mismatch at ID {identifier}")
    if record.get("valid"):
        valid, reason = validate_prediction(record.get("prediction"))
        if not valid:
            raise ValueError(
                f"pilot record marked valid but schema-invalid at ID "
                f"{identifier}: {reason}"
            )


def request_payload(prompt: str) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "top_p": 0.95,
        "seed": 2026,
        "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "icr_structured_judgment",
                "strict": True,
                "schema": STRUCTURED_SCHEMA,
            },
        },
    }


def call_row(
    row: dict[str, Any], base_url: str, api_key: str, timeout: int,
) -> dict[str, Any]:
    example = parse_icr_query(str(row["query"]))
    prompt = guideline_prompt(example.question, example.response)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    started = time.time()
    transport_error = None
    for attempt in range(2):
        try:
            response = requests.post(
                base_url.rstrip("/") + "/chat/completions",
                headers=headers,
                json=request_payload(prompt),
                timeout=timeout,
            )
            response.raise_for_status()
            raw = response.json()
            prediction = parse_json_content(
                raw["choices"][0]["message"]["content"]
            )
            valid, reason = validate_prediction(prediction)
            return {
                "id": int(row["id"]),
                "gold": str(row["answer"]),
                "model": MODEL,
                "model_revision": MODEL_REVISION,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prediction": prediction,
                "valid": valid,
                "validation_reason": reason,
                "seconds": time.time() - started,
                "transport_attempts": attempt + 1,
                "usage": raw.get("usage", {}),
            }
        except requests.RequestException as error:
            transport_error = repr(error)
            continue
        except Exception as error:
            return {
                "id": int(row["id"]),
                "gold": str(row["answer"]),
                "model": MODEL,
                "model_revision": MODEL_REVISION,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "valid": False,
                "validation_reason": "completed_generation_error",
                "error": repr(error),
                "seconds": time.time() - started,
                "transport_attempts": attempt + 1,
            }
    return {
        "id": int(row["id"]),
        "gold": str(row["answer"]),
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "valid": False,
        "validation_reason": "transport_error",
        "error": transport_error,
        "seconds": time.time() - started,
        "transport_attempts": 2,
    }


def score_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {int(record["id"]): record for record in records}
    if sorted(by_id) != EXPECTED_IDS or len(records) != len(EXPECTED_IDS):
        raise ValueError("pilot records are not an exact unique cover of frozen IDs")
    correct = 0
    valid = 0
    class_metrics = {}
    for label in LABELS:
        rows = [by_id[identifier] for identifier in EXPECTED_IDS
                if str(by_id[identifier]["gold"]) == label]
        label_correct = sum(
            bool(row.get("valid"))
            and str(row.get("prediction", {}).get("label")) == label
            for row in rows
        )
        class_metrics[label] = {
            "rows": len(rows),
            "correct": label_correct,
            "recall": label_correct / len(rows) if rows else 0.0,
        }
        correct += label_correct
        valid += sum(bool(row.get("valid")) for row in rows)
    negative_correct = sum(class_metrics[label]["correct"] for label in ["-2", "-1"])
    labels_with_nonzero_recall = sum(
        value["correct"] > 0 for value in class_metrics.values()
    )
    gate = {
        "all_22_schema_valid": valid == 22,
        "at_least_13_correct": correct >= 13,
        "at_least_7_of_13_negative_correct": negative_correct >= 7,
        "at_least_three_labels_nonzero_recall": labels_with_nonzero_recall >= 3,
        "no_current_or_hidden_test_used": True,
    }
    return {
        "rows": 22,
        "valid": valid,
        "correct": correct,
        "accuracy": correct / 22,
        "negative_correct": negative_correct,
        "labels_with_nonzero_recall": labels_with_nonzero_recall,
        "class_metrics": class_metrics,
        "advance_gate": gate | {"passed": all(gate.values())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/icr_nemotron_japanese_pilot_v1",
    )
    args = parser.parse_args()
    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    if sha256(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("preregistration hash mismatch")

    benchmark = ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet"
    frame = pd.read_parquet(benchmark)
    pilot = select_pilot(frame, per_common_class=3)
    if pilot.id.astype(int).tolist() != EXPECTED_IDS:
        raise ValueError("frozen pilot selection drift")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "judgments.jsonl"
    completed: dict[int, dict[str, Any]] = {}
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            record = json.loads(line)
            identifier = int(record["id"])
            if identifier in completed:
                raise ValueError(f"duplicate resumed pilot ID {identifier}")
            completed[identifier] = record
    pilot_by_id = {
        int(row.id): row.to_dict() for _, row in pilot.iterrows()
    }
    unexpected_ids = sorted(set(completed) - set(pilot_by_id))
    if unexpected_ids:
        raise ValueError(f"unexpected resumed pilot IDs: {unexpected_ids}")
    for identifier, record in completed.items():
        validate_record_provenance(record, pilot_by_id[identifier])
    pending = [row.to_dict() for _, row in pilot.iterrows()
               if int(row.id) not in completed]
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        generated = executor.map(
            lambda row: call_row(row, args.base_url, args.api_key, args.timeout),
            pending,
        )
        for record in generated:
            with output_path.open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            completed[int(record["id"])] = record
            print(
                f"id={record['id']} valid={record['valid']} "
                f"label={record.get('prediction', {}).get('label')}",
                flush=True,
            )

    ordered = [completed[identifier] for identifier in EXPECTED_IDS]
    metrics = score_records(ordered)
    summary = {
        "experiment": "icr_nemotron_japanese_guideline_pilot",
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "benchmark_sha256": sha256(benchmark),
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "total_parameters": 8888227328,
        "inference": {
            "temperature": 0.6,
            "top_p": 0.95,
            "seed": 2026,
            "max_tokens": 2048,
            "thinking": True,
        },
        "qwen_guideline_comparator_correct": 11,
        "qwen_guideline_comparator_rows": 22,
        "metrics": metrics,
        "decision": "advance" if metrics["advance_gate"]["passed"] else "stop",
        "current_or_hidden_test_used": False,
        "post_hoc_tuning_allowed": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
