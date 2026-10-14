#!/usr/bin/env python3
"""Translate selected PolyFiQA evidence with immutable values and audit logs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.ollama import generate_json  # noqa: E402
from fineval.polyfiqa_translation import (  # noqa: E402
    TRANSLATION_SCHEMA, extract_translatable_evidence, mask_immutables,
    validate_translation,
)


def prompt_for(language: str, masked_source: str) -> str:
    return f"""Translate the following {language} financial-news evidence into concise professional English.
Preserve every ZXQIMM...QXZ placeholder exactly once and in the same semantic role. Do not translate, alter, reorder, delete, or invent a placeholder. Do not introduce any digits.
Preserve every material claim, polarity, comparison, causal relation, and time scope. Do not add interpretation. Use no more than 45 whitespace-delimited English words.
Count the independently asserted source claims and the translated claims; the counts must match.

Source evidence:
{masked_source}

Return only the requested JSON object."""


def parse_json_content(content: str) -> dict:
    value = content.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(value)


def openai_generate(base_url: str, api_key: str, model: str, prompt: str, timeout: int):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 500,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "immutable_financial_translation", "strict": True,
                "schema": TRANSLATION_SCHEMA,
            },
        },
    }
    result = requests.post(
        base_url.rstrip("/") + "/chat/completions", headers=headers,
        json=payload, timeout=timeout,
    )
    result.raise_for_status()
    raw = result.json()
    return parse_json_content(raw["choices"][0]["message"]["content"]), raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--translator-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    predictions = [
        json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()
    ]
    segments = []
    for row_index, row in enumerate(predictions):
        for segment_index, segment in enumerate(extract_translatable_evidence(row["answer"])):
            segments.append((row_index, segment_index, row, segment))
    if args.limit is not None:
        segments = segments[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        completed = {
            (record["row_index"], record["segment_index"])
            for record in (
                json.loads(line) for line in args.output.read_text().splitlines() if line.strip()
            )
        }
    for row_index, segment_index, row, segment in segments:
        if (row_index, segment_index) in completed:
            continue
        masked = mask_immutables(segment["text"])
        prompt = prompt_for(segment["language"], masked.masked)
        started = time.time()
        try:
            if args.backend == "ollama":
                result, raw = generate_json(
                    args.model, prompt, TRANSLATION_SCHEMA, timeout=args.timeout,
                )
            else:
                result, raw = openai_generate(
                    args.base_url, args.api_key, args.model, prompt, args.timeout,
                )
            valid, reason, restored = validate_translation(result, masked)
            record = {
                "row_index": row_index, "segment_index": segment_index,
                "task_id": row["task_id"], "language": segment["language"],
                "source_text": segment["text"], "source_start": segment["start"],
                "source_end": segment["end"], "source_label": segment["label"],
                "source_sha256": hashlib.sha256(segment["text"].encode()).hexdigest(),
                "masked_source": masked.masked, "immutable_mapping": masked.mapping,
                "translator_id": args.translator_id, "model": args.model,
                "backend": args.backend, "raw_result": result,
                "translation": restored, "valid": valid,
                "validation_reason": reason, "seconds": time.time() - started,
                "usage": raw.get("usage", {
                    "prompt_tokens": raw.get("prompt_eval_count"),
                    "completion_tokens": raw.get("eval_count"),
                }),
            }
        except Exception as error:
            record = {
                "row_index": row_index, "segment_index": segment_index,
                "task_id": row["task_id"], "language": segment["language"],
                "source_text": segment["text"], "source_start": segment["start"],
                "source_end": segment["end"], "source_label": segment["label"],
                "translator_id": args.translator_id, "model": args.model,
                "backend": args.backend, "valid": False,
                "validation_reason": "generation_error", "error": repr(error),
                "seconds": time.time() - started,
            }
        with args.output.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"row={row_index} segment={segment_index} language={segment['language']} "
            f"valid={record['valid']} reason={record['validation_reason']}", flush=True,
        )


if __name__ == "__main__":
    main()

