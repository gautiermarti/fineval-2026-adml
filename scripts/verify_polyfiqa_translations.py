#!/usr/bin/env python3
"""Independently verify immutable-value PolyFiQA evidence translations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.ollama import generate_json  # noqa: E402
from fineval.polyfiqa_translation import (  # noqa: E402
    VERIFICATION_SCHEMA, verification_is_acceptable,
)


def prompt_for(language: str, source: str, translation: str) -> str:
    return f"""You are an independent bilingual financial translation auditor.
Compare the {language} source with the English translation. Reject any omitted or added material claim, changed polarity, changed comparison/causality, or changed temporal scope. Values and identifiers have already passed a mechanical preservation check; assess their semantic roles as well.
Use short verbatim support quotes from both texts. A stylistic difference is acceptable only when meaning is identical.

Source:
{source}

English translation:
{translation}

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
                "name": "bilingual_financial_translation_audit", "strict": True,
                "schema": VERIFICATION_SCHEMA,
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
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verifier-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = [
        json.loads(line) for line in args.translations.read_text().splitlines() if line.strip()
    ]
    rows = [row for row in rows if row.get("valid")]
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        completed = {
            (row["row_index"], row["segment_index"])
            for row in (
                json.loads(line) for line in args.output.read_text().splitlines() if line.strip()
            )
        }
    for row in rows:
        key = (row["row_index"], row["segment_index"])
        if key in completed:
            continue
        if args.verifier_id == row["translator_id"] or args.model == row["model"]:
            raise ValueError("translator and semantic verifier must be materially different")
        prompt = prompt_for(row["language"], row["source_text"], row["translation"])
        started = time.time()
        try:
            if args.backend == "ollama":
                verdict, raw = generate_json(
                    args.model, prompt, VERIFICATION_SCHEMA, timeout=args.timeout,
                )
            else:
                verdict, raw = openai_generate(
                    args.base_url, args.api_key, args.model, prompt, args.timeout,
                )
            accepted, reason = verification_is_acceptable(
                verdict, row["source_text"], row["translation"],
            )
            record = row | {
                "verifier_id": args.verifier_id,
                "verifier_model": args.model,
                "verifier_backend": args.backend,
                "verification": verdict,
                "accepted": accepted,
                "verification_reason": reason,
                "verification_seconds": time.time() - started,
                "verification_usage": raw.get("usage", {
                    "prompt_tokens": raw.get("prompt_eval_count"),
                    "completion_tokens": raw.get("eval_count"),
                }),
            }
        except Exception as error:
            record = row | {
                "verifier_id": args.verifier_id,
                "verifier_model": args.model,
                "verifier_backend": args.backend,
                "accepted": False,
                "verification_reason": "generation_error",
                "verification_error": repr(error),
                "verification_seconds": time.time() - started,
            }
        with args.output.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"row={row['row_index']} segment={row['segment_index']} "
            f"accepted={record['accepted']} reason={record['verification_reason']}",
            flush=True,
        )


if __name__ == "__main__":
    main()

