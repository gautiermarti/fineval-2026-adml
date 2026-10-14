#!/usr/bin/env python3
"""Generate minimal modality-only JF-ICR counterfactual candidates."""

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

from fineval.icr_silver import (  # noqa: E402
    COUNTERFACTUAL_SCHEMA,
    LABELS,
    validate_counterfactual_lattice,
)
from fineval.ollama import generate_json  # noqa: E402


def choose_source_span(row: dict) -> str | None:
    """Choose the most conservative grounded quote shared by accepted judgments."""
    spans = []
    response = row["response"]
    for record in row.get("teacher_judgments", []):
        span = str(record.get("judgment", {}).get("decisive_span", "")).strip()
        if span and span in response:
            spans.append(span)
    if not spans:
        return None
    # A longer span is more likely to contain the complete Japanese modality;
    # ties are deterministic. The validator later rejects values/identifiers.
    return sorted(set(spans), key=lambda value: (-len(value), value))[0]


def prompt_for(row: dict, source_span: str) -> str:
    return f"""あなたは日本企業IRの自然な日本語を編集する専門家です。
以下の会社回答について、指定された原文一箇所だけを置換し、同じ将来命題に対するコミットメント強度の最小対を5種類作成してください。

絶対条件:
- 質問、トピック、事実、数値、固有名詞は変えない。
- source_span以外を一文字も書き換えない。
- replacement_spanだけを出力し、回答全文を生成しない。
- +2は明確で確定的な約束、+1は条件・意向・限定を伴う肯定、0は将来行動への立場なし、-1は時間・条件付き拒否、-2は明確で最終的な拒否。
- 丁寧な努力表現だけを+2にしない。
- 五つのreplacement_spanは自然な日本語で、同じ命題を扱う。

質問:
{row['question']}

会社回答:
{row['response']}

変更対象の原文:
{source_span}

JSONのsource_spanには変更対象の原文を完全にコピーし、各ラベルを一度ずつ出力してください。"""


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
        "max_tokens": 1_200,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "icr_counterfactual_lattice",
                "strict": True,
                "schema": COUNTERFACTUAL_SCHEMA,
            },
        },
    }
    result = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    result.raise_for_status()
    raw = result.json()
    return parse_json_content(raw["choices"][0]["message"]["content"]), raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        completed = {
            json.loads(line)["pair_id"]
            for line in args.output.read_text().splitlines() if line.strip()
        }
    for row in rows:
        if row["pair_id"] in completed:
            continue
        source_span = choose_source_span(row)
        started = time.time()
        if source_span is None:
            record = {
                "pair_id": row["pair_id"], "valid": False,
                "validation_reason": "no_exact_decisive_span",
            }
        else:
            prompt = prompt_for(row, source_span)
            try:
                if args.backend == "ollama":
                    lattice, raw = generate_json(
                        args.model, prompt, COUNTERFACTUAL_SCHEMA, timeout=args.timeout,
                    )
                else:
                    lattice, raw = openai_generate(
                        args.base_url, args.api_key, args.model, prompt, args.timeout,
                    )
                valid, reason, variants = validate_counterfactual_lattice(
                    lattice, row["response"],
                )
                record = {
                    "pair_id": row["pair_id"],
                    "issuer": row.get("issuer"),
                    "question": row["question"],
                    "original_response": row["response"],
                    "original_silver_label": row.get("silver_label"),
                    "generator_id": args.generator_id,
                    "model": args.model,
                    "backend": args.backend,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "raw_lattice": lattice,
                    "variants": variants,
                    "valid": valid,
                    "validation_reason": reason,
                    "seconds": time.time() - started,
                    "usage": raw.get("usage", {
                        "prompt_tokens": raw.get("prompt_eval_count"),
                        "completion_tokens": raw.get("eval_count"),
                    }),
                }
            except Exception as error:
                record = {
                    "pair_id": row["pair_id"], "issuer": row.get("issuer"),
                    "generator_id": args.generator_id, "model": args.model,
                    "backend": args.backend, "valid": False,
                    "validation_reason": "generation_error", "error": repr(error),
                    "seconds": time.time() - started,
                }
        with args.output.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"pair={row['pair_id'][:10]} valid={record['valid']} "
            f"reason={record['validation_reason']}", flush=True,
        )

    # Emit verifier-ready rows. A materially different teacher must recover
    # target_label later; this generator never self-certifies its own labels.
    verifier_path = args.output.with_suffix(".verifier.jsonl")
    records = [json.loads(line) for line in args.output.read_text().splitlines() if line.strip()]
    with verifier_path.open("w") as handle:
        for record in records:
            if not record.get("valid"):
                continue
            for variant in record["variants"]:
                row = {
                    "pair_id": f"{record['pair_id']}::{variant['target_label']}",
                    "source_pair_id": record["pair_id"],
                    "issuer": record.get("issuer"),
                    "question": record["question"],
                    "response": variant["response"],
                    "target_label": variant["target_label"],
                    "generator_id": record["generator_id"],
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

