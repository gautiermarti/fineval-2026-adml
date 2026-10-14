#!/usr/bin/env python3
"""Checkpoint structured ICR teacher judgments via Ollama or an OpenAI API."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.icr_silver import JUDGMENT_SCHEMA, RUBRIC_VERSION, validate_judgment  # noqa: E402
from fineval.ollama import generate_json  # noqa: E402


BASE_RUBRIC = """JF-ICRは感情分析ではない。会社回答が表す将来の行動、決定、方針、目標、戦略的方向性へのコミットメントまたは拒否の強さを判定する。
+2: 明確で具体的かつ確定的な約束。+1: 肯定方向だが条件、留保、意向、目標、限定性を伴う。
0: 将来行動への立場を示さない説明・評価・真の未決定。-1: 現時点、期間、条件に限った婉曲な拒否。-2: 明確で最終的な拒否。
丁寧な語調、努力表現、過去・現在の事実だけでは非ゼロにしない。回答全体の支配的立場を用いる。"""

RUBRICS = {
    "factor_first": """最初に質問を肯定・否定できる将来命題へ直す。次に回答中の将来行動、方向、強さ、条件性、時間的範囲、応答性を独立に判定し、最後にだけラベルへ写像する。決定的根拠は回答から原文のまま短く引用する。反対方向の根拠がなければcounterevidence_spanは空文字にする。""",
    "boundary_contrast": """候補ラベルを決めた後、必ず隣接ラベルとの差を検討する。+2と+1は確定性、+1と0は実際の将来行動への方向性、0と-1は提案への拒否、-1と-2は再考余地・時間限定性で区別する。決定的根拠は回答から原文のまま短く引用する。反証がなければcounterevidence_spanは空文字にする。""",
    "answer_tail_audit": """回答末尾の述語、助動詞、否定、条件表現を先に監査し、その後に回答全体で覆されていないか確認する。末尾だけで機械的に決めず、将来行動への支配的立場を構造化してからラベルへ写像する。決定的根拠は回答の原文を引用する。""",
}


def build_prompt(question: str, response: str, rubric: str) -> str:
    return f"""あなたは日本企業IRと日本語語用論に精通した独立アノテータです。
{BASE_RUBRIC}

判定手順:
{RUBRICS[rubric]}

質問:
{question}

会社回答:
{response}

指定されたJSON項目だけを返してください。rationaleは簡潔にし、decisive_spanとcounterevidence_spanは会社回答の原文以外を作らないでください。"""


def parse_json_content(content: str) -> dict[str, Any]:
    value = content.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1]
        value = value.rsplit("```", 1)[0]
    return json.loads(value)


def openai_generate(
    base_url: str, api_key: str, model: str, prompt: str, timeout: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 900,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "icr_structured_judgment",
                "strict": True,
                "schema": JUDGMENT_SCHEMA,
            },
        },
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    raw = response.json()
    return parse_json_content(raw["choices"][0]["message"]["content"]), raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "data/external/japanese_ir_qa/forward_candidates.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--backend", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rubrics", nargs="+", choices=tuple(RUBRICS), default=list(RUBRICS))
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows_by_id = {row["pair_id"]: row for row in rows}
    completed: set[tuple[str, str]] = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            record = json.loads(line)
            pair_id = record["pair_id"]
            rubric = record["rubric"]
            key = (pair_id, rubric)
            if key in completed:
                raise ValueError(f"duplicate resumed judgment: {key}")
            if pair_id not in rows_by_id or rubric not in RUBRICS:
                raise ValueError(f"unexpected resumed judgment: {key}")
            source = rows_by_id[pair_id]
            expected_prompt = build_prompt(
                source["question"], source["response"], rubric,
            )
            if (
                record.get("teacher_id") != args.teacher_id
                or record.get("model") != args.model
                or record.get("model_revision") != args.model_revision
                or record.get("backend") != args.backend
                or record.get("rubric_version") != RUBRIC_VERSION
                or record.get("prompt_sha256")
                != hashlib.sha256(expected_prompt.encode()).hexdigest()
            ):
                raise ValueError(f"resumed judgment provenance mismatch: {key}")
            completed.add(key)

    for row in rows:
        for rubric in args.rubrics:
            key = (row["pair_id"], rubric)
            if key in completed:
                continue
            prompt = build_prompt(row["question"], row["response"], rubric)
            started = time.time()
            try:
                if args.backend == "ollama":
                    judgment, raw = generate_json(
                        args.model, prompt, JUDGMENT_SCHEMA, timeout=args.timeout,
                    )
                else:
                    judgment, raw = openai_generate(
                        args.base_url, args.api_key, args.model, prompt, args.timeout,
                    )
                valid, validation_reason = validate_judgment(judgment, row["response"])
                record = {
                    "pair_id": row["pair_id"],
                    "issuer": row.get("issuer"),
                    "teacher_id": args.teacher_id,
                    "model": args.model,
                    "model_revision": args.model_revision,
                    "backend": args.backend,
                    "rubric": rubric,
                    "rubric_version": RUBRIC_VERSION,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "judgment": judgment,
                    "valid": valid,
                    "validation_reason": validation_reason,
                    "seconds": time.time() - started,
                    "usage": raw.get("usage", {
                        "prompt_tokens": raw.get("prompt_eval_count"),
                        "completion_tokens": raw.get("eval_count"),
                    }),
                }
            except Exception as error:  # Preserve failures for resumable auditing.
                record = {
                    "pair_id": row["pair_id"],
                    "issuer": row.get("issuer"),
                    "teacher_id": args.teacher_id,
                    "model": args.model,
                    "model_revision": args.model_revision,
                    "backend": args.backend,
                    "rubric": rubric,
                    "rubric_version": RUBRIC_VERSION,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "error": repr(error),
                    "valid": False,
                    "validation_reason": "generation_error",
                    "seconds": time.time() - started,
                }
            with args.output.open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"pair={row['pair_id'][:10]} rubric={rubric} "
                f"valid={record['valid']} reason={record['validation_reason']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
