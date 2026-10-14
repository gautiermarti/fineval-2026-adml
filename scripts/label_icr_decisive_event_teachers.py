#!/usr/bin/env python3
"""Checkpoint gold-anchored decisive-event calibration through local inference.

The known training label is supplied only to ask *why* that label applies.  A
judgment is retained as structurally valid only when its label remains fixed,
its quoted spans occur in the response, and its modality factors are mutually
consistent.  Cross-teacher span/factor agreement is evaluated separately.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.icr_silver import JUDGMENT_SCHEMA, RUBRIC_VERSION, validate_judgment  # noqa: E402
from fineval.ollama import generate_json  # noqa: E402
from fineval.openai_compatible import generate_json as openai_generate_json  # noqa: E402


LABEL_GUIDE = """この課題は感情分析ではなく、会社回答が質問された将来行動に示す立場の強さを扱う。
+2 は明確で具体的かつ確定的な約束、+1 は肯定方向だが条件・留保・意向・目標を伴う、
0 は将来行動への立場を示さない説明・評価・真の未決定、-1 は時点・期間・条件に限った婉曲な拒否、
-2 は明確で最終的な拒否である。丁寧さ、努力表現、過去・現在の事実だけでは非ゼロにしない。"""

RUBRIC_GUIDES = {
    "event_factors": """質問を一つの将来命題へ直し、回答中でその命題への立場を決める最短の事象表現を探す。
source、time、conditionality、modality、actuality、evaluation、focus、responsiveness を考え、
指定JSONの対応する要因へ写像する。""",
    "ordinal_boundary": """固定ラベルと上下の隣接ラベルとの差を考える。+2/+1 は確定性、+1/0 は実際の将来行動への方向性、
0/-1 は提案への拒否、-1/-2 は再考余地・時間限定性で区別し、その境界を決める最短の事象表現を探す。""",
}


def build_prompt(row: dict) -> str:
    label = str(row["gold_label"])
    rubric = str(row["rubric"])
    if rubric not in RUBRIC_GUIDES:
        raise ValueError(f"unknown rubric {rubric!r}")
    return f"""あなたは日本企業IRと日本語語用論に精通した独立した分析者です。
{LABEL_GUIDE}

この訓練例の正解ラベルはすでに {label} と確定している。ラベルを再推定せず、なぜ {label} になるかを回答原文から分析する。
JSONのlabelには必ず {label} を入れる。

分析法:
{RUBRIC_GUIDES[rubric]}

質問:
{row['question']}

会社回答:
{row['response']}

指定されたJSON項目だけを返す。decisive_span はラベルを因果的に決める最短の連続した回答原文を一字も変更せず引用する。
逆方向の根拠が実際にあれば counterevidence_span に最短の回答原文を引用し、なければ空文字にする。
proposition と rationale は簡潔にする。"""


def label_row(row: dict, args: argparse.Namespace) -> dict:
    """Generate and validate one rubric view without mutating checkpoints."""
    prompt = build_prompt(row)
    started = time.time()
    try:
        if args.backend == "ollama":
            judgment, raw = generate_json(
                args.model, prompt, JUDGMENT_SCHEMA,
                timeout=args.timeout, num_gpu=args.num_gpu,
            )
        else:
            judgment, raw = openai_generate_json(
                args.base_url,
                args.api_key,
                args.model,
                prompt,
                JUDGMENT_SCHEMA,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                disable_thinking=True,
            )
        valid, reason = validate_judgment(judgment, row["response"])
        if valid and str(judgment["label"]) != str(row["gold_label"]):
            valid, reason = False, "teacher_changed_fixed_gold_label"
        return {
            "id": int(row["id"]),
            "source_fold": int(row["source_fold"]),
            "gold_label": str(row["gold_label"]),
            "teacher_id": args.teacher_id,
            "model": args.model,
            "model_revision": args.model_revision,
            "backend": args.backend,
            "num_gpu": args.num_gpu if args.backend == "ollama" else None,
            "rubric": str(row["rubric"]),
            "rubric_version": RUBRIC_VERSION,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "judgment": judgment,
            "valid": valid,
            "validation_reason": reason,
            "seconds": time.time() - started,
            "usage": raw.get("usage", {
                "prompt_tokens": raw.get("prompt_eval_count"),
                "completion_tokens": raw.get("eval_count"),
            }),
        }
    except Exception as error:  # Keep failures so the run is resumable/auditable.
        return {
            "id": int(row["id"]),
            "source_fold": int(row["source_fold"]),
            "gold_label": str(row["gold_label"]),
            "teacher_id": args.teacher_id,
            "model": args.model,
            "model_revision": args.model_revision,
            "backend": args.backend,
            "num_gpu": args.num_gpu if args.backend == "ollama" else None,
            "rubric": str(row["rubric"]),
            "rubric_version": RUBRIC_VERSION,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "error": repr(error),
            "valid": False,
            "validation_reason": "generation_error",
            "seconds": time.time() - started,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "artifacts/icr_decisive_event_queue_v1/teacher_prompts.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-revision", required=True,
        help="Immutable Hugging Face commit or Ollama model digest.",
    )
    parser.add_argument("--backend", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Concurrent requests; use more than one only with a batching server.",
    )
    parser.add_argument("--limit-rows", type=int)
    parser.add_argument("--ids", nargs="+", type=int)
    parser.add_argument(
        "--rubrics", nargs="+", choices=tuple(RUBRIC_GUIDES),
        default=list(RUBRIC_GUIDES),
    )
    parser.add_argument(
        "--num-gpu", type=int, default=0,
        help="Ollama GPU layers; defaults to zero so this calibration is CPU-only.",
    )
    args = parser.parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    if args.backend == "ollama" and args.concurrency != 1:
        raise ValueError("Ollama teacher runs must remain sequential")

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    selected_ids = set(args.ids or [])
    if selected_ids:
        rows = [row for row in rows if int(row["id"]) in selected_ids]
    rows = [row for row in rows if row["rubric"] in args.rubrics]
    if args.limit_rows is not None:
        retained_ids = []
        for row in rows:
            identifier = int(row["id"])
            if identifier not in retained_ids and len(retained_ids) < args.limit_rows:
                retained_ids.append(identifier)
        rows = [row for row in rows if int(row["id"]) in set(retained_ids)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[tuple[int, str]] = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            record = json.loads(line)
            completed.add((int(record["id"]), str(record["rubric"])))

    pending = [
        row for row in rows
        if (int(row["id"]), str(row["rubric"])) not in completed
    ]
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        records = executor.map(lambda row: label_row(row, args), pending)
        for record in records:
            row_id, rubric = record["id"], record["rubric"]
            with args.output.open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"id={row_id} rubric={rubric} valid={record['valid']} "
                f"reason={record['validation_reason']} seconds={record['seconds']:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
