#!/usr/bin/env python3
"""Checkpointed direct/few-shot local-LLM pilot for PolyFiQA-Expert."""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.expert import retrieve_statement_rows  # noqa: E402
from fineval.metrics import rouge1_f1  # noqa: E402
from fineval.ollama import generate_json  # noqa: E402
from fineval.parsing import parse_polyfiqa_query  # noqa: E402

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


def family(question: str) -> str:
    if question.startswith("Please list"):
        return "revenue_focus"
    if question.startswith("How is"):
        return "capital_allocation"
    if question.startswith("What is the company’s strategy"):
        return "margin"
    return "capex"


def packet(row: pd.Series, news: dict) -> str:
    context = parse_polyfiqa_query(row["query"])
    statement_rows = retrieve_statement_rows(context.financial_statements, row["question"], 8)
    news_rows = news[row["task_id"]][family(row["question"])][:6]
    return (
        "QUESTION\n" + row["question"]
        + "\n\nRELEVANT FINANCIAL-STATEMENT ROWS\n- " + "\n- ".join(statement_rows)
        + "\n\nRETRIEVED MULTILINGUAL NEWS\n- "
        + "\n- ".join(f"[{item['language']}] {item['text']}" for item in news_rows)
    )


def prompt(
    row: pd.Series,
    news: dict,
    exemplar: pd.Series | None = None,
    draft: str | None = None,
) -> str:
    demonstration = ""
    if exemplar is not None:
        demonstration = (
            "\nHere is one labeled example from a different company. Learn its concision and "
            "format, not its facts.\n" + packet(exemplar, news)
            + "\nREFERENCE OUTPUT\n" + exemplar["answer"] + "\n"
        )
    editing = ""
    if draft is not None:
        editing = f"""
DETERMINISTIC DRAFT
{draft}

Conservatively edit the deterministic draft. Preserve every supported concrete fact and
its required two-section format. Add or rephrase content only when the supplied evidence
clearly supports it. Prefer wording from the evidence, since evaluation rewards lexical
coverage. Do not replace a specific supported draft with generic commentary.
"""
    return f"""You are a careful financial analyst solving PolyFiQA-Expert.
Use only the supplied rows and news. Never introduce an unsupported number.
Answer the question from news, then provide concise supporting financial-statement facts.
Return no calculation steps and no meta-commentary. The answer string must use exactly:
Answer: <concise answer>
Financial Statement Evidence: <supported financial facts, or None.>
Keep the complete output under 110 words. Preserve requested top-three numbering.
{demonstration}
TARGET
{packet(row, news)}
{editing}"""


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default="qwen3.6:35b-a3b")
    parser.add_argument("--per-family", type=int, default=2)
    parser.add_argument(
        "--families", nargs="+",
        choices=("revenue_focus", "capital_allocation", "margin", "capex"),
        default=("revenue_focus", "capital_allocation", "margin", "capex"),
    )
    parser.add_argument(
        "--tracks", nargs="+", choices=(
            "direct", "cross_company_one_shot", "conservative_editor", "conservative_editor_v2",
        ),
        default=("direct", "cross_company_one_shot", "conservative_editor_v2"),
    )
    args = parser.parse_args()
    frame = pd.read_parquet(ROOT / "data/raw/polyfiqa_expert/test-00000-of-00001.parquet")
    frame = frame.assign(family=frame["question"].map(family), company=frame["task_id"].str.split("_").str[0])
    news_path = ROOT / "artifacts/polyfiqa_expert/news_candidates.jsonl"
    news = {
        record["task_id"]: record["families"]
        for record in map(json.loads, news_path.read_text().splitlines())
    }
    deterministic = [
        record["answer"] for record in map(
            json.loads,
            (ROOT / "artifacts/polyfiqa_expert/predictions.jsonl").read_text().splitlines(),
        )
    ]
    if len(deterministic) != len(frame):
        raise ValueError("deterministic predictions do not align with the Expert frame")
    eligible = frame[frame["family"].isin(args.families)]
    pilot = pd.concat([
        group.sample(min(args.per_family, len(group)), random_state=2026)
        for _, group in eligible.groupby("family")
    ]).sort_index()
    safe_model = args.model.replace(":", "_").replace("/", "_")
    output = ROOT / f"artifacts/polyfiqa_expert/pilot_{safe_model}.jsonl"
    completed = set()
    if output.exists():
        completed = {
            (record["row_index"], record["track"])
            for record in map(json.loads, output.read_text().splitlines())
        }
    for row_index, row in pilot.iterrows():
        alternatives = frame[(frame["family"] == row["family"]) & (frame["company"] != row["company"])]
        exemplar = alternatives.iloc[0]
        track_inputs = {
            "direct": (None, None),
            "cross_company_one_shot": (exemplar, None),
            # v1 incorrectly keyed drafts by the non-unique task_id; retain its name
            # only so the contaminated checkpoint remains readable.
            "conservative_editor": (None, deterministic[int(row_index)]),
            "conservative_editor_v2": (None, deterministic[int(row_index)]),
        }
        for track in args.tracks:
            exemplar_row, draft = track_inputs[track]
            if (int(row_index), track) in completed:
                continue
            started = time.time()
            prediction, raw = generate_json(
                args.model, prompt(row, news, exemplar_row, draft), ANSWER_SCHEMA,
            )
            record = {
                "row_index": int(row_index), "task_id": row["task_id"], "family": row["family"],
                "track": track, "model": args.model, "gold": row["answer"],
                "prediction": prediction["answer"], "seconds": time.time() - started,
                "prompt_tokens": raw.get("prompt_eval_count"), "output_tokens": raw.get("eval_count"),
            }
            with output.open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(row["task_id"], row["family"], track, f"rouge={rouge1_f1([row['answer']], [prediction['answer']]):.3f}", flush=True)
    results = pd.DataFrame(map(json.loads, output.read_text().splitlines()))
    results = results[results["row_index"].isin(pilot.index)]
    for track, group in results.groupby("track"):
        print(track, len(group), f"rouge={rouge1_f1(group['gold'].tolist(), group['prediction'].tolist()):.6f}")


if __name__ == "__main__":
    main()
