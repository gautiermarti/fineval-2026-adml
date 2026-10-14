#!/usr/bin/env python3
"""CPU-only cached 4B prompt screen for PolyFiQA-Expert."""

from pathlib import Path
import json
import sys
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.metrics import rouge1_f1  # noqa: E402
from pilot_polyfiqa_expert_llm import family, prompt  # noqa: E402

MODEL = "FutureMa/Eva-4B"


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/raw/polyfiqa_expert/test-00000-of-00001.parquet")
    frame = frame.assign(family=frame["question"].map(family), company=frame["task_id"].str.split("_").str[0])
    pilot = frame.groupby("family", sort=True).sample(1, random_state=2026).sort_index()
    news = {
        record["task_id"]: record["families"]
        for record in map(
            json.loads,
            (ROOT / "artifacts/polyfiqa_expert/news_candidates.jsonl").read_text().splitlines(),
        )
    }
    output = ROOT / "artifacts/polyfiqa_expert/pilot_eva4b_cpu.jsonl"
    completed = set()
    if output.exists():
        completed = {
            (record["row_index"], record["track"])
            for record in map(json.loads, output.read_text().splitlines())
        }
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16,
    )
    for row_index, row in pilot.iterrows():
        alternatives = frame[(frame["family"] == row["family"]) & (frame["company"] != row["company"])]
        exemplar = alternatives.iloc[0]
        for track, exemplar_row in (("direct", None), ("cross_company_one_shot", exemplar)):
            if (int(row_index), track) in completed:
                continue
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt(row, news, exemplar_row)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            inputs = tokenizer(chat, return_tensors="pt")
            started = time.time()
            generated = model.generate(**inputs, max_new_tokens=160, do_sample=False)
            answer = tokenizer.decode(
                generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True,
            ).strip()
            record = {
                "row_index": int(row_index), "task_id": row["task_id"], "family": row["family"],
                "track": track, "gold": row["answer"], "prediction": answer,
                "seconds": time.time() - started,
            }
            with output.open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(row["task_id"], row["family"], track,
                  f"rouge={rouge1_f1([row['answer']], [answer]):.3f}", flush=True)
    results = pd.DataFrame(map(json.loads, output.read_text().splitlines()))
    results = results[results["row_index"].isin(pilot.index)]
    for track, group in results.groupby("track"):
        print(track, f"rouge={rouge1_f1(group['gold'].tolist(), group['prediction'].tolist()):.6f}")


if __name__ == "__main__":
    main()
