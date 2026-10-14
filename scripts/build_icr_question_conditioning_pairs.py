#!/usr/bin/env python3
"""Find similar company responses whose labels differ because questions differ."""

from pathlib import Path
import sys

import pandas as pd
from rapidfuzz.fuzz import ratio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.parsing import parse_icr_query  # noqa: E402


def normalize(text: str) -> str:
    return "".join(text.split()).replace("・", "")


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    parsed = frame["query"].map(parse_icr_query)
    rows = []
    for left in range(len(frame)):
        for right in range(left + 1, len(frame)):
            if frame.iloc[left]["answer"] == frame.iloc[right]["answer"]:
                continue
            similarity = ratio(
                normalize(parsed.iloc[left].response),
                normalize(parsed.iloc[right].response),
            ) / 100
            if similarity < 0.65:
                continue
            rows.append(
                {
                    "response_similarity": similarity,
                    "left_id": int(frame.iloc[left]["id"]),
                    "left_label": frame.iloc[left]["answer"],
                    "left_question": parsed.iloc[left].question,
                    "left_response": parsed.iloc[left].response,
                    "right_id": int(frame.iloc[right]["id"]),
                    "right_label": frame.iloc[right]["answer"],
                    "right_question": parsed.iloc[right].question,
                    "right_response": parsed.iloc[right].response,
                }
            )
    pairs = pd.DataFrame(rows).sort_values("response_similarity", ascending=False)
    output = ROOT / "data/derived/icr_question_conditioning_pairs.csv"
    pairs.to_csv(output, index=False)
    print(f"pairs={len(pairs)} output={output}")
    if len(pairs):
        print(pairs[["response_similarity", "left_id", "left_label", "right_id", "right_label"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()

