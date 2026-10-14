#!/usr/bin/env python3
"""Index forward-looking language in EDINET blocks without using task labels."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
OUTLOOK_CUES = (
    "見通し", "今後", "計画", "リスク", "業績", "成長", "方針", "予想",
    "可能性", "不透明", "見込", "想定", "継続", "目標", "課題",
)


def cue_count(sequence, patterns) -> int:
    values = [int(token) for token in sequence]
    singles = set(values)
    pairs = set(zip(values, values[1:]))
    count = 0
    for pattern in patterns:
        if len(pattern) == 1:
            count += int(pattern[0] in singles)
        elif len(pattern) == 2:
            count += int(tuple(pattern) in pairs)
        else:
            count += int(any(
                tuple(values[index:index + len(pattern)]) == tuple(pattern)
                for index in range(len(values) - len(pattern) + 1)
            ))
    return count


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "data/derived/icr_edinet2023_mlm.parquet",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/derived/icr_edinet2023_outlook_index.parquet",
    )
    parser.add_argument("--model", default="cl-tohoku/bert-base-japanese-v3")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    patterns = [tokenizer(cue, add_special_tokens=False)["input_ids"] for cue in OUTLOOK_CUES]
    frame = pd.read_parquet(args.input)
    counts = [cue_count(sequence, patterns) for sequence in frame.input_ids]
    result = frame[["sha256", "company", "split", "token_count"]].copy()
    result["outlook_cue_count"] = counts
    result["outlook_rich"] = result.outlook_cue_count >= 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    print(
        f"blocks={len(result)} any={sum(result.outlook_cue_count > 0)} "
        f"rich={sum(result.outlook_rich)} "
        f"train_rich={sum((result.split == 'train') & result.outlook_rich)} "
        f"validation_rich={sum((result.split == 'validation') & result.outlook_rich)} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
