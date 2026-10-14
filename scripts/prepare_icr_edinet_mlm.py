#!/usr/bin/env python3
"""Prepare company-disjoint Japanese EDINET blocks for masked-LM adaptation."""

from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json

import pandas as pd
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "cl-tohoku/bert-base-japanese-v3"

# Conservative substrings covering every named JF-ICR source group. Exclusion
# is based on issuer metadata, not document labels or benchmark answers.
TASK_COMPANY_TERMS = (
    "三菱商事", "三菱ＵＦＪ", "三菱UFJ", "大林組", "三井物産", "三井住友", "大日本印刷",
    "野村", "清水建設", "伊藤忠", "ソフトバンク", "日産", "千代田化工",
    "鹿島建設", "住友重機械", "住友商事", "全日本空輸", "ＡＮＡ", "ANA",
)


def excluded_company(company: str) -> bool:
    return any(term in company for term in TASK_COMPANY_TERMS)


def company_split(company: str) -> str:
    """Stable 95/5 split with all blocks from an issuer in one partition."""
    bucket = int(hashlib.sha256(company.encode("utf-8")).hexdigest()[:8], 16) % 20
    return "validation" if bucket == 0 else "train"


def token_blocks(tokenizer, text: str, block_tokens: int) -> list[list[int]]:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    blocks = []
    for start in range(0, len(token_ids), block_tokens):
        piece = token_ids[start:start + block_tokens]
        # Very short tails mostly contain boilerplate fragments and produce
        # inefficient batches. Keep blocks with at least one quarter payload.
        if len(piece) < block_tokens // 4:
            continue
        blocks.append(tokenizer.build_inputs_with_special_tokens(piece))
    return blocks


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "data/external/edinet2023/2023.jsonl",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/derived/icr_edinet2023_mlm.parquet",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--block-tokens", type=int, default=510)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records = []
    documents = excluded = 0
    seen = set()
    with args.input.open() as handle:
        for line in handle:
            row = json.loads(line)
            company = str(row["company"])
            documents += 1
            if excluded_company(company):
                excluded += 1
                continue
            for block in token_blocks(tokenizer, str(row["text"]), args.block_tokens):
                digest = hashlib.sha256(bytes().join(
                    int(token).to_bytes(4, "little") for token in block
                )).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                records.append({
                    "company": company,
                    "split": company_split(company),
                    "input_ids": block,
                    "token_count": len(block),
                    "sha256": digest,
                })

    frame = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(
        f"documents={documents} excluded_task_companies={excluded} "
        f"companies={frame.company.nunique()} blocks={len(frame)} "
        f"tokens={frame.token_count.sum()} splits={frame.split.value_counts().to_dict()} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
