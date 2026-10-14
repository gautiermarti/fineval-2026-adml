#!/usr/bin/env python3
"""Classify explicit synthetic JF-ICR responses with a reproducible rubric."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import re

import pandas as pd


PROMPT = re.compile(
    r"Financial Question:\s*(.*?)\s*Company Response:\s*(.*?)\s*Directly output",
    re.DOTALL,
)

STRONG_COMMITMENT = (
    r"取締役会で承認しました",
    r"取締役会で、取得総額.*決議しました",
    r"計画を確定しています",
    r"最終契約を締結しました",
    r"一斉移行します",
    r"正式に投資決定しました",
    r"全額を.*充当します",
    r"開始日は.*決定しています",
    r"増配する方針を決議しました",
    r"引き上げることを決定しました",
)

STRONG_REFUSAL = (
    r"参入しません",
    r"停止する予定はありません",
    r"人員削減を行う計画はありません",
    r"今期中の値上げは行いません",
    r"完全に終了します",
    r"買収する意思はありません",
    r"利用することはありません",
    r"記念配当は実施しません",
    r"正式に中止しました",
    r"上場させる考えはありません",
)

QUALIFIED_COMMITMENT = (
    r"得られれば.*着工を想定",
    r"継続する場合には.*価格改定",
    r"条件に合う企業があれば前向きに検討",
    r"有力な選択肢の一つです",
    r"想定どおり進めば.*目指します",
    r"全社展開を目指します",
    r"確認できれば.*引き上げを目指したい",
    r"確認できれば.*増設に踏み切りたい",
    r"整えば、具体的な準備に入りたい",
    r"目指すべき水準.*確約はできません",
)

WEAK_REFUSAL = (
    r"直ちに閉鎖するのは難しい",
    r"現時点で一律の大幅値下げは考えていません",
    r"見送る方向です",
    r"当面の優先順位は低い",
    r"大規模な削減に踏み切る考えはありません",
    r"達成は難しい",
    r"具体的な実施予定はありません",
    r"ガイダンスの開示は考えていません",
    r"準備開始は時期尚早",
    r"追加増配は現時点では難しい",
)


def matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def classify(response: str) -> str:
    if matches_any(response, STRONG_REFUSAL):
        return "-2"
    if matches_any(response, STRONG_COMMITMENT):
        return "+2"
    if matches_any(response, WEAK_REFUSAL):
        return "-1"
    if matches_any(response, QUALIFIED_COMMITMENT):
        return "+1"
    return "0"


def is_balanced(counts: Counter[str], rows: int) -> bool:
    labels = ("+2", "+1", "0", "-1", "-2")
    return rows > 0 and rows % len(labels) == 0 and counts == Counter(
        {label: rows // len(labels) for label in labels}
    )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-balanced-50", action="store_true")
    parser.add_argument(
        "--require-balanced-packet", action="store_true",
        help="require equal support for all five labels at any packet size",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    records = []
    for row in frame.itertuples(index=False):
        match = PROMPT.search(str(row.query))
        if match is None:
            raise ValueError(f"could not parse JF-ICR prompt id={row.id}")
        records.append({"id": int(row.id), "answer": classify(match.group(2))})
    counts = Counter(row["answer"] for row in records)
    if args.require_balanced_50 and (
        len(records) != 50 or counts != Counter({label: 10 for label in ("+2", "+1", "0", "-1", "-2")})
    ):
        raise ValueError(f"official balanced-packet invariant failed: {dict(counts)}")
    if args.require_balanced_packet and not is_balanced(counts, len(records)):
        raise ValueError(f"five-way balanced-packet invariant failed: {dict(counts)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    print(json.dumps({
        "rows": len(records),
        "label_counts": dict(sorted(counts.items())),
        "output": str(args.output),
        "current_or_hidden_labels_used": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
