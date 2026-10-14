#!/usr/bin/env python3
"""Create a compact, auditable JF-ICR boundary corpus for expert review."""

from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.parsing import parse_icr_query  # noqa: E402


CUES = {
    "consider": r"検討|考慮",
    "intend_or_policy": r"考えている|方針|意向|予定",
    "target": r"目指|めざ|目標",
    "undecided": r"未定|決まっていない|決定していない|現時点では.*ない",
    "difficult": r"難しい|困難",
    "cannot_answer": r"回答.*控|コメント.*控|差し控",
    "conditional": r"場合|次第|状況.*見|前提|可能性",
    "explicit_negative": r"ない|ません|できない|否定",
}


def cue_names(text: str) -> str:
    return ",".join(name for name, pattern in CUES.items() if re.search(pattern, text))


def main() -> None:
    source = ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet"
    frame = pd.read_parquet(source)
    parsed = frame["query"].map(parse_icr_query)
    frame = frame.assign(
        question=parsed.map(lambda item: item.question),
        response=parsed.map(lambda item: item.response),
        response_ending=parsed.map(lambda item: item.response[-240:]),
        cues=parsed.map(lambda item: cue_names(item.response)),
    )

    # Include every scarce strong/negative item and a deterministic sample from
    # the two common adjacent classes. The latter is deliberately balanced.
    mandatory = frame[frame["answer"].isin(["-2", "-1", "+2"])]
    boundary = pd.concat(
        [
            frame[frame["answer"] == label].sample(20, random_state=2026)
            for label in ["0", "+1"]
        ]
    )
    review = pd.concat([mandatory, boundary]).drop_duplicates("id").sort_values("id")
    review = review[
        ["id", "answer", "question", "response", "response_ending", "cues"]
    ].rename(columns={"answer": "gold_label"})
    review["reviewed_label"] = ""
    review["decisive_span"] = ""
    review["review_notes"] = ""

    output_dir = ROOT / "data/derived"
    output_dir.mkdir(parents=True, exist_ok=True)
    review.to_csv(output_dir / "icr_boundary_review.csv", index=False)

    rows = []
    for name, pattern in CUES.items():
        present = frame["response"].str.contains(pattern, regex=True)
        counts = frame.loc[present, "answer"].value_counts().to_dict()
        rows.append({"cue": name, "count": int(present.sum()), **counts})
    cue_frame = pd.DataFrame(rows).fillna(0)
    cue_frame.to_csv(output_dir / "icr_cue_counts.csv", index=False)
    print(f"review_rows={len(review)}")
    print(cue_frame.to_string(index=False))


if __name__ == "__main__":
    main()

