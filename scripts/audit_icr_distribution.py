#!/usr/bin/env python3
"""Audit JF-ICR distribution shift, cue reliability, and contradictory neighbors."""

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.parsing import parse_icr_query  # noqa: E402


CUES = {
    "no_plan": r"予定.{0,5}(?:は|が)?ない|計画.{0,5}(?:は|が)?ない|方針.{0,5}(?:は|が)?ない",
    "withhold": r"回答.{0,8}差し控|コメント.{0,8}差し控|非開示|ノーコメント",
    "undecided": r"未定|検討中|決定.{0,5}(?:してい|されてい)?ない|決断.{0,5}していない",
    "cannot_forecast": r"見通.{0,8}(?:困難|できない|せない)|何とも言えない",
    "want_intent": r"していきたい|取り組んでいきたい|目指していく|考えている|方針",
    "firm_action": r"実施する|実行する|決定した|達成する|取り組んでいく|進めていく",
    "ordinary_hedge": r"と考えている|と思っている|見込んでいる|予定している",
    "temporary": r"現時点|今のところ|当面|しばらく|現状では",
    "negative": r"行わない|実施しない|考えていない|目標.{0,5}わけではない|影響はない",
}


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    parsed = frame["query"].map(parse_icr_query)
    frame = frame.assign(
        question=parsed.map(lambda item: item.question),
        response=parsed.map(lambda item: item.response),
        split=np.where(frame["id"] <= 93, "core", "expanded"),
    )
    sources = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv")[["id", "source_group"]]
    frame = frame.merge(sources, on="id", how="left")

    distribution = frame.groupby(["split", "answer"]).agg(
        rows=("id", "size"),
        mean_question_chars=("question", lambda x: x.str.len().mean()),
        mean_response_chars=("response", lambda x: x.str.len().mean()),
        sources=("source_group", "nunique"),
    ).reset_index()

    cue_rows = []
    for name, pattern in CUES.items():
        matched = frame["response"].str.contains(pattern, regex=True, na=False)
        for split, split_mask in (
            ("all", np.ones(len(frame), dtype=bool)),
            ("core", frame["id"] <= 93),
            ("expanded", frame["id"] > 93),
        ):
            selected = frame[matched & split_mask]
            counts = selected["answer"].value_counts()
            cue_rows.append({
                "cue": name, "pattern": pattern, "split": split, "rows": len(selected),
                **{f"label_{label}": int(counts.get(label, 0)) for label in ["-2", "-1", "0", "+1", "+2"]},
                "majority_label": counts.index[0] if len(counts) else "",
                "purity": counts.iloc[0] / len(selected) if len(selected) else 0.0,
            })

    texts = ("質問: " + frame["question"] + " 回答: " + frame["response"]).tolist()
    matrix = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), min_df=1, sublinear_tf=True,
    ).fit_transform(texts)
    similarities = linear_kernel(matrix, matrix)
    np.fill_diagonal(similarities, -1.0)
    neighbor_rows = []
    for index, row in frame.iterrows():
        different = frame["answer"].to_numpy() != row["answer"]
        candidates = np.flatnonzero(different)
        nearest = candidates[np.argmax(similarities[index, candidates])]
        other = frame.iloc[nearest]
        neighbor_rows.append({
            "id": int(row["id"]), "label": row["answer"],
            "neighbor_id": int(other["id"]), "neighbor_label": other["answer"],
            "similarity": float(similarities[index, nearest]),
            "same_source": row["source_group"] == other["source_group"],
            "question": row["question"], "response": row["response"],
            "neighbor_question": other["question"], "neighbor_response": other["response"],
        })

    output = ROOT / "artifacts/icr_audit"
    output.mkdir(parents=True, exist_ok=True)
    distribution.to_csv(output / "distribution_by_split_label.csv", index=False)
    pd.DataFrame(cue_rows).to_csv(output / "cue_label_counts.csv", index=False)
    neighbors = pd.DataFrame(neighbor_rows).sort_values("similarity", ascending=False)
    neighbors.to_csv(output / "nearest_different_label.csv", index=False)
    neighbors.head(40).to_csv(output / "top_potential_contradictions.csv", index=False)
    print("DISTRIBUTION")
    print(distribution.to_string(index=False))
    print("\nCUES")
    print(pd.DataFrame(cue_rows).to_string(index=False))
    print("\nTOP DIFFERENT-LABEL NEIGHBORS")
    print(neighbors[["id", "label", "neighbor_id", "neighbor_label", "similarity", "same_source"]].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
