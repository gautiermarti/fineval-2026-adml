#!/usr/bin/env python3
"""Retrieve nearest gold example from every label for contrastive prompting."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.parsing import parse_icr_query  # noqa: E402


LABELS = ["-2", "-1", "0", "+1", "+2"]


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    embeddings = np.load(ROOT / "data/derived/icr_multilingual_embeddings.npy")
    similarities = embeddings @ embeddings.T
    parsed = frame["query"].map(parse_icr_query)
    labels = frame["answer"].to_numpy()
    folds = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id")["source_fold"]
    records = []
    for position, row in frame.iterrows():
        exemplars = {}
        oof_exemplars = {}
        for label in LABELS:
            candidates = np.flatnonzero(labels == label)
            candidates = candidates[candidates != position]
            if not len(candidates):
                continue
            nearest = candidates[np.argmax(similarities[position, candidates])]
            item = parsed.iloc[nearest]
            exemplars[label] = {
                "id": int(frame.iloc[nearest]["id"]),
                "similarity": float(similarities[position, nearest]),
                "question": item.question,
                "response": item.response,
            }
            oof_candidates = candidates[
                np.array([folds.loc[int(frame.iloc[index]["id"])] != folds.loc[int(row["id"])] for index in candidates])
            ]
            if len(oof_candidates):
                oof_nearest = oof_candidates[np.argmax(similarities[position, oof_candidates])]
                oof_item = parsed.iloc[oof_nearest]
                oof_exemplars[label] = {
                    "id": int(frame.iloc[oof_nearest]["id"]),
                    "similarity": float(similarities[position, oof_nearest]),
                    "question": oof_item.question,
                    "response": oof_item.response,
                }
        target = parsed.iloc[position]
        records.append(
            {
                "id": int(row["id"]),
                "gold_label": row["answer"],
                "question": target.question,
                "response": target.response,
                "exemplars": exemplars,
                "oof_exemplars": oof_exemplars,
            }
        )
    output = ROOT / "data/derived/icr_contrastive_exemplars.jsonl"
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote={len(records)} path={output}")
    for label in LABELS:
        values = [r["exemplars"][label]["similarity"] for r in records if label in r["exemplars"]]
        print(label, "mean_similarity", round(float(np.mean(values)), 4), "n", len(values))


if __name__ == "__main__":
    main()
