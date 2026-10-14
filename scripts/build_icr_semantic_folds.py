#!/usr/bin/env python3
"""Build semantic-group folds without consuming the competition GPU.

Run this script with an environment containing sentence-transformers. It forces
CPU inference so concurrent Ollama work on the DGX Spark is not disturbed.
The resulting groups are conservative topic/style proxies, not asserted ground-
truth source-document identities.
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import BisectingKMeans
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.parsing import parse_icr_query  # noqa: E402


MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
N_GROUPS = 24


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    parsed = frame["query"].map(parse_icr_query)
    texts = [f"質問: {item.question}\n回答: {item.response}" for item in parsed]
    model = SentenceTransformer(MODEL, device="cpu")
    embeddings = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    clusterer = BisectingKMeans(
        n_clusters=N_GROUPS,
        random_state=2026,
        bisecting_strategy="largest_cluster",
    )
    groups = clusterer.fit_predict(embeddings)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
    folds = np.full(len(frame), -1, dtype=int)
    for fold, (_, test) in enumerate(splitter.split(texts, frame["answer"], groups)):
        folds[test] = fold

    output_dir = ROOT / "data/derived"
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments = pd.DataFrame(
        {
            "id": frame["id"],
            "gold_label": frame["answer"],
            "semantic_group": groups,
            "fold": folds,
            "question": parsed.map(lambda item: item.question),
        }
    )
    assignments.to_csv(output_dir / "icr_semantic_folds.csv", index=False)
    np.save(output_dir / "icr_multilingual_embeddings.npy", embeddings)

    summaries = []
    for group, rows in assignments.groupby("semantic_group"):
        summaries.append(
            {
                "semantic_group": group,
                "size": len(rows),
                "id_min": rows["id"].min(),
                "id_max": rows["id"].max(),
                "labels": rows["gold_label"].value_counts().to_dict(),
                "sample_questions": " || ".join(rows["question"].head(3)),
            }
        )
    pd.DataFrame(summaries).sort_values("size", ascending=False).to_csv(
        output_dir / "icr_semantic_group_summary.csv", index=False
    )
    print("group_sizes", assignments["semantic_group"].value_counts().describe().to_dict())
    print("fold_sizes", assignments["fold"].value_counts().sort_index().to_dict())
    print("fold_label_counts")
    print(pd.crosstab(assignments["fold"], assignments["gold_label"]).to_string())


if __name__ == "__main__":
    main()
