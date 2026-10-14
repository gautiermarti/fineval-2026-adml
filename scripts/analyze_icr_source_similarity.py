#!/usr/bin/env python3
"""Calibrate diagnostics that distinguish continued from novel JF-ICR sources."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.parsing import parse_icr_query  # noqa: E402


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    parsed = frame["query"].map(parse_icr_query)
    texts = [f"{x.question} {x.response}" for x in parsed]
    groups = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv").set_index("id").loc[
        frame.id, "source_group"
    ].to_numpy()
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 5), min_df=2, max_features=150_000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    similarity = (matrix @ matrix.T).toarray()
    rows = []
    for position in range(len(frame)):
        for scenario in ("continued_source", "novel_source"):
            allowed = np.ones(len(frame), dtype=bool)
            allowed[position] = False
            if scenario == "novel_source":
                allowed &= groups != groups[position]
            indices = np.flatnonzero(allowed)
            nearest = indices[np.argsort(-similarity[position, indices])[:3]]
            counts = pd.Series(groups[nearest]).value_counts()
            rows.append({
                "row_index": position, "id": frame.iloc[position].id,
                "scenario": scenario,
                "nearest_similarity": float(similarity[position, nearest[0]]),
                "source_agreement": int(counts.iloc[0]),
            })
    result = pd.DataFrame(rows)
    out = ROOT / "artifacts/icr_source_prior/source_similarity_calibration.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    wide = result.pivot(index="row_index", columns="scenario", values="nearest_similarity")
    labels = np.r_[np.ones(len(wide)), np.zeros(len(wide))]
    scores = np.r_[wide.continued_source, wide.novel_source]
    print(result.groupby("scenario").agg(
        median=("nearest_similarity", "median"),
        p10=("nearest_similarity", lambda x: x.quantile(.1)),
        p90=("nearest_similarity", lambda x: x.quantile(.9)),
        agreement_ge_2=("source_agreement", lambda x: (x >= 2).mean()),
    ).to_string())
    print(f"similarity_auc={roc_auc_score(labels, scores):.6f} wrote={out}")


if __name__ == "__main__":
    main()
