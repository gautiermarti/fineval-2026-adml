#!/usr/bin/env python3
"""Evaluate cached ICR judges and source-held-out calibration."""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.metrics import accuracy  # noqa: E402

LABELS = ["-2", "-1", "0", "+1", "+2"]
ANOMALOUS_IDS = {100, 129, 176}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.6_35b-a3b")
    parser.add_argument("--track", default="guideline")
    args = parser.parse_args()
    path = ROOT / "artifacts/icr_pilot" / f"{args.model}.jsonl"
    records = pd.DataFrame(json.loads(line) for line in path.read_text().splitlines())
    records = records[records["track"] == args.track].drop_duplicates("id", keep="last")
    records["label"] = records["prediction"].map(lambda value: value["label"])
    print(f"rows={len(records)} raw_accuracy={accuracy(records['gold'].tolist(), records['label'].tolist()):.6f}")
    clean = ~records["id"].isin(ANOMALOUS_IDS)
    print(
        f"clean_rows={clean.sum()} clean_accuracy="
        f"{accuracy(records.loc[clean, 'gold'].tolist(), records.loc[clean, 'label'].tolist()):.6f}"
    )
    print(pd.crosstab(records["gold"], records["label"], dropna=False).reindex(index=LABELS, columns=LABELS, fill_value=0))
    if len(records) < 253:
        print("calibration_skipped=incomplete_full_pass")
        return

    folds = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv")[["id", "source_fold"]]
    data = records.merge(folds, on="id", how="inner")
    for field in ("responsiveness", "direction", "strength", "conditionality"):
        data[field] = data["prediction"].map(lambda value: value.get(field, "missing"))
    data["confidence"] = data["prediction"].map(lambda value: float(value.get("confidence", 0.5)))
    categorical = ["label", "responsiveness", "direction", "strength", "conditionality"]
    numeric = ["confidence"]
    predictions = pd.Series(index=data.index, dtype=object)
    for fold in sorted(data["source_fold"].unique()):
        train = data["source_fold"] != fold
        test = ~train
        transform = ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", StandardScaler(), numeric),
        ])
        model = make_pipeline(
            transform,
            LogisticRegression(C=0.25, max_iter=2_000, random_state=2026),
        )
        model.fit(data.loc[train, categorical + numeric], data.loc[train, "gold"])
        predictions.loc[test] = model.predict(data.loc[test, categorical + numeric])
    print(f"source_oof_calibrated_accuracy={accuracy(data['gold'].tolist(), predictions.tolist()):.6f}")
    clean = ~data["id"].isin(ANOMALOUS_IDS)
    print(
        "source_oof_calibrated_clean_accuracy="
        f"{accuracy(data.loc[clean, 'gold'].tolist(), predictions.loc[clean].tolist()):.6f}"
    )


if __name__ == "__main__":
    main()
