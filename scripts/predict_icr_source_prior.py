#!/usr/bin/env python3
"""Aggressive JF-ICR predictor for hidden rows resembling known public sources."""

from argparse import ArgumentParser
from pathlib import Path
import json
import re
import sys
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.parsing import parse_icr_query  # noqa: E402


def texts(frame: pd.DataFrame) -> list[str]:
    parsed = frame["query"].map(parse_icr_query)
    return [f"{item.question} {item.response}" for item in parsed]


def load_input(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.DataFrame(
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    )


def record_id(row: pd.Series, fallback: int):
    value = row.get("id", fallback)
    if pd.isna(value):
        return fallback
    return int(value) if isinstance(value, Integral) else str(value)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument(
        "--rule-overrides", action="store_true",
        help="apply the audited explicit-target and short-neutral rules last",
    )
    parser.add_argument("--minimal", action="store_true", help="emit only id and answer")
    parser.add_argument(
        "--diagnostics-output", type=Path,
        help="optional CSV with inferred source and similarity diagnostics",
    )
    args = parser.parse_args()
    training = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    source = pd.read_csv(
        ROOT / "data/derived/icr_source_folds.csv", dtype={"gold_label": str},
    ).set_index("id").loc[training.id]
    test = load_input(args.input)
    train_texts, test_texts = texts(training), texts(test)
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 5), min_df=2,
        max_features=150_000, sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    test_matrix = vectorizer.transform(test_texts)
    similarities = (test_matrix @ train_matrix.T).toarray()
    same_public = args.input.resolve() == (
        ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet"
    ).resolve()
    if same_public:
        np.fill_diagonal(similarities, -2.0)
    groups = source.source_group.to_numpy()
    labels = source.gold_label.to_numpy(dtype=str)
    predictions, diagnostics = [], []
    k = min(args.neighbors, len(training) - int(same_public))
    nearest = np.argpartition(-similarities, k - 1, axis=1)[:, :k]
    for position, neighbors in enumerate(nearest):
        neighbors = neighbors[np.argsort(-similarities[position, neighbors])]
        counts = pd.Series(groups[neighbors]).value_counts()
        inferred_source = counts.index[0]
        source_agreement = int(counts.iloc[0])
        pool = groups == inferred_source
        if same_public:
            pool[position] = False
        prior = pd.Series(labels[pool]).value_counts() if pool.any() else pd.Series(dtype=int)
        prediction = str(prior.index[0]) if source_agreement >= 2 and not prior.empty else "+1"
        if args.rule_overrides:
            parsed = parse_icr_query(test.iloc[position]["query"])
            short_neutral = (
                len(parsed.response) < 80
                and not re.search(
                    r"差し控|非開示|ノーコメント|お答えでき|申し上げられ", parsed.response,
                )
                and not re.search(
                    r"実施|増配|自社株買い|投資|削減|還元|上場|MBO", parsed.question,
                )
            )
            explicit_target = re.search(
                r"目指します|必ず.{0,8}(?:達成|やり切)|決議.{0,8}いたしました|維持する方針",
                parsed.response,
            )
            if short_neutral:
                prediction = "0"
            if explicit_target:
                prediction = "+2"
        predictions.append(prediction)
        diagnostics.append({
            "inferred_source": inferred_source,
            "source_agreement": source_agreement,
            "nearest_similarity": float(similarities[position, neighbors[0]]),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for position, ((_, row), prediction, diagnostic) in enumerate(
            zip(test.iterrows(), predictions, diagnostics)
        ):
            record = {
                "row_index": position,
                "id": record_id(row, position),
                "answer": prediction,
                **diagnostic,
            }
            if args.minimal:
                record = {"id": record["id"], "answer": prediction}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(test)} predictions to {args.output}")
    if args.diagnostics_output:
        args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "row_index": range(len(test)),
            "id": [
                record_id(row, position)
                for position, (_, row) in enumerate(test.iterrows())
            ],
            "prediction": predictions,
            **{key: [item[key] for item in diagnostics] for key in diagnostics[0]},
        }).to_csv(args.diagnostics_output, index=False)
        print(f"wrote diagnostics to {args.diagnostics_output}")
        diagnostic_frame = pd.DataFrame(diagnostics)
        print(
            "source_diagnostics "
            f"agreement_ge_2={(diagnostic_frame.source_agreement >= 2).mean():.3f} "
            f"similarity_median={diagnostic_frame.nearest_similarity.median():.3f} "
            f"similarity_p10={diagnostic_frame.nearest_similarity.quantile(0.10):.3f}"
        )
    if "answer" in test:
        gold = test.answer.astype(str).to_numpy()
        print(f"diagnostic_accuracy={np.mean(np.asarray(predictions) == gold):.6f}")


if __name__ == "__main__":
    main()
