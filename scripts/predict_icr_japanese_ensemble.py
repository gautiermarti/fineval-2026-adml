#!/usr/bin/env python3
"""Predict JF-ICR with the frozen Japanese two-seed ensemble."""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys
from numbers import Integral

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.parsing import parse_icr_query  # noqa: E402
from predict_icr import rule_prediction  # noqa: E402
from train_icr_japanese_cv import LABELS, encode_pair  # noqa: E402


def decode_probabilities(probabilities: np.ndarray, ordinal_mean: bool = False) -> np.ndarray:
    """Decode categorical probabilities, optionally respecting label ordinality."""
    if not ordinal_mean:
        indices = probabilities.argmax(axis=1)
    else:
        values = np.arange(-2, 3, dtype=np.float64)
        expected = probabilities @ values
        indices = np.clip(np.rint(expected).astype(int) + 2, 0, 4)
    return np.asarray(LABELS)[indices]


class QueryDataset(Dataset):
    def __init__(
        self, queries: list[str], tokenizer, max_length: int,
        truncation_profile: str = "longest_first",
        prepend_cls: bool = False,
    ):
        self.features = []
        for query in queries:
            parsed = parse_icr_query(query)
            self.features.append(encode_pair(
                tokenizer, parsed.question, parsed.response, max_length,
                truncation_profile, prepend_cls,
            ))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict:
        return self.features[index]


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.DataFrame(
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    )


def load_prediction_frame(path: Path) -> pd.DataFrame:
    """Load only input fields, never a colocated diagnostic answer column."""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        names = set(pq.read_schema(path).names)
        columns = ["query"] + (["id"] if "id" in names else [])
        return pd.read_parquet(path, columns=columns)
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows.append({
            key: record[key] for key in ("query", "id") if key in record
        })
    return pd.DataFrame(rows)


def record_id(row: pd.Series, fallback: int):
    value = row.get("id", fallback)
    if pd.isna(value):
        return fallback
    return int(value) if isinstance(value, Integral) else str(value)


def uses_rinna_roberta_compat(model_path: str | Path, force: bool = False) -> bool:
    """Read the explicit compatibility marker saved by the full-data trainer."""
    if force or str(model_path).rstrip("/") == "rinna/japanese-roberta-base":
        return True
    marker = Path(model_path) / "training.txt"
    return marker.exists() and "rinna_roberta_compat=True" in marker.read_text()


def collate_queries(
    items: list[dict], tokenizer, *, zero_based_position_ids: bool = False,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer.pad(items, return_tensors="pt")
    if zero_based_position_ids:
        encoded["position_ids"] = torch.arange(
            encoded["input_ids"].shape[1], dtype=torch.long,
        ).unsqueeze(0).expand_as(encoded["input_ids"])
    return encoded


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--models", type=Path, nargs="+",
        default=[
            ROOT / "artifacts/task2_japanese_full/seed7",
            ROOT / "artifacts/task2_japanese_full/seed19",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--truncation-profile",
        choices=("longest_first", "response_head_tail"),
        default="longest_first",
    )
    parser.add_argument("--no-rule-overrides", action="store_true")
    parser.add_argument(
        "--rinna-roberta-compat", action="store_true",
        help="Force Rinna tokenizer/position compatibility for all supplied models.",
    )
    parser.add_argument(
        "--ordinal-mean", action="store_true",
        help="Decode by rounding the expected ordinal score instead of argmax.",
    )
    parser.add_argument(
        "--lexical-weight", type=float, default=0.0,
        help="Blend a full-public-data TF-IDF model at this fixed probability weight.",
    )
    parser.add_argument(
        "--lexical-training", type=Path,
        default=ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet",
    )
    parser.add_argument("--minimal", action="store_true")
    args = parser.parse_args()

    frame = load_prediction_frame(args.input)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    probability_sum = np.zeros((len(frame), len(LABELS)), dtype=np.float64)
    for model_path in args.models:
        rinna_compat = uses_rinna_roberta_compat(
            model_path, force=args.rinna_roberta_compat,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False if rinna_compat else None,
        )
        if rinna_compat:
            tokenizer.do_lower_case = True
        dataset = QueryDataset(
            frame["query"].tolist(), tokenizer, args.max_length,
            args.truncation_profile, rinna_compat,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=lambda items: collate_queries(
                items, tokenizer, zero_based_position_ids=rinna_compat,
            ),
        )
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        model.eval()
        probabilities = []
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                probabilities.append(model(**batch).logits.softmax(dim=1).cpu().numpy())
        probability_sum += np.concatenate(probabilities)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    mean_probabilities = probability_sum / len(args.models)
    if not 0.0 <= args.lexical_weight <= 1.0:
        raise ValueError("--lexical-weight must be between zero and one")
    if args.lexical_weight:
        # Imported lazily to keep ordinary neural inference independent of
        # scikit-learn and to avoid a module-level circular import.
        from evaluate_icr_nested_lexical_blend import lexical_estimator

        lexical_frame = load_frame(args.lexical_training)
        lexical_parsed = lexical_frame["query"].map(parse_icr_query)
        lexical_texts = np.asarray([
            f"質問: {item.question}\n回答: {item.response}" for item in lexical_parsed
        ])
        input_parsed = frame["query"].map(parse_icr_query)
        input_texts = np.asarray([
            f"質問: {item.question}\n回答: {item.response}" for item in input_parsed
        ])
        fitted = lexical_estimator().fit(
            lexical_texts, lexical_frame["answer"].astype(str).to_numpy(),
        )
        lexical_probabilities = np.zeros_like(mean_probabilities)
        values = fitted.predict_proba(input_texts)
        for column, label in enumerate(fitted.classes_):
            lexical_probabilities[:, LABELS.index(str(label))] = values[:, column]
        mean_probabilities = (
            (1.0 - args.lexical_weight) * mean_probabilities
            + args.lexical_weight * lexical_probabilities
        )
    predictions = decode_probabilities(
        mean_probabilities, ordinal_mean=args.ordinal_mean,
    )
    if not args.no_rule_overrides:
        rules = np.asarray([
            rule_prediction(query, "safe_rules") for query in frame["query"]
        ])
        predictions = np.where(rules != "+1", rules, predictions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for position, ((_, row), answer) in enumerate(zip(frame.iterrows(), predictions)):
            record = {
                "row_index": position,
                "id": record_id(row, position),
                "answer": str(answer),
            }
            if args.minimal:
                record = {"id": record["id"], "answer": record["answer"]}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(frame)} predictions to {args.output}")


if __name__ == "__main__":
    main()
