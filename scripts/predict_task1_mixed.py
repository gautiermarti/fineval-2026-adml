#!/usr/bin/env python3
"""Prompt-aware robust inference for mixed GRFinNUM and GRFinNER inputs."""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from predict_task1_ensemble import DEFAULT_MODELS, FINAL_REFIT_MODELS  # noqa: E402
from predict_task1_lookup_safe import task_kind  # noqa: E402
from predict_task1_textual_multilabel import predict_frame as predict_multilabel  # noqa: E402
from task1_postprocess import (  # noqa: E402
    add_explicit_identifiers, add_high_support_nested_locations,
)
from train_task1_transformer import decode as decode_numeric  # noqa: E402
from train_task1_transformer import make_examples as make_numeric_examples  # noqa: E402
from train_task1_textual import decode as decode_textual  # noqa: E402
from train_task1_textual import encode_release_entity  # noqa: E402
from train_task1_textual import make_examples as make_textual_examples  # noqa: E402


DEFAULT_TEXTUAL_MODELS = [
    ROOT / "artifacts/task1_textual/legal_base_seed7",
    ROOT / "artifacts/task1_textual/legal_ner_init_seed7",
    ROOT / "artifacts/task1_textual/finance_adapted_seed7",
]


def add_nested_locations(flat_entities, multilabel_entities):
    """Add independently predicted locations that are nested in a flat ORG span."""
    output = list(flat_entities)
    organizations = [entity for entity, kind in flat_entities if kind == "ΟΡΓΑΝΙΣΜΟΣ"]
    for entity, kind in multilabel_entities:
        candidate = (entity, kind)
        if (
            kind == "ΤΟΠΟΘΕΣΙΑ"
            and any(entity != organization and entity in organization for organization in organizations)
            and candidate not in output
        ):
            output.append(candidate)
    return output


def model_logits(frame: pd.DataFrame, paths: list[Path], make_examples, device):
    all_logits = []
    reference_examples = None
    reference_tokenizer = None
    for path in paths:
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        examples = make_examples(frame.assign(answer=""), tokenizer)
        # Textual preparation also reports unaligned rows.
        if isinstance(examples, tuple):
            examples = examples[0]
        model = AutoModelForTokenClassification.from_pretrained(path).to(device).eval()
        rows = []
        with torch.no_grad():
            for encoded, *_ in examples:
                input_ids = torch.tensor(encoded["input_ids"])[None].to(device)
                attention = torch.tensor(encoded["attention_mask"])[None].to(device)
                rows.append(model(input_ids=input_ids, attention_mask=attention).logits[0].cpu())
        if reference_examples is not None:
            offsets = [row[0].offset_mapping for row in examples]
            old_offsets = [row[0].offset_mapping for row in reference_examples]
            if offsets != old_offsets:
                raise ValueError("ensemble tokenizers produce incompatible offsets")
        else:
            reference_examples, reference_tokenizer = examples, tokenizer
        all_logits.append(rows)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    averaged = [
        sum((model[index] for model in all_logits), torch.zeros_like(all_logits[0][index]))
        / len(all_logits)
        for index in range(len(frame))
    ]
    return averaged, reference_examples, reference_tokenizer


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numeric-models", type=Path, nargs="+")
    parser.add_argument("--numeric-final-refit", action="store_true")
    parser.add_argument("--textual-models", type=Path, nargs="+")
    parser.add_argument("--nested-multilabel", action="store_true")
    parser.add_argument(
        "--multilabel-model-dir", type=Path,
        default=ROOT / "artifacts/task1_textual/multilabel_legal_seed7",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    kinds = [task_kind(query) for query in frame["query"]]
    unknown = [index for index, kind in enumerate(kinds) if kind == "unknown"]
    if unknown:
        raise SystemExit(f"cannot identify Task 1 track for rows {unknown[:20]}")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    output: list[list[tuple[str, str]] | None] = [None] * len(frame)

    numeric_indices = [index for index, kind in enumerate(kinds) if kind == "numeric"]
    if numeric_indices:
        numeric_frame = frame.iloc[numeric_indices].reset_index(drop=True)
        numeric_paths = args.numeric_models or [
            ROOT / path for path in (
                FINAL_REFIT_MODELS if args.numeric_final_refit else DEFAULT_MODELS
            )
        ]
        logits, examples, tokenizer = model_logits(
            numeric_frame, numeric_paths, make_numeric_examples, device,
        )
        predictions = decode_numeric(logits, examples, tokenizer)
        predictions = [
            add_explicit_identifiers(row.text, entities)
            for row, entities in zip(numeric_frame.itertuples(), predictions)
        ]
        for index, entities in zip(numeric_indices, predictions):
            output[index] = entities

    textual_indices = [index for index, kind in enumerate(kinds) if kind == "textual"]
    if textual_indices:
        textual_frame = frame.iloc[textual_indices].reset_index(drop=True)
        textual_paths = args.textual_models or DEFAULT_TEXTUAL_MODELS
        logits, examples, _ = model_logits(
            textual_frame, textual_paths, make_textual_examples, device,
        )
        predictions = decode_textual(logits, examples)
        if args.nested_multilabel:
            overlapping = predict_multilabel(
                textual_frame, args.multilabel_model_dir, device,
            )
            predictions = [
                add_nested_locations(flat, multi)
                for flat, multi in zip(predictions, overlapping)
            ]
        predictions = [add_high_support_nested_locations(row) for row in predictions]
        predictions = [
            [(encode_release_entity(entity, kind), kind) for entity, kind in entities]
            for entities in predictions
        ]
        for index, entities in zip(textual_indices, predictions):
            output[index] = entities

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for index, entities in enumerate(output):
            if entities is None:
                raise RuntimeError(f"row {index} was not routed")
            answer = "\n".join(f"{entity}, {kind}" for entity, kind in entities)
            handle.write(json.dumps({"row_index": index, "answer": answer}, ensure_ascii=False) + "\n")
    counts = pd.Series(kinds).value_counts().to_dict()
    print(f"wrote={len(output)} routes={counts} device={device} output={args.output}")


if __name__ == "__main__":
    main()
