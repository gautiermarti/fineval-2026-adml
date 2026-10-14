#!/usr/bin/env python3
"""Fine-tune a nested multilingual GLiNER proposer for GRFinNER.

The model is trained on the official 320-row training split only.  Validation
is used for reporting and threshold selection; the public test is never read.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import random
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from task1_numeric_baseline import parse_answer, score  # noqa: E402
from train_task1_textual import encode_release_entity  # noqa: E402
from train_task1_textual_multilabel import align_overlapping  # noqa: E402


GREEK_TO_LABEL = {
    "ΠΡΟΣΩΠΟ": "person",
    "ΟΡΓΑΝΙΣΜΟΣ": "organization",
    "ΤΟΠΟΘΕΣΙΑ": "location",
}
LABEL_TO_GREEK = {value: key for key, value in GREEK_TO_LABEL.items()}


def train_model_compat(model, train_dataset, training_args):
    """Train across the Transformers 4/5 ``Trainer`` API transition.

    GLiNER 0.2.24's convenience method still supplies ``tokenizer=``.  That
    keyword was renamed to ``processing_class=`` in Transformers 5.  Calling
    the compatible custom trainer directly avoids pinning the whole research
    environment to an old Transformers release.
    """
    from gliner.training import Trainer

    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": None,
        "data_collator": model._create_data_collator(),
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    tokenizer = model.data_processor.transformer_tokenizer
    if "processing_class" in trainer_parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**kwargs)
    # GLiNER 0.2.24 checks this legacy Trainer attribute unconditionally in
    # ``training_step``.  Transformers 5 removed it when Apex support moved;
    # without the compatibility value GLiNER silently catches the resulting
    # AttributeError and skips *every* optimizer step with zero loss.
    if not hasattr(trainer, "use_apex"):
        trainer.use_apex = False
    trainer.train()
    return trainer


def configure_max_width(model, max_width: int) -> None:
    """Resize this checkpoint's width-agnostic span enumeration safely.

    ``gliner_multi-v2.1`` uses ``SpanMarkerV0``: the marker MLP has no
    width-sized parameters, but stores ``max_width`` for its final reshape.
    GLiNER updates the processor/config without updating that nested attribute,
    producing an invalid tensor.  Keep all three values synchronized and fail
    closed for a representation family whose parameters may depend on width.
    """
    marker = model.model.span_rep_layer.span_rep_layer
    if marker.__class__.__name__ not in {"SpanMarkerV0", "SpanMarkerV1"}:
        raise ValueError(
            f"cannot safely resize width for {marker.__class__.__name__}; "
            "only width-agnostic SpanMarkerV0/V1 are supported"
        )
    model.config.max_width = max_width
    model.data_processor.config.max_width = max_width
    marker.max_width = max_width


def make_gliner_dataset(frame: pd.DataFrame, words_splitter, max_width: int):
    """Convert released answer strings to GLiNER inclusive token spans."""
    dataset = []
    audit = {
        "rows": len(frame), "gold_entities": 0, "aligned_entities": 0,
        "unaligned_entities": [], "over_width_entities": [],
    }
    for row_index, row in frame.iterrows():
        text = str(row.text)
        gold = parse_answer(row.answer)
        annotations = align_overlapping(text, gold)
        split = list(words_splitter(text))
        tokens = [token for token, _, _ in split]
        start_to_index = {start: index for index, (_, start, _) in enumerate(split)}
        end_to_index = {end: index for index, (_, _, end) in enumerate(split)}
        ner = set()
        audit["gold_entities"] += len(set(gold))
        for start, end, kind in annotations:
            if start not in start_to_index or end not in end_to_index:
                audit["unaligned_entities"].append({
                    "row_index": int(row_index), "entity": text[start:end], "type": kind,
                    "start": start, "end": end,
                })
                continue
            first, last = start_to_index[start], end_to_index[end]
            if last - first + 1 > max_width:
                audit["over_width_entities"].append({
                    "row_index": int(row_index), "entity": text[start:end], "type": kind,
                    "width": last - first + 1,
                })
                continue
            ner.add((first, last, GREEK_TO_LABEL[kind]))
        audit["aligned_entities"] += len(ner)
        dataset.append({"tokenized_text": tokens, "ner": sorted(ner)})
    return dataset, audit


def evaluate(model, frame: pd.DataFrame, *, batch_size: int):
    raw = model.batch_predict_entities(
        frame["text"].astype(str).tolist(), list(LABEL_TO_GREEK),
        flat_ner=False, multi_label=False, threshold=0.05, batch_size=batch_size,
    )
    gold = [set(parse_answer(answer)) for answer in frame["answer"]]
    summaries = []
    all_predictions = {}
    for threshold in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        predictions = []
        for entities in raw:
            row = set()
            for entity in entities:
                if float(entity["score"]) < threshold:
                    continue
                kind = LABEL_TO_GREEK[str(entity["label"])]
                surface = encode_release_entity(str(entity["text"]), kind)
                row.add((surface, kind))
            predictions.append(row)
        macro_f1 = sum(
            score(list(target), list(prediction))[2]
            for target, prediction in zip(gold, predictions)
        ) / len(gold)
        hits = sum(len(target & prediction) for target, prediction in zip(gold, predictions))
        summaries.append({
            "threshold": threshold,
            "macro_entity_f1": macro_f1,
            "typed_proposal_hits": hits,
            "gold_entities": sum(len(row) for row in gold),
            "predicted_entities": sum(len(row) for row in predictions),
        })
        all_predictions[threshold] = predictions
    best = max(summaries, key=lambda item: item["macro_entity_f1"])
    return raw, summaries, best, all_predictions[best["threshold"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="urchade/gliner_multi-v2.1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-width", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task1_gliner_finetuned_seed7",
    )
    args = parser.parse_args()

    from gliner import GLiNER

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = pd.read_parquet(
        ROOT / "data/external/plutus_finner_text/train-00000-of-00001.parquet"
    )
    validation = pd.read_parquet(
        ROOT / "data/external/plutus_finner_text/validation-00000-of-00001.parquet"
    )
    model = GLiNER.from_pretrained(args.model, map_location="cpu")
    configure_max_width(model, args.max_width)
    train_dataset, train_audit = make_gliner_dataset(
        train, model.data_processor.words_splitter, args.max_width,
    )
    validation_dataset, validation_audit = make_gliner_dataset(
        validation, model.data_processor.words_splitter, args.max_width,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "alignment_audit.json").write_text(json.dumps({
        "train": train_audit, "validation": validation_audit,
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"train": train_audit, "validation": validation_audit},
                     ensure_ascii=False, indent=2), flush=True)

    if args.smoke_only:
        collator = model._create_data_collator()
        batch = collator(train_dataset[:2])
        with torch.no_grad():
            output = model(**batch)
        print(f"smoke_loss={float(output.loss):.8f}")
        return

    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --cpu only for an intentional CPU run")
    training_args = model.create_training_args(
        output_dir=args.output_dir / "checkpoints",
        learning_rate=args.encoder_learning_rate,
        others_lr=args.head_learning_rate,
        weight_decay=0.01,
        others_weight_decay=0.01,
        warmup_ratio=0.10,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        save_steps=40,
        save_total_limit=3,
        logging_steps=10,
        use_cpu=args.cpu,
        bf16=not args.cpu,
        dataloader_num_workers=1,
        report_to="none",
        eval_strategy="no",
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    train_model_compat(model, train_dataset, training_args)
    final_dir = args.output_dir / "final"
    model.save_pretrained(final_dir)

    model.to("cpu")
    raw, summaries, best, predictions = evaluate(
        model, validation, batch_size=args.batch_size,
    )
    (args.output_dir / "validation_raw.jsonl").write_text("".join(
        json.dumps({"row_index": index, "entities": entities}, ensure_ascii=False) + "\n"
        for index, entities in enumerate(raw)
    ))
    (args.output_dir / "validation_predictions.jsonl").write_text("".join(
        json.dumps({
            "row_index": index,
            "answer": "\n".join(f"{entity}, {kind}" for entity, kind in sorted(row)),
        }, ensure_ascii=False) + "\n"
        for index, row in enumerate(predictions)
    ))
    metrics = {
        "base_model": args.model,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "max_width": args.max_width,
        "encoder_learning_rate": args.encoder_learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "best_validation": best,
        "threshold_sweep": summaries,
        "public_test_used_for_selection": False,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
