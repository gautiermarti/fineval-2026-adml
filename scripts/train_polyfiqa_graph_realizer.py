#!/usr/bin/env python3
"""LoRA-tune a constrained fact-plan -> delexicalized reference realizer.

This is a *realizer-only* experiment: historical evaluation plans contain
reference-oracle evidence/copy selections.  It cannot be reported as an
end-to-end competition system until an input-only planner reproduces those
slots.  Company-held-out splits and strict marker restoration are mandatory.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import re
import sys

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.metrics import rouge1_f1  # noqa: E402
from fineval.polyfiqa_delex import restore_reference  # noqa: E402
from fineval.polyfiqa_facts import build_fact_graph  # noqa: E402
from fineval.polyfiqa_programs import COMPANY_NAMES  # noqa: E402


SYSTEM = """You realize grounded financial answers from a typed JSON plan.
Return only one delexicalized reference template. Use every requested __VALUE_n__
marker exactly once unless the demonstrated template requires a repeated fact.
Never invent a number, company, year, metric, or unsupported claim. Preserve
__COMPANY__, __COMPANY_POSSESSIVE__, and __YEAR_*__ markers when applicable."""


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def user_prompt(plan: dict) -> str:
    return "PLAN_JSON:\n" + json.dumps(
        plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ) + "\n\nDELEXICALIZED_REFERENCE_TEMPLATE:"


def render_prompt(tokenizer, plan: dict) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_prompt(plan)},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            # Qwen-family templates otherwise enable a long hidden-reasoning
            # prefix.  The target is a short deterministic realization, and
            # training/evaluation must use the same non-thinking surface.
            enable_thinking=False,
        )
    return SYSTEM + "\n\n" + user_prompt(plan) + "\n"


class TemplateDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length: int):
        self.examples = []
        for row in rows:
            prompt = render_prompt(tokenizer, row["plan"])
            target = str(row["target_template"]).strip() + (tokenizer.eos_token or "")
            prompt_ids = tokenizer(
                prompt, add_special_tokens=False, truncation=True,
                max_length=max_length,
            )["input_ids"]
            remaining = max_length - len(prompt_ids)
            if remaining <= 0:
                continue
            target_ids = tokenizer(
                target, add_special_tokens=False, truncation=True,
                max_length=remaining,
            )["input_ids"]
            if not target_ids or len(prompt_ids) + len(target_ids) > max_length:
                continue
            input_ids = prompt_ids + target_ids
            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": [-100] * len(prompt_ids) + target_ids,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class Collator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        length = max(len(row["input_ids"]) for row in rows)
        return {
            "input_ids": torch.tensor([
                row["input_ids"] + [self.pad_token_id] * (length - len(row["input_ids"]))
                for row in rows
            ]),
            "attention_mask": torch.tensor([
                row["attention_mask"] + [0] * (length - len(row["attention_mask"]))
                for row in rows
            ]),
            "labels": torch.tensor([
                row["labels"] + [-100] * (length - len(row["labels"]))
                for row in rows
            ]),
        }


def marker_multiset(text: str):
    return sorted(re.findall(r"__(?:VALUE_\d+|COMPANY(?:_POSSESSIVE)?|YEAR_[A-Z_]+)__", text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-revision", required=True,
        help="Immutable Hugging Face commit used for the base model/tokenizer",
    )
    parser.add_argument(
        "--corpus-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_graph_to_reference_v5",
    )
    parser.add_argument("--held-company", required=True, choices=["HON", "JNJ", "MSFT", "UVV"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--maximum-new-tokens", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="Validate the immutable split and token budget without loading the model",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = (
            ROOT / "artifacts/polyfiqa_graph_realizer" / args.held_company / "seed7"
        )
    from transformers import AutoTokenizer

    split_dir = args.corpus_dir / "loco" / args.held_company
    train_rows = load_jsonl(split_dir / "train.jsonl")
    evaluation_rows = load_jsonl(split_dir / "evaluation.jsonl")
    held_leaks = [
        row for row in train_rows
        if row.get("company") == args.held_company
        or row.get("donor_company") == args.held_company
        or row.get("target_company") == args.held_company
    ]
    if held_leaks:
        raise RuntimeError(
            f"{len(held_leaks)} training rows involve held company {args.held_company}"
        )
    if any(row.get("company") != args.held_company for row in evaluation_rows):
        raise RuntimeError("evaluation split contains a non-held-company row")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.model_revision, use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = TemplateDataset(train_rows, tokenizer, args.max_length)
    if len(dataset) < int(0.8 * len(train_rows)):
        raise RuntimeError(
            f"only {len(dataset)}/{len(train_rows)} rows fit max_length={args.max_length}"
        )
    if args.preflight_only:
        lengths = [len(row["input_ids"]) for row in dataset.examples]
        result = {
            "model": args.model,
            "model_revision": args.model_revision,
            "corpus_dir": str(args.corpus_dir.resolve()),
            "held_company": args.held_company,
            "training_rows_available": len(train_rows),
            "training_rows_fitting_context": len(dataset),
            "evaluation_rows": len(evaluation_rows),
            "minimum_training_tokens": min(lengths),
            "maximum_training_tokens": max(lengths),
            "held_company_training_leaks": 0,
            "preflight_only": True,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if not torch.cuda.is_available():
        raise RuntimeError("This LoRA experiment requires an intentional GPU window")
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("Install the pinned PEFT dependency before the GPU run") from exc
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.model_revision, dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    lora = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank * 2, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.10,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer_kwargs = {
        "model": model, "args": training_args, "train_dataset": dataset,
        "data_collator": Collator(tokenizer.pad_token_id),
    }
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    model.save_pretrained(args.output_dir / "adapter")
    tokenizer.save_pretrained(args.output_dir / "adapter")

    model.config.use_cache = True
    model.eval()
    predictions = []
    valid_restorations = 0
    template_scores, restored_scores = [], []
    frame_cache = {}
    with torch.no_grad():
        for record in evaluation_rows:
            prompt = render_prompt(tokenizer, record["plan"])
            encoded = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=args.max_length,
            ).to(model.device)
            output = model.generate(
                **encoded, do_sample=False, max_new_tokens=args.maximum_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            generated = tokenizer.decode(
                output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True,
            ).strip()
            expected_template = str(record["target_template"]).strip()
            markers_valid = marker_multiset(generated) == marker_multiset(expected_template)
            restored = None
            if markers_valid:
                source = record["source"]
                if source not in frame_cache:
                    frame_cache[source] = __import__("pandas").read_parquet(source)
                row = frame_cache[source].loc[int(record["source_index"])]
                graph = build_fact_graph(row.task_id, row.query, row.question)
                try:
                    restored = restore_reference(
                        {"template": generated, "slots": record["slots"],
                         "company_possessive_surface": COMPANY_NAMES[record["company"]] + "'s"},
                        graph, row.task_id,
                    )
                except (KeyError, ValueError):
                    restored = None
            template_score = rouge1_f1([expected_template], [generated])
            restored_score = (
                rouge1_f1([record["reference_core"]], [restored]) if restored else 0.0
            )
            valid_restorations += int(restored is not None)
            template_scores.append(template_score)
            restored_scores.append(restored_score)
            predictions.append({
                "task_id": record["plan"]["task_id"],
                "held_company": args.held_company,
                "source": record["source"],
                "source_index": int(record["source_index"]),
                "tier": record["plan"]["tier"],
                "family": record["plan"]["family"],
                "expected_template": expected_template,
                "reference_core": record["reference_core"],
                "generated_template": generated,
                "markers_valid": markers_valid, "restored": restored,
                "template_rouge1": template_score, "restored_rouge1": restored_score,
            })
    (args.output_dir / "evaluation_predictions.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in predictions
    ))
    summary = {
        "model": args.model,
        "model_revision": args.model_revision,
        "corpus_dir": str(args.corpus_dir.resolve()),
        "held_company": args.held_company,
        "training_rows_available": len(train_rows),
        "training_rows_fitting_context": len(dataset),
        "evaluation_rows": len(evaluation_rows),
        "valid_marker_restorations": valid_restorations,
        "template_rouge1": sum(template_scores) / len(template_scores),
        "restored_rouge1": sum(restored_scores) / len(restored_scores),
        "by_family": {
            family: {
                "rows": sum(row["family"] == family for row in predictions),
                "valid_marker_restorations": sum(
                    row["family"] == family and row["restored"] is not None
                    for row in predictions
                ),
                "restored_rouge1": sum(
                    row["restored_rouge1"] for row in predictions
                    if row["family"] == family
                ) / sum(row["family"] == family for row in predictions),
            }
            for family in sorted({row["family"] for row in predictions})
        },
        "realizer_only_oracle_plan": True,
        "end_to_end_competition_claim_allowed": False,
        "public_test_answers_used": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
