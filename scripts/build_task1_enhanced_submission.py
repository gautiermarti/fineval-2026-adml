#!/usr/bin/env python3
"""Build, validate, and fingerprint the promoted Task 1 grouped-consensus lanes."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
sys.path.insert(0, str(ROOT / "scripts"))

from build_submission_portfolio import (  # noqa: E402
    deduplicate_candidate_records,
    digest,
    display_path,
    file_record,
)


TEXTUAL_MODELS = [
    ROOT / "artifacts/task1_textual_full/legal_base_seed7",
    ROOT / "artifacts/task1_textual_full/legal_ner_init_seed7",
    ROOT / "artifacts/task1_textual/finance_adapted_seed7",
]
MULTILABEL_MODEL = ROOT / "artifacts/task1_textual_full/multilabel_legal_seed7"
VALIDATION_SAFE_TEXTUAL_MODELS = [
    ROOT / "artifacts/task1_textual/legal_base_seed7",
    ROOT / "artifacts/task1_textual/legal_ner_init_seed7",
    ROOT / "artifacts/task1_textual/finance_adapted_seed7",
]
VALIDATION_SAFE_MULTILABEL_MODEL = (
    ROOT / "artifacts/task1_textual/multilabel_legal_seed7"
)
FULL_GLINER = ROOT / "artifacts/task1_gliner_finetuned_seed7_cpu80/final"
SEED7_FOLDS = ROOT / "artifacts/task1_gliner_deploy_seed7_width16"
SEED19_FOLDS = ROOT / "artifacts/task1_gliner_deploy_seed19_width16"
VALIDATION_INPUT_SHA256 = "f1ad98216089445318e83eff7e8a04cef9fcc8fb86cb2288e9275180b9497a59"
VALIDATION_PRIMARY_SHA256 = "837a7a9dcda8cffba4e28658c6f582642053d44e6c838db00bd188d88d000f69"


def run(*arguments: str) -> None:
    subprocess.run([str(PYTHON), *arguments], cwd=ROOT, check=True)


def candidate_record(name: str, path: Path) -> dict:
    return {"name": name, "task": 1, **file_record(path)}


def normalize_and_validate(
    input_path: Path,
    predictions: Path,
    output: Path,
    report: Path,
) -> None:
    """Apply input-only invariants before admitting a Task 1 candidate."""
    run(
        "scripts/normalize_task1_submission.py",
        "--input", str(input_path),
        "--predictions", str(predictions),
        "--output", str(output),
        "--report", str(report),
    )
    run(
        "scripts/validate_submission.py", "--task", "1",
        "--input", str(input_path), "--predictions", str(output),
    )


def update_manifest(
    manifest: dict, candidates: list[dict], *, input_path: Path,
    overlap_fraction: float | None = None,
) -> dict:
    existing = {
        row["name"]: row for row in manifest.get("candidates", [])
    }
    existing.update({row["name"]: row for row in candidates})
    unique, aliases = deduplicate_candidate_records(
        list(existing.values()),
        preferred_names=(
            "task1_grouped_consensus_primary",
            "task1_two_seed_consensus_diversity",
            "task1_grouped_lookup_safe",
            "task1_mixed_full_refit",
        ),
    )
    manifest["manifest_format_version"] = 2
    manifest["candidates"] = unique
    previous_aliases = {
        (row["task"], row["name"], row["sha256"]): row
        for row in manifest.get("deduplicated_candidates", [])
    }
    previous_aliases.update({
        (row["task"], row["name"], row["sha256"]): row for row in aliases
    })
    manifest["deduplicated_candidates"] = list(previous_aliases.values())
    manifest.setdefault("inputs", {})["task1"] = file_record(input_path)
    support = manifest.setdefault("decision_support", {}).setdefault("task1", {})
    if overlap_fraction is not None:
        support["public_overlap_fraction"] = float(overlap_fraction)
    overlap = float(support.get("public_overlap_fraction", 0.0))
    support.update({
        "policy_independent_primary": "task1_grouped_consensus_primary",
        "policy_independent_diversity": "task1_two_seed_consensus_diversity",
        "public_reuse_candidate": "task1_grouped_lookup_safe",
        "recommended_first_run": (
            "task1_grouped_lookup_safe (POLICY CONFIRMATION REQUIRED)"
            if overlap > 0 else "task1_grouped_consensus_primary"
        ),
        "textual_validation_macro_entity_f1": 0.933157,
        "numeric_validation_macro_entity_f1": 0.985,
        "numeric_released_public_test_macro_entity_f1": 0.9941196581196582,
        "numeric_released_public_test_score_is_non_blind_diagnostic": True,
        "numeric_calendar_suffix_dedup_validation_changed_rows": 0,
        "numeric_calendar_suffix_dedup_post_public_hypothesis": True,
        "full_refit_public_test_macro_entity_f1": 0.8942362082362082,
        "full_refit_public_test_baseline_macro_entity_f1": 0.882068931068931,
    })
    return manifest


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    path.with_suffix(".sha256").write_text(
        f"{digest(path)}  {path.name}\n"
    )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--portfolio-manifest", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--validation-safe-replay", action="store_true",
        help="Use train-only incumbents and require exact promoted validation hash.",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work = output_dir / "task1_enhanced_work"
    work.mkdir(parents=True, exist_ok=True)
    schema_columns = pq.read_schema(input_path).names
    if "answer" in schema_columns:
        effective_input = work / "label_stripped_input.parquet"
        pd.read_parquet(
            input_path, columns=[column for column in schema_columns if column != "answer"],
        ).to_parquet(effective_input, index=False)
    else:
        effective_input = input_path
    textual_input = work / "textual_input.parquet"
    run(
        "scripts/prepare_task1_textual_subset.py", "--input", str(effective_input),
        "--output", str(textual_input),
    )
    textual_rows = len(pd.read_parquet(textual_input))
    textual_models = (
        VALIDATION_SAFE_TEXTUAL_MODELS
        if args.validation_safe_replay else TEXTUAL_MODELS
    )
    multilabel_model = (
        VALIDATION_SAFE_MULTILABEL_MODEL
        if args.validation_safe_replay else MULTILABEL_MODEL
    )

    base = args.base.resolve() if args.base else output_dir / "task1_mixed_full_refit.jsonl"
    if not args.base:
        run(
            "scripts/predict_task1_mixed.py", "--input", str(effective_input),
            "--output", str(base), "--numeric-final-refit", "--textual-models",
            *(str(path) for path in textual_models), "--device", args.device,
        )
    # A caller may supply a cached base produced before a newer input-only
    # invariant was introduced. Normalize it into a distinct file so the
    # enhanced merge cannot silently inherit stale numeric entities.
    normalized_base = output_dir / "task1_mixed_full_refit_normalized.jsonl"
    normalize_and_validate(
        effective_input,
        base,
        normalized_base,
        work / "mixed_full_refit_normalization.json",
    )
    flat = work / "textual_flat.jsonl"
    nested = work / "textual_nested.jsonl"
    run(
        "scripts/predict_task1_mixed.py", "--input", str(textual_input),
        "--output", str(flat), "--textual-models",
        *(str(path) for path in textual_models), "--device", args.device,
    )
    run(
        "scripts/predict_task1_mixed.py", "--input", str(textual_input),
        "--output", str(nested), "--textual-models",
        *(str(path) for path in textual_models), "--nested-multilabel",
        "--multilabel-model-dir", str(multilabel_model), "--device", args.device,
    )

    full_raw = work / "seed7_full_raw.jsonl"
    seed7 = work / "seed7"
    seed19 = work / "seed19"
    device_flag = ["--cpu"] if args.device == "cpu" else []
    run(
        "scripts/predict_task1_gliner_raw.py", "--model", str(FULL_GLINER),
        "--input", str(textual_input), "--output", str(full_raw),
        "--threshold", "0.05", "--batch-size", str(args.batch_size), *device_flag,
    )
    for models, destination in ((SEED7_FOLDS, seed7), (SEED19_FOLDS, seed19)):
        run(
            "scripts/predict_task1_gliner_fold_ensemble.py",
            "--models-dir", str(models), "--input", str(textual_input),
            "--output-dir", str(destination), "--threshold", "0.05",
            "--batch-size", str(args.batch_size), *device_flag,
        )

    primary_work = work / "grouped_primary"
    diversity_work = work / "grouped_diversity"
    common = [
        "--input", str(textual_input), "--incumbent", str(flat), str(nested),
        "--seed7-full-raw", str(full_raw),
        "--seed7-mean-raw", str(seed7 / "ensemble_mean_raw.jsonl"),
        "--seed7-vote-raw", str(seed7 / "ensemble_vote3_raw.jsonl"),
    ]
    run(
        "scripts/predict_task1_grouped_consensus.py", *common,
        "--consensus-mode", "single_seed", "--output-dir", str(primary_work),
    )
    run(
        "scripts/predict_task1_grouped_consensus.py", *common,
        "--seed19-mean-raw", str(seed19 / "ensemble_mean_raw.jsonl"),
        "--seed19-vote-raw", str(seed19 / "ensemble_vote3_raw.jsonl"),
        "--output-dir", str(diversity_work),
    )

    primary = output_dir / "task1_grouped_consensus_primary.jsonl"
    diversity = output_dir / "task1_two_seed_consensus_diversity.jsonl"
    for textual_predictions, destination in (
        (primary_work / "predictions.jsonl", primary),
        (diversity_work / "predictions.jsonl", diversity),
    ):
        merged = work / f"{destination.stem}_pre_normalization.jsonl"
        run(
            "scripts/merge_task1_textual_predictions.py",
            "--input", str(effective_input), "--base", str(normalized_base),
            "--textual-input", str(textual_input),
            "--textual-predictions", str(textual_predictions),
            "--output", str(merged),
        )
        normalize_and_validate(
            effective_input,
            merged,
            destination,
            work / f"{destination.stem}_normalization.json",
        )

    lookup = output_dir / "task1_grouped_lookup_safe.jsonl"
    overlap_audit = output_dir / "task1_grouped_overlap_audit.csv"
    run(
        "scripts/predict_task1_lookup_safe.py", "--input", str(effective_input),
        "--fallback", str(primary), "--output", str(lookup),
        "--audit-output", str(overlap_audit),
    )
    run(
        "scripts/validate_submission.py", "--task", "1",
        "--input", str(effective_input), "--predictions", str(lookup),
    )
    records = [
        candidate_record("task1_mixed_full_refit", normalized_base),
        candidate_record("task1_grouped_consensus_primary", primary),
        candidate_record("task1_two_seed_consensus_diversity", diversity),
        candidate_record("task1_grouped_lookup_safe", lookup),
    ]
    if args.validation_safe_replay:
        if digest(input_path) != VALIDATION_INPUT_SHA256:
            raise ValueError(
                "--validation-safe-replay is restricted to the pinned validation input"
            )
        if digest(primary) != VALIDATION_PRIMARY_SHA256:
            raise ValueError(
                "validation replay did not reproduce the promoted primary hash"
            )
    overlap_fraction = float(pd.read_csv(overlap_audit)["matched"].mean())
    if args.portfolio_manifest:
        manifest_path = args.portfolio_manifest.resolve()
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest_path = output_dir / "manifest.json"
        manifest = {
            "team": "ADML", "candidates": [], "decision_support": {},
        }
    manifest = update_manifest(
        manifest, records, input_path=input_path,
        overlap_fraction=overlap_fraction,
    )
    write_manifest(manifest_path, manifest)
    print(json.dumps({
        "input": display_path(input_path),
        "input_sha256": digest(input_path),
        "effective_input_answer_column_present": False,
        "textual_rows": textual_rows,
        "model_mode": (
            "validation_safe_train_only"
            if args.validation_safe_replay else "hidden_deployment_full_refit"
        ),
        "candidates": records,
        "manifest": display_path(manifest_path),
        "manifest_sha256": digest(manifest_path),
        "answer_column_accessed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
