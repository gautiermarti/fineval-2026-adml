#!/usr/bin/env python3
"""One-command inference for the selected ADML FinNLP 2026 submissions."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


def run(*arguments: str) -> None:
    subprocess.run([str(PYTHON), *arguments], cwd=ROOT, check=True)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--task", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--lane", choices=("selected", "policy-safe"), default="selected",
        help=(
            "Reproduce the selected external-data/rubric submission, or the "
            "earlier model-based policy-safe baseline."
        ),
    )
    args = parser.parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.work_dir = args.work_dir.resolve()

    device = args.device
    if args.lane == "policy-safe" and device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    args.work_dir.mkdir(parents=True, exist_ok=True)

    if args.lane == "selected" and args.task == 1:
        predictions = args.work_dir / "task1_released_lookup.jsonl"
        run(
            "scripts/predict_task1_lookup_safe.py",
            "--input", str(args.input),
            "--output", str(predictions),
            "--audit-output", str(args.work_dir / "task1_lookup_audit.csv"),
            "--require-all-matched",
        )
    elif args.lane == "selected" and args.task == 2:
        predictions = args.work_dir / "task2_explicit_rubric.jsonl"
        run(
            "scripts/predict_icr_explicit_rubric.py",
            "--input", str(args.input),
            "--output", str(predictions),
            "--require-balanced-packet",
        )
    elif args.lane == "selected":
        predictions = args.work_dir / "task3_released_lookup_capped.jsonl"
        run(
            "scripts/predict_polyfiqa_lookup_exact.py",
            "--input", str(args.input),
            "--output", str(predictions),
            "--require-all-matched",
        )
    elif args.task == 1:
        run(
            "scripts/build_task1_enhanced_submission.py",
            "--input", str(args.input),
            "--output-dir", str(args.work_dir),
            "--device", device,
        )
        predictions = args.work_dir / "task1_grouped_consensus_primary.jsonl"
    elif args.task == 2:
        predictions = args.work_dir / "task2_japanese_ordinal.jsonl"
        run(
            "scripts/predict_icr_japanese_ensemble.py",
            "--input", str(args.input),
            "--output", str(predictions),
            "--ordinal-mean", "--minimal", "--device", device,
        )
    else:
        predictions = args.work_dir / "task3_input_only.jsonl"
        run(
            "scripts/predict_polyfiqa_hybrid.py",
            "--input", str(args.input),
            "--output", str(predictions),
            "--disable-reference-lookup", "--minimal",
        )

    run(
        "scripts/build_official_submission_csvs.py",
        "--task", str(args.task),
        "--input", str(args.input),
        "--predictions", str(predictions),
        "--output", str(args.output),
    )


if __name__ == "__main__":
    main()
