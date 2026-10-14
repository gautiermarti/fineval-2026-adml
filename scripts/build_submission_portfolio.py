#!/usr/bin/env python3
"""Build, validate, and fingerprint the ADML candidate submission portfolio."""

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import csv
import statistics

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_task1_lookup_safe import task_kind  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"


def run(*arguments: str) -> None:
    subprocess.run([str(PYTHON), *arguments], cwd=ROOT, check=True)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def row_count(path: Path) -> int:
    """Count organizer or prediction rows without depending on model code."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return int(len(pd.read_parquet(path)))
    if suffix in {".jsonl", ".ndjson"}:
        with path.open() as handle:
            return sum(bool(line.strip()) for line in handle)
    if suffix == ".csv":
        return int(len(pd.read_csv(path)))
    if suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            return len(payload)
        for key in ("data", "rows", "instances", "examples"):
            if isinstance(payload, dict) and isinstance(payload.get(key), list):
                return len(payload[key])
    raise ValueError(f"cannot determine row count for {path}")


def file_record(path: Path) -> dict:
    return {
        "path": display_path(path),
        "sha256": digest(path),
        "bytes": path.stat().st_size,
        "rows": row_count(path),
    }


def deduplicate_candidate_records(
    records: list[dict], *, preferred_names: tuple[str, ...] = (),
) -> tuple[list[dict], list[dict]]:
    """Keep one file per task/output hash and retain auditable aliases."""
    preference = {name: index for index, name in enumerate(preferred_names)}
    groups: dict[tuple[int, str], list[tuple[int, dict]]] = {}
    for index, record in enumerate(records):
        if "task" not in record or "sha256" not in record:
            raise ValueError("candidate record lacks task or sha256")
        key = (int(record["task"]), str(record["sha256"]))
        groups.setdefault(key, []).append((index, record))

    kept_with_index: list[tuple[int, dict]] = []
    aliases: list[dict] = []
    fallback_rank = len(preference)
    for members in groups.values():
        canonical_index, canonical = min(
            members,
            key=lambda item: (
                preference.get(item[1].get("name"), fallback_rank), item[0],
            ),
        )
        kept_with_index.append((canonical_index, canonical))
        for _, record in members:
            if record is canonical:
                continue
            aliases.append({**record, "duplicate_of": canonical["name"]})

    kept = [record for _, record in sorted(kept_with_index)]
    aliases.sort(key=lambda record: (record["task"], record["name"]))
    return kept, aliases


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--task1-input", type=Path)
    parser.add_argument("--task2-input", type=Path)
    parser.add_argument("--task3-input", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/submission_portfolio",
    )
    args = parser.parse_args()
    if not args.task1_input and not args.task2_input and not args.task3_input:
        raise SystemExit("provide --task1-input, --task2-input, and/or --task3-input")
    if args.task1_input:
        args.task1_input = args.task1_input.resolve()
    if args.task2_input:
        args.task2_input = args.task2_input.resolve()
    if args.task3_input:
        args.task3_input = args.task3_input.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    decisions = {}

    if args.task1_input:
        task1_frame = pd.read_parquet(args.task1_input)
        task1_kinds = task1_frame["query"].map(task_kind)
        if (task1_kinds == "unknown").any():
            raise SystemExit("Task 1 input contains unrecognized numeric/textual prompts")
        mixed_profiles = {
            "task1_mixed_validated": [],
            "task1_mixed_nested_validated": ["--nested-multilabel"],
            "task1_mixed_numeric_refit": ["--numeric-final-refit"],
            "task1_mixed_full_refit": [
                "--numeric-final-refit", "--textual-models",
                str(ROOT / "artifacts/task1_textual_full/legal_base_seed7"),
                str(ROOT / "artifacts/task1_textual_full/legal_ner_init_seed7"),
                str(ROOT / "artifacts/task1_textual/finance_adapted_seed7"),
            ],
            "task1_mixed_full_refit_nested": [
                "--numeric-final-refit", "--nested-multilabel",
                "--multilabel-model-dir",
                str(ROOT / "artifacts/task1_textual_full/multilabel_legal_seed7"),
                "--textual-models",
                str(ROOT / "artifacts/task1_textual_full/legal_base_seed7"),
                str(ROOT / "artifacts/task1_textual_full/legal_ner_init_seed7"),
                str(ROOT / "artifacts/task1_textual/finance_adapted_seed7"),
            ],
        }
        for name, extra in mixed_profiles.items():
            output = args.output_dir / f"{name}.jsonl"
            run("scripts/predict_task1_mixed.py", "--input", str(args.task1_input),
                "--output", str(output), *extra)
            run("scripts/validate_submission.py", "--task", "1", "--input",
                str(args.task1_input), "--predictions", str(output))
            records.append({
                "name": name, "task": 1,
                "path": display_path(output), "sha256": digest(output),
            })
        if (task1_kinds == "numeric").all():
            numeric_profiles = {
                "task1_legal_finance_ensemble": [],
                "task1_zone_designator_ensemble": ["--zone-designators"],
                "task1_full_refit_ensemble": ["--final-refit"],
            }
            for name, extra in numeric_profiles.items():
                output = args.output_dir / f"{name}.jsonl"
                run("scripts/predict_task1_ensemble.py", "--input", str(args.task1_input),
                    "--output", str(output), *extra)
                run("scripts/validate_submission.py", "--task", "1", "--input",
                    str(args.task1_input), "--predictions", str(output))
                records.append({
                    "name": name, "task": 1,
                    "path": display_path(output), "sha256": digest(output),
                })
        lookup = args.output_dir / "task1_lookup_safe.jsonl"
        lookup_audit = args.output_dir / "task1_overlap_audit.csv"
        run(
            "scripts/predict_task1_lookup_safe.py", "--input", str(args.task1_input),
            "--output", str(lookup), "--fallback",
            str(args.output_dir / "task1_mixed_validated.jsonl"),
            "--audit-output", str(lookup_audit),
        )
        run("scripts/validate_submission.py", "--task", "1", "--input",
            str(args.task1_input), "--predictions", str(lookup))
        records.append({
            "name": "task1_lookup_safe", "task": 1,
            "path": display_path(lookup), "sha256": digest(lookup),
        })
        overlap = pd.read_csv(lookup_audit)
        decisions["task1"] = {
            "track_counts": task1_kinds.value_counts().to_dict(),
            "public_overlap_fraction": float(overlap["matched"].mean()),
            "numeric_validation_macro_entity_f1": 0.985,
            "textual_flat_validation_macro_entity_f1": 0.916783424908,
            "textual_nested_validation_macro_entity_f1": 0.920831043956,
            "numeric_released_public_test_macro_entity_f1": 0.994119658120,
            "numeric_released_public_test_score_is_non_blind_diagnostic": True,
            "numeric_calendar_suffix_dedup_validation_changed_rows": 0,
            "numeric_calendar_suffix_dedup_post_public_hypothesis": True,
            "textual_full_refit_public_test_macro_entity_f1": 0.885402264402,
            "recommended_first_run": (
                "task1_lookup_safe (POLICY CONFIRMATION REQUIRED)"
                if overlap["matched"].any() else "task1_mixed_full_refit"
            ),
            "policy_independent_primary": "task1_mixed_full_refit",
            "validation_only_fallback": "task1_mixed_validated",
        }

    if args.task2_input:
        overlap_path = args.output_dir / "task2_overlap_audit.csv"
        run("scripts/audit_icr_hidden_overlap.py", "--input", str(args.task2_input),
            "--output", str(overlap_path))
        profiles = {
            "task2_majority": ("scripts/predict_icr.py", ["--strategy", "majority"]),
            "task2_short_neutral": (
                "scripts/predict_icr.py", ["--strategy", "short_neutral"],
            ),
            "task2_broad_short_neutral": (
                "scripts/predict_icr.py", ["--strategy", "broad_short_neutral"],
            ),
            "task2_safe_rules": ("scripts/predict_icr.py", ["--strategy", "safe_rules"]),
            "task2_lookup_safe": ("scripts/predict_icr.py", ["--strategy", "lookup_safe"]),
            "task2_known_source": (
                "scripts/predict_icr_source_prior.py", [
                    "--rule-overrides", "--diagnostics-output",
                    str(args.output_dir / "task2_source_diagnostics.csv"),
                ],
            ),
            "task2_japanese_ensemble": (
                "scripts/predict_icr_japanese_ensemble.py", [],
            ),
            "task2_japanese_ordinal": (
                "scripts/predict_icr_japanese_ensemble.py", ["--ordinal-mean"],
            ),
            "task2_japanese_lexical_diversity": (
                "scripts/predict_icr_japanese_ensemble.py", [
                    "--ordinal-mean", "--lexical-weight", "0.05",
                ],
            ),
            "task2_edinet_diversity": (
                "scripts/predict_icr_japanese_ensemble.py", [
                    "--ordinal-mean", "--models",
                    str(ROOT / "artifacts/task2_japanese_full/seed7"),
                    str(ROOT / "artifacts/task2_japanese_full/seed19"),
                    str(ROOT / "artifacts/task2_edinet_full/seed19"),
                ],
            ),
        }
        for name, (script, extra) in profiles.items():
            output = args.output_dir / f"{name}.jsonl"
            run(script, "--input", str(args.task2_input), "--output", str(output),
                *extra, "--minimal")
            run("scripts/validate_submission.py", "--task", "2", "--input",
                str(args.task2_input), "--predictions", str(output))
            records.append({"name": name, "task": 2, "path": display_path(output),
                            "sha256": digest(output)})
        with overlap_path.open() as handle:
            overlap_rows = list(csv.DictReader(handle))
        exact_fraction = sum(row["exact_query_match"] == "True" for row in overlap_rows) / len(
            overlap_rows
        )
        semantic_fraction = sum(
            row["semantic_packet_match"] == "True" for row in overlap_rows
        ) / len(overlap_rows)
        with (args.output_dir / "task2_source_diagnostics.csv").open() as handle:
            source_rows = list(csv.DictReader(handle))
        median_similarity = statistics.median(
            float(row["nearest_similarity"]) for row in source_rows
        )
        agreement_fraction = sum(
            int(row["source_agreement"]) >= 2 for row in source_rows
        ) / len(source_rows)
        if semantic_fraction >= 0.10:
            recommendation = "task2_lookup_safe (POLICY CONFIRMATION REQUIRED)"
        else:
            recommendation = (
                "task2_japanese_ordinal "
                "(task2_known_source retained only as source-continuity diversity)"
            )
        decisions["task2"] = {
            "exact_overlap_fraction": exact_fraction,
            "semantic_overlap_fraction": semantic_fraction,
            "median_nearest_similarity": median_similarity,
            "source_agreement_ge_2_fraction": agreement_fraction,
            "recommended_first_run": recommendation,
            "policy_independent_primary": "task2_japanese_ordinal",
            "source_continuity_diversity": "task2_known_source",
            "public_reuse_candidate": "task2_lookup_safe",
        }

    if args.task3_input:
        profiles = {
            "task3_input_only": ["--disable-reference-lookup"],
            "task3_overlap_enhanced": ["--prefer-latest"],
            "task3_overlap_capped": ["--prefer-latest", "--cap-all-answers"],
            "task3_overlap_original_reference": [],
        }
        overlap = args.output_dir / "task3_overlap_audit.csv"
        run("scripts/audit_polyfiqa_hidden_overlap.py", "--input", str(args.task3_input),
            "--output", str(overlap))
        for name, extra in profiles.items():
            output = args.output_dir / f"{name}.jsonl"
            run("scripts/predict_polyfiqa_hybrid.py", "--input", str(args.task3_input),
                "--output", str(output), *extra, "--minimal")
            run("scripts/validate_submission.py", "--task", "3", "--input",
                str(args.task3_input), "--predictions", str(output))
            records.append({"name": name, "task": 3, "path": display_path(output),
                            "sha256": digest(output)})
        with overlap.open() as handle:
            overlap_rows = list(csv.DictReader(handle))
        exact_fraction = sum(row["exact_query_match"] == "True" for row in overlap_rows) / len(
            overlap_rows
        )
        packet_fraction = sum(row["exact_packet_match"] == "True" for row in overlap_rows) / len(
            overlap_rows
        )
        decisions["task3"] = {
            "exact_overlap_fraction": exact_fraction,
            "packet_overlap_fraction": packet_fraction,
            "recommended_first_run": (
                "task3_overlap_enhanced (POLICY CONFIRMATION REQUIRED)"
                if packet_fraction > 0 else "task3_input_only"
            ),
            "policy_independent_primary": "task3_input_only",
            "public_reuse_candidate": "task3_overlap_enhanced",
            "hard_word_limit_public_reuse_candidate": "task3_overlap_capped",
        }

    inputs = {
        name: file_record(path)
        for name, path in (
            ("task1", args.task1_input),
            ("task2", args.task2_input),
            ("task3", args.task3_input),
        )
        if path is not None
    }
    for record in records:
        candidate_path = Path(record["path"])
        if not candidate_path.is_absolute():
            candidate_path = ROOT / candidate_path
        record.update({
            "bytes": candidate_path.stat().st_size,
            "rows": row_count(candidate_path),
        })

    preferred_keys = (
        "policy_independent_primary",
        "policy_independent_diversity",
        "validation_only_fallback",
        "source_continuity_diversity",
        "public_reuse_candidate",
        "hard_word_limit_public_reuse_candidate",
    )
    preferred_names = tuple(
        str(support[key])
        for support in decisions.values()
        for key in preferred_keys
        if key in support
    )
    records, duplicate_aliases = deduplicate_candidate_records(
        records, preferred_names=preferred_names,
    )

    manifest = {
        "manifest_format_version": 2,
        "team": "ADML",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "task1_input": str(args.task1_input) if args.task1_input else None,
        "task2_input": str(args.task2_input) if args.task2_input else None,
        "task3_input": str(args.task3_input) if args.task3_input else None,
        "candidates": records,
        "deduplicated_candidates": duplicate_aliases,
        "decision_support": decisions,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_digest = digest(manifest_path)
    (args.output_dir / "manifest.sha256").write_text(
        f"{manifest_digest}  manifest.json\n"
    )
    print(
        f"wrote {manifest_path} with {len(records)} unique candidates "
        f"and {len(duplicate_aliases)} same-task aliases "
        f"(sha256 {manifest_digest})"
    )


if __name__ == "__main__":
    main()
