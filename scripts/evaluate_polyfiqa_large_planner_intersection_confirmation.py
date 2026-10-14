#!/usr/bin/env python3
"""Independently confirm the Task 3 incumbent/large-planner intersection.

The 35B planner can only delete facts from the incumbent ordered plan.  Target
plans are selected and written only after the preregistration has been frozen;
they are never included in prompts or row-level progress output.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_polyfiqa_large_content_planner import (  # noqa: E402
    COMPANIES,
    baseline_ids,
    baseline_lookup,
    candidate_facts,
    complete,
    fact_only,
    load_jsonl,
    parse_selection,
    prompt_for,
    record_key,
    safe_model_name,
    select_demonstrations,
    summarize,
    target_ids,
)


MODEL = "qwen3.6:35b-a3b"
FAMILIES = (
    "balance", "capex", "cash", "margin", "rd", "revenue", "revenue_focus",
)
PREREGISTRATION_SHA256 = (
    "889659404b53dd67cbec1161cca2e98ee55155976f4edb1af4d1640dbe4f55a1"
)


def v1_identities(path: Path) -> set[tuple[str, str, int]]:
    manifest = json.loads(path.read_text())
    return {
        (str(row["held_company"]), str(row["source"]), int(row["source_index"]))
        for row in manifest["rows"]
    }


def confirmation_hash(held: str, family: str, record_index: int, record: dict) -> str:
    value = (
        "large-planner-intersection-confirm-v1"
        f"|{held}|{family}|{record['plan']['task_id']}|{record_index}"
    )
    return hashlib.sha256(value.encode()).hexdigest()


def select_confirmation_screen(loco_dir: Path, excluded: set[tuple[str, str, int]]) -> list[dict]:
    selected = []
    for held in COMPANIES:
        rows = [row for row in load_jsonl(loco_dir / held / "evaluation.jsonl") if fact_only(row)]
        by_family: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        for record_index, row in enumerate(rows):
            identity = (held, str(row["source"]), int(row["source_index"]))
            family = str(row["plan"]["family"])
            if identity not in excluded and family in FAMILIES:
                by_family[family].append((record_index, row))
        if set(by_family) != set(FAMILIES):
            raise ValueError(f"incomplete unused family coverage for {held}: {sorted(by_family)}")
        for family in FAMILIES:
            record_index, row = min(
                by_family[family],
                key=lambda item: confirmation_hash(held, family, item[0], item[1]),
            )
            selected.append({"held_company": held, "record_index": record_index, "record": row})
    return selected


def intersection(incumbent: list[str], large: list[str] | None) -> list[str]:
    if large is None:
        return list(incumbent)
    allowed = set(large)
    return [identifier for identifier in incumbent if identifier in allowed]


def score_delta(target: list[str], candidate: list[str], incumbent: list[str]) -> float:
    # Import locally so tests of selection do not duplicate the metric definition.
    from scripts.evaluate_polyfiqa_large_content_planner import set_f1

    return set_f1(target, candidate) - set_f1(target, incumbent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL, choices=[MODEL])
    parser.add_argument("--api-base", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--corpus-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_graph_to_reference_v5",
    )
    parser.add_argument(
        "--baseline", type=Path,
        default=ROOT / "artifacts/polyfiqa_numeric_planner_v5/predictions.jsonl",
    )
    parser.add_argument(
        "--v1-manifest", type=Path,
        default=ROOT / "artifacts/polyfiqa_large_content_planner_v1/manifest.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_large_planner_intersection_confirmation_v1",
    )
    args = parser.parse_args()

    preregistration = ROOT / "docs/polyfiqa_large_planner_intersection_confirmation_preregistration.md"
    preregistration_sha = hashlib.sha256(preregistration.read_bytes()).hexdigest()
    if preregistration_sha != PREREGISTRATION_SHA256:
        raise ValueError(
            f"preregistration changed: {preregistration_sha} != {PREREGISTRATION_SHA256}"
        )

    corpus = load_jsonl(args.corpus_dir / "training.jsonl")
    selected = select_confirmation_screen(
        args.corpus_dir / "loco", v1_identities(args.v1_manifest),
    )
    baselines = baseline_lookup(args.baseline)
    question_cache = {}
    prepared = []
    for item in selected:
        row = item["record"]
        held = item["held_company"]
        family = str(row["plan"]["family"])
        demos = select_demonstrations(corpus, held, family)
        baseline = baselines[(held, *record_key(row))]
        prepared.append({
            "held_company": held,
            "record_index": item["record_index"],
            "task_id": str(row["plan"]["task_id"]),
            "tier": str(row["plan"]["tier"]),
            "family": family,
            "source": str(row["source"]),
            "source_index": int(row["source_index"]),
            "target": target_ids(row),
            "offered": [fact["id"] for fact in candidate_facts(row)],
            "incumbent": baseline_ids(baseline["two_stage"]),
            "demonstrations": [
                {
                    "company": demo["company"],
                    "task_id": demo["plan"]["task_id"],
                    "source_index": demo["source_index"],
                }
                for demo in demos
            ],
            "prompt": prompt_for(row, demos, question_cache),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "preregistration_sha256": preregistration_sha,
        "selection": "one unused stable-hash record per 28 company-family cells",
        "excluded_v1_target_count": len(v1_identities(args.v1_manifest)),
        "rows": [{key: value for key, value in row.items() if key != "prompt"} for row in prepared],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )

    output_path = args.output_dir / f"{safe_model_name(args.model)}.jsonl"
    prior = load_jsonl(output_path) if output_path.exists() else []
    completed = {
        (row["held_company"], row["task_id"], row["tier"], row["family"])
        for row in prior
    }
    for ordinal, row in enumerate(prepared, 1):
        identity = (row["held_company"], row["task_id"], row["tier"], row["family"])
        if identity in completed:
            continue
        raw, seconds = complete(args.api_base, args.model, row["prompt"])
        selection, error = parse_selection(raw, row["offered"])
        result = {
            key: value for key, value in row.items()
            if key not in {"prompt", "target", "incumbent"}
        }
        result.update({
            "model": args.model,
            "raw": raw,
            "prediction": selection,
            "error": error,
            "seconds": seconds,
            "reference_exposed_to_target": False,
            "public_test_answers_used": False,
        })
        with output_path.open("a") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(
            f"completed={ordinal}/{len(prepared)} company={row['held_company']} "
            f"family={row['family']} valid={error is None} seconds={seconds:.2f}",
            flush=True,
        )

    outputs = load_jsonl(output_path)
    output_map = {
        (row["held_company"], row["task_id"], row["tier"], row["family"]): row
        for row in outputs
    }
    scored = []
    for row in prepared:
        identity = (row["held_company"], row["task_id"], row["tier"], row["family"])
        model_prediction = output_map[identity]["prediction"]
        scored.append({
            **{key: value for key, value in row.items() if key != "prompt"},
            "model_prediction": model_prediction,
            "hybrid": intersection(row["incumbent"], model_prediction),
        })

    incumbent_metrics = summarize(scored, "incumbent")
    model_metrics = summarize(scored, "model_prediction")
    hybrid_metrics = summarize(scored, "hybrid")
    deltas = [score_delta(row["target"], row["hybrid"], row["incumbent"]) for row in scored]
    company_deltas = {}
    for held in COMPANIES:
        subset = [row for row in scored if row["held_company"] == held]
        company_deltas[held] = (
            summarize(subset, "hybrid")["macro_set_f1"]
            - summarize(subset, "incumbent")["macro_set_f1"]
        )
    changed = sum(row["hybrid"] != row["incumbent"] for row in scored)
    wins = sum(delta > 1e-12 for delta in deltas)
    harms = sum(delta < -1e-12 for delta in deltas)
    valid = model_metrics["valid_outputs"]
    delta_macro = hybrid_metrics["macro_set_f1"] - incumbent_metrics["macro_set_f1"]
    gate_checks = {
        "macro_delta_at_least_0_01": delta_macro >= 0.01 - 1e-12,
        "at_least_three_nonnegative_companies": sum(
            delta >= -1e-12 for delta in company_deltas.values()
        ) >= 3,
        "no_company_below_minus_0_01": min(company_deltas.values()) >= -0.01 - 1e-12,
        "exact_set_accuracy_nonnegative": (
            hybrid_metrics["exact_set_accuracy"] >= incumbent_metrics["exact_set_accuracy"] - 1e-12
        ),
        "at_least_two_changed_rows": changed >= 2,
        "wins_exceed_harms": wins > harms,
        "at_most_one_harm": harms <= 1,
        "at_least_27_valid_outputs": valid >= 27,
    }
    summary = {
        "preregistration_sha256": preregistration_sha,
        "confirmation_rows": len(scored),
        "model": args.model,
        "incumbent": incumbent_metrics,
        "large_planner": model_metrics,
        "intersection_hybrid": hybrid_metrics,
        "hybrid_delta_macro_set_f1": delta_macro,
        "company_deltas": company_deltas,
        "changed_rows": changed,
        "wins": wins,
        "harms": harms,
        "invalid_outputs": len(scored) - valid,
        "gate_checks": gate_checks,
        "advance_to_full_history": all(gate_checks.values()),
        "discovery_screen_used_for_promotion": False,
        "reference_exposed_to_model": False,
        "public_test_answers_used": False,
    }
    (args.output_dir / "scored.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored)
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
