#!/usr/bin/env python3
"""Evaluate large LLMs as leakage-controlled Task 3 fact planners.

The model can select only IDs from an input-derived typed fact graph.  It never
sees a target reference and never writes answer prose or numeric values.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import time
from urllib import request

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COMPANIES = ("HON", "JNJ", "MSFT", "UVV")
FAMILIES = (
    "allocation", "balance", "capex", "cash", "margin", "rd", "revenue",
    "revenue_focus",
)
SYSTEM = """You are a financial QA content planner. Choose which offered facts a
short reference-style answer should mention in order to answer the question.
Return JSON only, with exactly one field named selected containing an ordered
list of fact IDs. You may return an empty list for a qualitative answer. Never
return an ID that is not offered. Do not calculate, explain, or write prose.
The examples are from unrelated companies and teach content-selection policy."""


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fact_only(record: dict) -> bool:
    return all(slot["source"] == "fact" for slot in record["plan"]["requested_slots"])


def record_key(record: dict) -> tuple[str, str, str]:
    plan = record["plan"]
    return str(plan["task_id"]), str(plan["tier"]), str(plan["family"])


def candidate_facts(record: dict) -> list[dict]:
    result = []
    for fact in record["plan"]["numeric"]:
        for index, value in enumerate(fact["values"]):
            result.append({
                "id": f"{fact['metric']}:{index}",
                "unit": str(fact["unit"]),
                "derived": bool(fact["derived"]),
                "value": float(f"{float(value['value']):.8g}"),
            })
    return result


def target_ids(record: dict) -> list[str]:
    ordered = []
    for slot in record["plan"]["requested_slots"]:
        identifier = f"{slot['metric']}:{int(slot['index'])}"
        if identifier not in ordered:
            ordered.append(identifier)
    return ordered


def stable_choice(records: list[tuple[int, dict]], prefix: str) -> tuple[int, dict]:
    return min(
        records,
        key=lambda item: hashlib.sha256(
            (prefix + f"|{item[1]['plan']['task_id']}|{item[0]}").encode()
        ).hexdigest(),
    )


def select_screen(loco_dir: Path) -> list[dict]:
    selected = []
    for held in COMPANIES:
        rows = [
            row for row in load_jsonl(loco_dir / held / "evaluation.jsonl")
            if fact_only(row)
        ]
        by_family = defaultdict(list)
        for index, row in enumerate(rows):
            by_family[row["plan"]["family"]].append((index, row))
        if set(by_family) != set(FAMILIES):
            raise ValueError(f"incomplete family coverage for {held}: {sorted(by_family)}")
        for family in FAMILIES:
            index, row = stable_choice(
                by_family[family], f"large-planner-v1|{held}|{family}",
            )
            selected.append({"held_company": held, "record_index": index, "record": row})
    return selected


def select_demonstrations(corpus: list[dict], held: str, family: str) -> list[dict]:
    demos = []
    for donor in COMPANIES:
        if donor == held:
            continue
        eligible = [
            (int(row["source_index"]), row) for row in corpus
            if row["company"] == donor and row["plan"]["family"] == family
            and fact_only(row)
        ]
        if not eligible:
            raise ValueError(f"no {family} demonstration for donor {donor}")
        _, chosen = stable_choice(
            eligible, f"large-planner-demo-v1|{held}|{family}|{donor}",
        )
        demos.append(chosen)
    return demos


def question(record: dict, cache: dict[str, pd.DataFrame]) -> str:
    source = str(record["source"])
    if source not in cache:
        cache[source] = pd.read_parquet(source, columns=["question"])
    return str(cache[source].iloc[int(record["source_index"])].question)


def prompt_for(record: dict, demos: list[dict], cache: dict[str, pd.DataFrame]) -> str:
    blocks = []
    for number, demo in enumerate(demos, 1):
        example = {
            "question": question(demo, cache),
            "family": demo["plan"]["family"],
            "candidate_facts": candidate_facts(demo),
        }
        blocks.append(
            f"EXAMPLE {number} INPUT\n"
            + json.dumps(example, ensure_ascii=False, sort_keys=True)
            + f"\nEXAMPLE {number} OUTPUT\n"
            + json.dumps({"selected": target_ids(demo)}, ensure_ascii=False)
        )
    target = {
        "question": question(record, cache),
        "family": record["plan"]["family"],
        "candidate_facts": candidate_facts(record),
    }
    blocks.append(
        "TARGET INPUT\n" + json.dumps(target, ensure_ascii=False, sort_keys=True)
        + "\nTARGET OUTPUT"
    )
    return "\n\n".join(blocks)


def complete(api_base: str, model: str, prompt: str) -> tuple[str, float]:
    schema = {
        "type": "object",
        "properties": {
            "selected": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["selected"],
        "additionalProperties": False,
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
        "format": schema,
        "options": {"temperature": 0, "num_predict": 192, "seed": 7},
    }
    req = request.Request(
        api_base.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    with request.urlopen(req, timeout=600) as response:
        result = json.loads(response.read())
    return str(result["message"]["content"]), time.time() - started


def parse_selection(raw: str, offered: list[str]) -> tuple[list[str] | None, str | None]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(parsed, dict) or set(parsed) != {"selected"}:
        return None, "invalid_schema"
    selected = parsed["selected"]
    if not isinstance(selected, list) or not all(isinstance(item, str) for item in selected):
        return None, "invalid_selected_type"
    if len(selected) != len(set(selected)):
        return None, "duplicate_id"
    if any(item not in offered for item in selected):
        return None, "unknown_id"
    return selected, None


def set_f1(target: list[str], prediction: list[str] | None) -> float:
    if prediction is None:
        return 0.0
    left, right = set(target), set(prediction)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return 2 * len(left & right) / (len(left) + len(right))


def ordered_lcs_f1(target: list[str], prediction: list[str] | None) -> float:
    if prediction is None:
        return 0.0
    if not target and not prediction:
        return 1.0
    if not target or not prediction:
        return 0.0
    table = [[0] * (len(prediction) + 1) for _ in range(len(target) + 1)]
    for i, left in enumerate(target, 1):
        for j, right in enumerate(prediction, 1):
            table[i][j] = (
                table[i - 1][j - 1] + 1 if left == right
                else max(table[i - 1][j], table[i][j - 1])
            )
    return 2 * table[-1][-1] / (len(target) + len(prediction))


def summarize(rows: list[dict], field: str) -> dict:
    scores = []
    ordered_scores = []
    exact = true_positive = predicted_count = gold_count = 0
    nonempty_scores = []
    valid = 0
    for row in rows:
        prediction = row[field]
        target = row["target"]
        score = set_f1(target, prediction)
        scores.append(score)
        ordered_scores.append(ordered_lcs_f1(target, prediction))
        if target:
            nonempty_scores.append(score)
        if prediction is not None:
            valid += 1
            exact += int(target == prediction or set(target) == set(prediction))
            true_positive += len(set(target) & set(prediction))
            predicted_count += len(prediction)
        gold_count += len(target)
    precision = true_positive / predicted_count if predicted_count else 0.0
    recall = true_positive / gold_count if gold_count else 0.0
    return {
        "rows": len(rows),
        "valid_outputs": valid,
        "exact_set_accuracy": exact / len(rows),
        "macro_set_f1": sum(scores) / len(scores),
        "nonempty_macro_set_f1": (
            sum(nonempty_scores) / len(nonempty_scores)
            if nonempty_scores else 0.0
        ),
        "ordered_lcs_f1": sum(ordered_scores) / len(ordered_scores),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def baseline_lookup(path: Path) -> dict[tuple[str, str, str, str], dict]:
    return {
        (row["held_company"], row["task_id"], row["tier"], row["family"]): row
        for row in load_jsonl(path)
    }


def baseline_ids(pairs: list[list]) -> list[str]:
    return [f"{metric}:{int(index)}" for metric, index in pairs]


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True)
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
        "--output-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_large_content_planner_v1",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    preregistration = ROOT / "docs/polyfiqa_large_content_planner_preregistration.md"
    preregistration_sha = hashlib.sha256(preregistration.read_bytes()).hexdigest()
    amendment = ROOT / "docs/polyfiqa_large_content_planner_compliance_amendment.md"
    amendment_sha = hashlib.sha256(amendment.read_bytes()).hexdigest()
    corpus = load_jsonl(args.corpus_dir / "training.jsonl")
    targets = select_screen(args.corpus_dir / "loco")
    baselines = baseline_lookup(args.baseline)
    question_cache: dict[str, pd.DataFrame] = {}

    prepared = []
    for item in targets:
        row = item["record"]
        held = item["held_company"]
        demos = select_demonstrations(corpus, held, row["plan"]["family"])
        key = (held, *record_key(row))
        baseline = baselines[key]
        prepared.append({
            "held_company": held,
            "record_index": item["record_index"],
            "task_id": row["plan"]["task_id"],
            "tier": row["plan"]["tier"],
            "family": row["plan"]["family"],
            "source": row["source"],
            "source_index": row["source_index"],
            "target": target_ids(row),
            "offered": [fact["id"] for fact in candidate_facts(row)],
            "fixed_threshold": baseline_ids(baseline["threshold"]),
            "two_stage": baseline_ids(baseline["two_stage"]),
            "topk": baseline_ids(baseline["topk"]),
            "select_all": baseline_ids(baseline["all"]),
            "demonstrations": [
                {"company": demo["company"], "task_id": demo["plan"]["task_id"],
                 "source_index": demo["source_index"]}
                for demo in demos
            ],
            "prompt": prompt_for(row, demos, question_cache),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "preregistration_sha256": preregistration_sha,
        "compliance_amendment_sha256": amendment_sha,
        "selection": "one stable-hash record per 32 company-family cells",
        "rows": [{k: v for k, v in row.items() if k != "prompt"} for row in prepared],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    if args.prepare_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    by_identity = {
        (row["held_company"], row["task_id"], row["tier"], row["family"]): row
        for row in prepared
    }
    summaries = {}
    for model in args.model:
        output_path = args.output_dir / f"{safe_model_name(model)}.jsonl"
        prior = load_jsonl(output_path) if output_path.exists() else []
        completed = {
            (row["held_company"], row["task_id"], row["tier"], row["family"])
            for row in prior
        }
        for row in prepared:
            identity = (row["held_company"], row["task_id"], row["tier"], row["family"])
            if identity in completed:
                continue
            raw, seconds = complete(args.api_base, model, row["prompt"])
            selection, error = parse_selection(raw, row["offered"])
            result = {
                k: v for k, v in row.items()
                if k not in {"prompt", "fixed_threshold", "two_stage", "topk", "select_all"}
            }
            result.update({
                "model": model, "raw": raw, "prediction": selection,
                "error": error, "seconds": seconds,
                "reference_exposed_to_target": False,
                "public_test_answers_used": False,
            })
            with output_path.open("a") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                model, row["held_company"], row["family"],
                error or selection, f"f1={set_f1(row['target'], selection):.3f}",
                f"seconds={seconds:.2f}", flush=True,
            )

        outputs = load_jsonl(output_path)
        output_map = {
            (row["held_company"], row["task_id"], row["tier"], row["family"]): row
            for row in outputs
        }
        scored = []
        for identity, base in by_identity.items():
            row = dict(base)
            row["model_prediction"] = output_map[identity]["prediction"]
            scored.append(row)
        model_metrics = summarize(scored, "model_prediction")
        incumbent_metrics = summarize(scored, "two_stage")
        company_deltas = {}
        for held in COMPANIES:
            subset = [row for row in scored if row["held_company"] == held]
            company_deltas[held] = (
                summarize(subset, "model_prediction")["macro_set_f1"]
                - summarize(subset, "two_stage")["macro_set_f1"]
            )
        advance = (
            model_metrics["macro_set_f1"] >= incumbent_metrics["macro_set_f1"] + 0.01
            and sum(delta >= 0 for delta in company_deltas.values()) >= 3
            and model_metrics["valid_outputs"] >= 30
        )
        summaries[model] = {
            "model": model,
            "metrics": model_metrics,
            "two_stage_incumbent": incumbent_metrics,
            "delta_macro_set_f1": (
                model_metrics["macro_set_f1"] - incumbent_metrics["macro_set_f1"]
            ),
            "company_deltas": company_deltas,
            "advance_to_full_154": advance,
        }

    baseline_summary = {
        key: summarize(prepared, key)
        for key in ("fixed_threshold", "two_stage", "topk", "select_all")
    }
    summary = {
        "preregistration_sha256": preregistration_sha,
        "compliance_amendment_sha256": amendment_sha,
        "screen_rows": len(prepared),
        "baselines": baseline_summary,
        "models": summaries,
        "end_to_end_answer_claim": False,
        "public_test_answers_used": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
