#!/usr/bin/env python3
"""Evaluate a large LLM as a strictly constrained Task 3 surface realizer.

The historical v15 answer is delexicalized and only its answer core is exposed
for editing.  Its evidence suffix remains byte-for-byte unchanged.  A generated
core is accepted only when all immutable placeholders and all non-placeholder
numeric tokens are preserved and the mechanically restored full answer passes
the existing input-grounding validator.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from urllib import request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_polyfiqa_candidate_lattice import candidate_valid  # noqa: E402
from fineval.metrics import rouge1_f1  # noqa: E402
from fineval.polyfiqa_delex import (  # noqa: E402
    delexicalize_reference,
    realization_plan,
    restore_reference,
)
from fineval.polyfiqa_facts import build_fact_graph  # noqa: E402
from fineval.polyfiqa_programs import NUMBER_RE  # noqa: E402
from predict_polyfiqa_hybrid import ASIAN_NUMBERS  # noqa: E402


SYSTEM = """You are a constrained financial-answer surface realizer.
Return JSON with exactly one string field named template. Rewrite only the
TARGET_DRAFT_TEMPLATE for clarity and benchmark-like financial prose.

Hard rules:
1. Preserve the exact multiset of every __PLACEHOLDER__ token.
2. Preserve the exact multiset of any literal numeric tokens.
3. Do not add, remove, generalize, strengthen, or change a factual claim.
4. Do not add evidence, commentary, headings, or facts absent from the draft.
5. Keep the Answer: prefix. Be concise.

The examples teach writing style only. Their companies and facts are unrelated
to the target."""

MARKER_RE = re.compile(
    r"__(?:VALUE_\d+|COMPANY(?:_POSSESSIVE)?|YEAR_[A-Z_]+)__"
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def marker_multiset(text: str) -> list[str]:
    return sorted(MARKER_RE.findall(text))


def literal_number_multiset(text: str) -> list[str]:
    without_markers = MARKER_RE.sub("", text)
    return sorted(match.group(0) for match in NUMBER_RE.finditer(without_markers))


def answer_suffix(answer: str, core: str) -> str:
    if not answer.startswith(core):
        raise ValueError("delexicalized core is not an exact answer prefix")
    return answer[len(core):]


def compact_plan(plan: dict) -> dict:
    return {
        key: plan[key] for key in (
            "tier", "family", "numeric", "requested_slots", "oracle_copy_slots",
        )
    }


def row_graph(row: dict):
    return build_fact_graph(
        row["internal_id"], row["query"], row["question"],
        ASIAN_NUMBERS.get(row["internal_id"]),
    )


def delex(row: dict, answer: str):
    graph = row_graph(row)
    record, reason = delexicalize_reference(
        row["internal_id"], answer, row["query"], graph,
    )
    return record, reason, graph


def select_targets(rows: list[dict]) -> list[dict]:
    """Select one label-blind row per company/family cell."""
    eligible = []
    for row in rows:
        if row["partition"] != "early_confirmation":
            continue
        record, _, _ = delex(row, row["candidates"]["baseline"])
        if record is not None:
            eligible.append(row)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in eligible:
        company = row["internal_id"].split("_", 1)[0]
        grouped.setdefault((company, row["family"]), []).append(row)
    selected = []
    for key in sorted(grouped):
        choices = sorted(
            grouped[key],
            key=lambda row: hashlib.sha256(
                ("large-realizer-v1|" + row["packet_fingerprint"]).encode()
            ).hexdigest(),
        )
        selected.append(choices[0])
    expected = {
        (company, family)
        for company in ("HON", "JNJ", "MSFT", "UVV")
        for family in ("allocation", "balance", "capex", "cash", "margin", "rd", "revenue")
    }
    observed = {
        (row["internal_id"].split("_", 1)[0], row["family"])
        for row in selected
    }
    if observed != expected:
        raise ValueError(f"incomplete company/family screen: {sorted(expected - observed)}")
    return selected


def demonstrations(corpus: list[dict], company: str, family: str, count: int = 2) -> list[dict]:
    eligible = [
        row for row in corpus
        if row["company"] != company and row["plan"]["family"] == family
        and not row["plan"].get("oracle_copy_slots")
    ]
    ranked = sorted(
        eligible,
        key=lambda row: hashlib.sha256(
            (
                "large-realizer-demo-v1|" + company + "|" + family + "|"
                + row["plan"]["task_id"] + "|" + str(row["source_index"])
            ).encode()
        ).hexdigest(),
    )
    return ranked[:count]


def make_prompt(row: dict, record: dict, graph, corpus: list[dict]) -> str:
    company = row["internal_id"].split("_", 1)[0]
    examples = []
    for number, demo in enumerate(demonstrations(corpus, company, row["family"]), 1):
        examples.append(
            f"EXAMPLE_{number}_PLAN\n"
            + json.dumps(compact_plan(demo["plan"]), ensure_ascii=False, sort_keys=True)
            + f"\nEXAMPLE_{number}_OUTPUT\n"
            + json.dumps({"template": demo["target_template"]}, ensure_ascii=False)
        )
    plan = realization_plan(record, graph, include_oracle_evidence=False)
    target = (
        "TARGET_QUESTION\n" + row["question"]
        + "\nTARGET_PLAN\n" + json.dumps(compact_plan(plan), ensure_ascii=False, sort_keys=True)
        + "\nTARGET_DRAFT_TEMPLATE\n" + record["template"]
        + "\nTARGET_OUTPUT"
    )
    return "\n\n".join(examples + [target])


def complete(api_base: str, model: str, prompt: str, backend: str) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]
    if backend == "ollama":
        endpoint = api_base.rstrip("/") + "/api/chat"
        payload = {
            "model": model, "messages": messages, "stream": False,
            "think": False, "format": "json",
            "options": {"temperature": 0, "num_predict": 256},
        }
    else:
        endpoint = api_base.rstrip("/") + "/v1/chat/completions"
        payload = {
            "model": model, "messages": messages, "temperature": 0,
            "max_tokens": 256, "response_format": {"type": "json_object"},
        }
    started = time.time()
    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=300) as response:
        result = json.loads(response.read())
    content = (
        result["message"]["content"] if backend == "ollama"
        else result["choices"][0]["message"]["content"]
    )
    template = json.loads(content)["template"]
    return str(template).strip(), time.time() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--backend", choices=("openai", "ollama"), default="openai")
    parser.add_argument(
        "--lattice", type=Path,
        default=ROOT / "artifacts/polyfiqa_historical_sparse_lattice_v1/lattice.jsonl",
    )
    parser.add_argument(
        "--demonstrations", type=Path,
        default=ROOT / "artifacts/polyfiqa_graph_to_reference_v5/training.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_large_realizer_pilot_v1",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.lattice)
    targets = select_targets(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection = {
        "selection_rule": "sha256 stable sample; one early-confirmation row per company/family",
        "score_blind": True,
        "rows": [
            {
                "row_index": row["row_index"], "task_id": row["task_id"],
                "internal_id": row["internal_id"], "tier": row["tier"],
                "family": row["family"], "packet_fingerprint": row["packet_fingerprint"],
            }
            for row in targets
        ],
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n"
    )
    if args.prepare_only:
        print(json.dumps(selection, ensure_ascii=False, indent=2))
        return

    corpus = load_jsonl(args.demonstrations)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model)
    output_path = args.output_dir / f"{safe_model}.jsonl"
    completed = {
        row["row_index"] for row in load_jsonl(output_path)
    } if output_path.exists() else set()
    lattice_by_index = {row["row_index"]: row for row in rows}

    for row in targets:
        if row["row_index"] in completed:
            continue
        baseline = row["candidates"]["baseline"]
        record, reason, graph = delex(row, baseline)
        if record is None:
            raise AssertionError(reason)
        generated, seconds = complete(
            args.api_base, args.model, make_prompt(row, record, graph, corpus), args.backend,
        )
        rejection = []
        if marker_multiset(generated) != marker_multiset(record["template"]):
            rejection.append("placeholder_multiset")
        if literal_number_multiset(generated) != literal_number_multiset(record["template"]):
            rejection.append("literal_number_multiset")
        restored_core = None
        candidate = baseline
        if not rejection:
            try:
                generated_record = dict(record)
                generated_record["template"] = generated
                restored_core = restore_reference(generated_record, graph, row["internal_id"])
                candidate = restored_core + answer_suffix(baseline, record["core"])
            except (KeyError, ValueError) as exc:
                rejection.append("restoration:" + type(exc).__name__)
                candidate = baseline
        validation_row = pd.Series({
            "query": row["query"], "question": row["question"],
            "task_id": row["task_id"], "tier": row["tier"],
        })
        if not rejection and not candidate_valid(candidate, validation_row, row["internal_id"]):
            rejection.append("input_grounding_or_structure")
            candidate = baseline
        baseline_score = rouge1_f1([row["reference"]], [baseline])
        candidate_score = rouge1_f1([row["reference"]], [candidate])
        result = {
            "row_index": row["row_index"], "task_id": row["task_id"],
            "internal_id": row["internal_id"], "company": row["internal_id"].split("_", 1)[0],
            "tier": row["tier"], "family": row["family"], "model": args.model,
            "baseline_template": record["template"], "generated_template": generated,
            "restored_core": restored_core, "accepted": not rejection,
            "rejection": rejection, "baseline": baseline, "candidate": candidate,
            "reference": row["reference"], "baseline_rouge1": baseline_score,
            "candidate_rouge1": candidate_score, "delta": candidate_score - baseline_score,
            "seconds": seconds, "public_test_answers_used": False,
        }
        with output_path.open("a") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(
            row["row_index"], row["internal_id"], row["family"],
            "accepted" if not rejection else ",".join(rejection),
            f"delta={result['delta']:.4f}", f"seconds={seconds:.2f}", flush=True,
        )

    results = [
        row for row in load_jsonl(output_path)
        if row["row_index"] in {target["row_index"] for target in targets}
    ]
    baseline_rouge = sum(row["baseline_rouge1"] for row in results) / len(results)
    candidate_rouge = sum(row["candidate_rouge1"] for row in results) / len(results)
    summary = {
        "model": args.model, "rows": len(results),
        "accepted_rows": sum(row["accepted"] for row in results),
        "baseline_rouge": baseline_rouge, "candidate_rouge": candidate_rouge,
        "delta": candidate_rouge - baseline_rouge,
        "wins": sum(row["delta"] > 0 for row in results),
        "ties": sum(row["delta"] == 0 for row in results),
        "losses": sum(row["delta"] < 0 for row in results),
        "rejections": dict(Counter(reason for row in results for reason in row["rejection"])),
        "by_company": {
            company: {
                "rows": len(group), "delta": sum(row["delta"] for row in group) / len(group),
                "wins": sum(row["delta"] > 0 for row in group),
                "losses": sum(row["delta"] < 0 for row in group),
            }
            for company in sorted({row["company"] for row in results})
            for group in [[row for row in results if row["company"] == company]]
        },
        "by_family": {
            family: {
                "rows": len(group), "delta": sum(row["delta"] for row in group) / len(group),
                "wins": sum(row["delta"] > 0 for row in group),
                "losses": sum(row["delta"] < 0 for row in group),
            }
            for family in sorted({row["family"] for row in results})
            for group in [[row for row in results if row["family"] == family]]
        },
        "free_form_editing_reused": False,
        "current_or_hidden_test_answers_used": False,
    }
    (args.output_dir / f"{safe_model}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
