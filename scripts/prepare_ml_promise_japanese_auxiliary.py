#!/usr/bin/env python3
"""Build a provenance-rich, benchmark-disjoint ML-Promise Japanese corpus."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

import pandas as pd
from rapidfuzz.fuzz import ratio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.parsing import parse_icr_query  # noqa: E402


LICENSE = "CC-BY-NC-SA-4.0 annotations; source reports retain publisher rights"
PRIMARY_ENDPOINT = "http://MLpromise.nlpfin.com/"
OVERLAPPING_HOSTS = {"www.itochu.co.jp", "itochu.co.jp"}


def normalize(value: str) -> str:
    value = str(value or "").replace("\ufeff", "").replace("\u3000", " ")
    value = re.sub(r"\s+", "", value)
    return value.casefold()


def grounded_span(span: str, text: str) -> bool:
    return bool(normalize(span)) and normalize(span) in normalize(text)


def benchmark_similarity(
    text: str, promise: str, benchmark: list[tuple[int, str, str]],
) -> tuple[float, int, str]:
    best = (0.0, -1, "")
    for identifier, question, response in benchmark:
        comparisons = (
            (ratio(text, question) / 100, "data_to_question"),
            (ratio(text, response) / 100, "data_to_response"),
            (ratio(promise, question) / 100 if promise else 0.0, "promise_to_question"),
            (ratio(promise, response) / 100 if promise else 0.0, "promise_to_response"),
        )
        score, view = max(comparisons)
        if score > best[0]:
            best = score, identifier, view
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "data/external/ml_promise/Training/PromiseEval_Trainset_Japanese.json",
    )
    parser.add_argument(
        "--benchmark", type=Path,
        default=ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/external/ml_promise_japanese_safe",
    )
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    args = parser.parse_args()

    source_bytes = args.input.read_bytes()
    rows = json.loads(source_bytes.decode("utf-8-sig"))
    # Deliberately load no answer column: target labels play no role in
    # auxiliary-data selection or overlap auditing.
    benchmark_frame = pd.read_parquet(args.benchmark, columns=["id", "query"])
    benchmark = []
    for row in benchmark_frame.itertuples(index=False):
        parsed = parse_icr_query(row.query)
        benchmark.append((int(row.id), normalize(parsed.question), normalize(parsed.response)))

    accepted = []
    rejected = []
    seen_texts: set[str] = set()
    for index, row in enumerate(rows):
        text = str(row.get("data", "")).strip()
        promise = str(row.get("promise_string", "")).strip()
        status = str(row.get("promise_status", "")).strip()
        source_url = str(row.get("URL", "")).strip()
        host = urlparse(source_url).netloc.casefold()
        base = {
            "original_index": index,
            "source_url": source_url,
            "source_host": host,
            "page_number": str(row.get("page_number", "")),
        }
        if host in OVERLAPPING_HOSTS:
            rejected.append(base | {"reason": "overlapping_jf_icr_issuer"})
            continue
        if status not in {"Yes", "No"}:
            rejected.append(base | {"reason": "invalid_promise_status"})
            continue
        if not normalize(text):
            rejected.append(base | {"reason": "empty_text"})
            continue
        if status == "Yes" and not grounded_span(promise, text):
            rejected.append(base | {"reason": "ungrounded_positive_promise_span"})
            continue
        if status == "No" and normalize(promise):
            rejected.append(base | {"reason": "negative_row_has_promise_span"})
            continue
        text_hash = sha256(normalize(text).encode()).hexdigest()
        if text_hash in seen_texts:
            rejected.append(base | {"reason": "duplicate_normalized_text"})
            continue
        similarity, nearest_id, similarity_view = benchmark_similarity(
            normalize(text), normalize(promise), benchmark,
        )
        audit = base | {
            "benchmark_max_similarity": round(similarity, 6),
            "benchmark_nearest_id": nearest_id,
            "benchmark_similarity_view": similarity_view,
        }
        if similarity >= args.near_duplicate_threshold:
            rejected.append(audit | {"reason": "benchmark_near_duplicate"})
            continue
        seen_texts.add(text_hash)
        accepted.append({
            "auxiliary_id": sha256(
                (source_url + "\x1f" + str(row.get("page_number", "")) + "\x1f" + text).encode()
            ).hexdigest(),
            **audit,
            "text": text,
            "promise_status": status,
            "promise_span": promise,
            "verification_timeline": str(row.get("verification_timeline", "")),
            "evidence_status": str(row.get("evidence_status", "")),
            "evidence_span": str(row.get("evidence_string", "")).strip(),
            "evidence_quality": str(row.get("evidence_quality", "")),
            "esg_type": str(row.get("ESG_type", "")),
            "license": LICENSE,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("auxiliary_train.jsonl", accepted), ("rejected.jsonl", rejected)):
        (args.output_dir / name).write_text("".join(
            json.dumps(value, ensure_ascii=False) + "\n" for value in values
        ))
    summary = {
        "primary_endpoint": PRIMARY_ENDPOINT,
        "source_file": str(args.input),
        "source_sha256": sha256(source_bytes).hexdigest(),
        "license": LICENSE,
        "source_rows": len(rows),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "accepted_status_counts": dict(Counter(row["promise_status"] for row in accepted)),
        "accepted_host_counts": dict(sorted(Counter(row["source_host"] for row in accepted).items())),
        "rejection_reasons": dict(sorted(Counter(row["reason"] for row in rejected).items())),
        "maximum_retained_benchmark_similarity": max(
            (row["benchmark_max_similarity"] for row in accepted), default=0.0,
        ),
        "near_duplicate_threshold": args.near_duplicate_threshold,
        "positive_spans_grounded": all(
            row["promise_status"] != "Yes" or grounded_span(row["promise_span"], row["text"])
            for row in accepted
        ),
        "benchmark_columns_loaded": ["id", "query"],
        "jf_icr_labels_used": False,
        "ml_promise_test_used": False,
        "current_or_hidden_test_used": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
