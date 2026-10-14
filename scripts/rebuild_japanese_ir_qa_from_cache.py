#!/usr/bin/env python3
"""Rebuild Japanese IR Q&A pairs from the audited local document cache."""

from __future__ import annotations

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from collect_japanese_ir_qa import (  # noqa: E402
    benchmark_pairs,
    benchmark_similarity,
    extract_pairs,
    normalize,
)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--corpus-dir", type=Path,
        default=ROOT / "data/external/japanese_ir_qa",
    )
    args = parser.parse_args()
    manifest_path = args.corpus_dir / "manifest.csv"
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    released = benchmark_pairs()
    pairs = []
    excluded = []
    seen_pair_hashes = set()
    seen_documents = set()
    missing_cached_text = []

    for row in manifest.to_dict("records"):
        if row["status"] != "ok" or not row["sha256"]:
            continue
        document_key = (str(row["issuer"]), str(row["sha256"]))
        if document_key in seen_documents:
            continue
        seen_documents.add(document_key)
        text_path = (
            args.corpus_dir / "text" / str(row["issuer"])
            / f"{row['sha256']}.txt"
        )
        if not text_path.is_file():
            missing_cached_text.append(str(text_path))
            continue
        text = text_path.read_text(errors="replace")
        for pair_index, (question, answer) in enumerate(extract_pairs(text)):
            pair_hash = sha256(
                (normalize(question) + "\x1f" + normalize(answer)).encode()
            ).hexdigest()
            if pair_hash in seen_pair_hashes:
                continue
            similarity, benchmark_index = benchmark_similarity(
                question, answer, released,
            )
            record = {
                "pair_id": pair_hash,
                "issuer": str(row["issuer"]),
                "source_url": str(row["final_url"] or row["url"]),
                "source_sha256": str(row["sha256"]),
                "source_pair_index": pair_index,
                "question": question,
                "response": answer,
                "benchmark_max_similarity": round(similarity, 6),
                "benchmark_nearest_id": benchmark_index,
                "license": "public-corporate-IR; reuse_terms_not_asserted",
            }
            if similarity >= 0.92:
                excluded.append(record)
                continue
            seen_pair_hashes.add(pair_hash)
            pairs.append(record)

    if missing_cached_text:
        raise ValueError(
            f"manifest references {len(missing_cached_text)} missing text files"
        )
    for filename, records in (
        ("pairs.jsonl", pairs),
        ("excluded_benchmark_near_duplicates.jsonl", excluded),
    ):
        with (args.corpus_dir / filename).open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "documents_attempted": len(manifest),
        "documents_ok": int((manifest.status == "ok").sum()),
        "documents_error": int((manifest.status == "error").sum()),
        "unique_cached_documents_rebuilt": len(seen_documents),
        "issuers_with_pairs": len({record["issuer"] for record in pairs}),
        "qa_pairs": len(pairs),
        "excluded_benchmark_near_duplicates": len(excluded),
        "maximum_retained_benchmark_similarity": max(
            (record["benchmark_max_similarity"] for record in pairs),
            default=0.0,
        ),
        "offline_rebuild": True,
    }
    (args.corpus_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
