#!/usr/bin/env python3
"""Audit whether GRFinNER snippets can be retrieved from the public corpus.

This is a labels-free retrieval audit.  It indexes whitespace/punctuation-
insensitive versions of every benchmark snippet with Aho--Corasick, then scans
the cleaned public-document corpus once.  Full-query matches are strong source
recovery evidence; multiple long anchors in the same document are reported as
weaker candidates for a later fuzzy/contextual retriever.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import unicodedata

import ahocorasick
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ALNUM_RE = re.compile(r"[^0-9a-zα-ωάέήίόύώϊϋΐΰ]+", re.IGNORECASE)


def compact(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    return ALNUM_RE.sub("", value)


def anchors(value: str, width: int) -> list[str]:
    if len(value) <= width:
        return [value]
    starts = (0, max(0, (len(value) - width) // 2), len(value) - width)
    return list(dict.fromkeys(value[start:start + width] for start in starts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-dir", type=Path,
        default=ROOT / "data/processed/finance_corpus_night",
    )
    parser.add_argument("--anchor-width", type=int, default=72)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task1_corpus_retrieval_audit_v1",
    )
    args = parser.parse_args()

    split_paths = {
        "train": ROOT / "data/external/plutus_finner_text/train-00000-of-00001.parquet",
        "validation": ROOT / "data/external/plutus_finner_text/validation-00000-of-00001.parquet",
        "test": ROOT / "data/external/plutus_finner_text/test-00000-of-00001.parquet",
    }
    queries = []
    for split, path in split_paths.items():
        frame = pd.read_parquet(path, columns=["text"])
        for row_index, text in enumerate(frame["text"].astype(str)):
            value = compact(text)
            queries.append({
                "query_id": len(queries), "split": split, "row_index": row_index,
                "characters": len(text), "compact_characters": len(value),
                "compact": value, "anchors": anchors(value, args.anchor_width),
            })

    automaton = ahocorasick.Automaton()
    pattern_payloads = defaultdict(list)
    for query in queries:
        pattern_payloads[query["compact"]].append(
            (query["query_id"], "full", -1)
        )
        for anchor_index, value in enumerate(query["anchors"]):
            pattern_payloads[value].append(
                (query["query_id"], "anchor", anchor_index)
            )
    for value, payload in pattern_payloads.items():
        if value:
            automaton.add_word(value, (value, payload))
    automaton.make_automaton()

    full_hits = defaultdict(set)
    anchor_hits = defaultdict(lambda: defaultdict(set))
    corpus_files = sorted(
        path for path in args.corpus_dir.rglob("*.txt") if path.is_file()
    )
    corpus_characters = 0
    for path in corpus_files:
        value = compact(path.read_text(errors="replace"))
        corpus_characters += len(value)
        relative = str(path.relative_to(args.corpus_dir))
        for _, (_, payload) in automaton.iter(value):
            for query_id, kind, anchor_index in payload:
                if kind == "full":
                    full_hits[query_id].add(relative)
                else:
                    anchor_hits[query_id][relative].add(anchor_index)

    records = []
    for query in queries:
        query_id = query["query_id"]
        full_documents = sorted(full_hits[query_id])
        anchor_documents = sorted(
            (
                {"document": document, "anchors": len(indices)}
                for document, indices in anchor_hits[query_id].items()
            ),
            key=lambda row: (-row["anchors"], row["document"]),
        )
        required = min(2, len(query["anchors"]))
        records.append({
            key: query[key] for key in (
                "query_id", "split", "row_index", "characters", "compact_characters"
            )
        } | {
            "full_match": bool(full_documents),
            "full_match_documents": full_documents,
            "multi_anchor_match": any(
                row["anchors"] >= required for row in anchor_documents
            ),
            "best_anchor_count": max(
                (row["anchors"] for row in anchor_documents), default=0
            ),
            "anchor_count": len(query["anchors"]),
            "best_anchor_documents": anchor_documents[:5],
        })

    split_summary = {}
    for split in split_paths:
        subset = [row for row in records if row["split"] == split]
        split_summary[split] = {
            "rows": len(subset),
            "full_matches": sum(row["full_match"] for row in subset),
            "multi_anchor_matches": sum(row["multi_anchor_match"] for row in subset),
        }
    summary = {
        "corpus_dir": str(args.corpus_dir),
        "corpus_files": len(corpus_files),
        "corpus_compact_characters": corpus_characters,
        "normalization": "NFKC + casefold + remove non-alphanumeric",
        "anchor_width": args.anchor_width,
        "labels_loaded": False,
        "public_test_answers_used": False,
        "splits": split_summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "matches.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in records
    ))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
