#!/usr/bin/env python3
"""Build leakage-safe entity-rich Greek financial MLM chunks for Task 1."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.task1_folds import (  # noqa: E402
    assign_entity_disjoint_folds,
    normalize_entity_surface,
)
from task1_numeric_baseline import parse_answer  # noqa: E402
from train_task1_textual import repair_embedded_label  # noqa: E402


GREEK_TYPES = {
    "person": "ΠΡΟΣΩΠΟ",
    "organization": "ΟΡΓΑΝΙΣΜΟΣ",
    "location": "ΤΟΠΟΘΕΣΙΑ",
}
REPORT_NAME = re.compile(
    r"οικονομικ[ήη]\s+(?:έκθεση|κατάσταση)\s+(.+?)\s+"
    r"\((?:19|20)\d{2}",
    re.IGNORECASE,
)
GENERIC_SURFACES = frozenset({
    "αε", "α ε", "a e", "ae", "εταιρεια", "ομιλος", "group", "bank",
    "ελλαδα", "ελλαδος", "αθηνα", "αθηνας", "κυπρος",
})


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def normalized_text(value: str) -> str:
    return " ".join(str(value).casefold().split())


def clean_surface(value: str) -> str:
    return " ".join(unquote(value).replace("_", " ").split()).strip(" .-")


def report_issuer_alias(url: str) -> str | None:
    filename = unquote(url.rsplit("/", 1)[-1])
    match = REPORT_NAME.search(filename)
    if match is None:
        return None
    value = clean_surface(match.group(1))
    return value if len(re.sub(r"\W", "", value, flags=re.UNICODE)) >= 3 else None


def document_issuer(path: Path) -> str | None:
    for part in reversed(path.parts):
        match = re.match(r"^(\d+)(?:_|$)", part)
        if match:
            return match.group(1)
    match = re.search(r"athex_issuer_(\d+)", str(path))
    return match.group(1) if match else None


def word_chunks(text: str, words_per_chunk: int = 180):
    words = list(re.finditer(r"\S+", text))
    for start in range(0, len(words), words_per_chunk):
        selected = words[start:start + words_per_chunk]
        if not selected:
            continue
        left, right = selected[0].start(), selected[-1].end()
        chunk = text[left:right]
        if len(chunk) >= 100:
            yield chunk


def compile_surface_pattern(surfaces: list[str]) -> re.Pattern | None:
    values = sorted(set(surfaces), key=lambda value: (-len(value), value.casefold()))
    if not values:
        return None
    return re.compile("|".join(re.escape(value) for value in values), re.IGNORECASE)


def exact_matches(
    text: str,
    pattern: re.Pattern | None,
    inventory: dict[str, tuple[str, str]],
    signal: str,
) -> list[dict]:
    if pattern is None:
        return []
    matches = []
    for match in pattern.finditer(text):
        key = clean_surface(match.group(0)).casefold()
        if key not in inventory:
            continue
        canonical, entity_type = inventory[key]
        matches.append({
            "start": match.start(), "end": match.end(),
            "text": match.group(0), "canonical": canonical,
            "type": entity_type, "signals": [signal],
        })
    return matches


def resolve_spans(spans: list[dict]) -> list[dict]:
    merged = {}
    for span in spans:
        key = (span["start"], span["end"], span["type"])
        if key not in merged:
            merged[key] = dict(span)
        else:
            merged[key]["signals"] = sorted(set(
                merged[key]["signals"] + span["signals"]
            ))
    accepted = []
    occupied = []
    for span in sorted(
        merged.values(), key=lambda row: (-(row["end"] - row["start"]), row["start"]),
    ):
        if any(span["start"] < right and span["end"] > left for left, right in occupied):
            continue
        accepted.append(span)
        occupied.append((span["start"], span["end"]))
    return sorted(accepted, key=lambda row: (row["start"], row["end"]))


def benchmark_overlap(text: str, released_texts: list[str]) -> bool:
    candidate = normalized_text(text)
    return any(
        len(reference) >= 100 and (reference in candidate or candidate in reference)
        for reference in released_texts
    )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--corpus-dir", type=Path,
        default=ROOT / "data/processed/finance_corpus_url_recovered_v2",
    )
    parser.add_argument(
        "--ledger", type=Path,
        default=ROOT / "data/raw/finance_corpus/athex_collection_progress.jsonl",
    )
    parser.add_argument(
        "--gliner-model", type=Path,
        default=ROOT / "artifacts/task1_gliner_deploy_seed7_width16/fold_0/final",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-chunks", type=int, default=4096)
    parser.add_argument("--min-documents", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task1_entity_aware_fold0_v1/silver",
    )
    args = parser.parse_args()
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --cpu for intentional CPU inference")

    train_path = ROOT / "data/external/plutus_finner_text/train-00000-of-00001.parquet"
    validation_path = ROOT / "data/external/plutus_finner_text/validation-00000-of-00001.parquet"
    train = pd.read_parquet(train_path)
    entity_rows = [parse_answer(answer) for answer in train.answer]
    folds = assign_entity_disjoint_folds(entity_rows, n_splits=5, seed=args.seed)
    if args.fold not in set(folds):
        raise ValueError("requested fold is unavailable")
    held_keys = {
        (kind, normalize_entity_surface(repair_embedded_label(surface)))
        for index, entities in enumerate(entity_rows) if folds[index] == args.fold
        for surface, kind in entities
    }

    train_candidates: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for index, entities in enumerate(entity_rows):
        if folds[index] == args.fold:
            continue
        for surface, kind in entities:
            surface = clean_surface(repair_embedded_label(surface))
            normalized = normalize_entity_surface(surface)
            if (
                len(normalized) < 3 or normalized in GENERIC_SURFACES
                or (kind, normalized) in held_keys
            ):
                continue
            train_candidates[surface.casefold()].add((surface, kind))
    train_inventory = {
        key: next(iter(values)) for key, values in train_candidates.items()
        if len({kind for _, kind in values}) == 1
    }
    train_pattern = compile_surface_pattern([value[0] for value in train_inventory.values()])

    aliases_by_issuer: dict[str, set[str]] = defaultdict(set)
    for line in args.ledger.read_text().splitlines():
        record = json.loads(line)
        alias = report_issuer_alias(str(record["url"]))
        if alias is None:
            continue
        if ("ΟΡΓΑΝΙΣΜΟΣ", normalize_entity_surface(alias)) in held_keys:
            continue
        aliases_by_issuer[str(record["issuer"])].add(alias)

    released_texts = [
        normalized_text(value)
        for value in pd.concat([
            pd.read_parquet(train_path, columns=["text"]),
            pd.read_parquet(validation_path, columns=["text"]),
        ]).text.astype(str)
    ]
    documents = sorted(args.corpus_dir.rglob("*.txt"))
    candidates = []
    surface_documents: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for path in documents:
        relative = str(path.relative_to(args.corpus_dir))
        issuer = document_issuer(path)
        official_inventory = {
            surface.casefold(): (surface, "ΟΡΓΑΝΙΣΜΟΣ")
            for surface in aliases_by_issuer.get(issuer or "", set())
        }
        official_pattern = compile_surface_pattern([
            value[0] for value in official_inventory.values()
        ])
        text = path.read_text(errors="ignore")
        for chunk_index, chunk in enumerate(word_chunks(text)):
            spans = exact_matches(
                chunk, train_pattern, train_inventory, "released_train_inventory",
            )
            spans += exact_matches(
                chunk, official_pattern, official_inventory, "official_issuer_metadata",
            )
            spans = resolve_spans(spans)
            if not spans or benchmark_overlap(chunk, released_texts):
                continue
            for span in spans:
                for signal in span["signals"]:
                    surface_documents[
                        (span["canonical"].casefold(), span["type"], signal)
                    ].add(relative)
            candidates.append({
                "document": relative, "issuer": issuer,
                "chunk_index": chunk_index, "text": chunk, "spans": spans,
            })

    repeated = set()
    for key, paths in surface_documents.items():
        if len(paths) >= args.min_documents:
            repeated.add(key)
    repeated_candidates = []
    for row in candidates:
        spans = []
        for span in row["spans"]:
            signals = [
                signal for signal in span["signals"]
                if (span["canonical"].casefold(), span["type"], signal) in repeated
            ]
            if signals:
                spans.append({**span, "signals": signals})
        if spans:
            repeated_candidates.append({**row, "spans": resolve_spans(spans)})

    repeated_candidates.sort(key=lambda row: hashlib.sha256(
        f"{row['document']}:{row['chunk_index']}:{row['text']}".encode()
    ).hexdigest())
    repeated_candidates = repeated_candidates[:args.max_chunks]

    from gliner import GLiNER

    model = GLiNER.from_pretrained(args.gliner_model, map_location="cpu")
    if not args.cpu:
        model = model.to("cuda")
    predictions = model.batch_predict_entities(
        [row["text"] for row in repeated_candidates], list(GREEK_TYPES),
        flat_ner=False, multi_label=False, threshold=args.threshold,
        batch_size=args.batch_size,
    )
    final_rows = []
    rejection_counts = Counter()
    for row, proposals in zip(repeated_candidates, predictions, strict=True):
        proposal_keys = {
            (int(item["start"]), int(item["end"]), GREEK_TYPES[str(item["label"])])
            for item in proposals if str(item["label"]) in GREEK_TYPES
        }
        spans = []
        for span in row["spans"]:
            official = "official_issuer_metadata" in span["signals"]
            gliner_agrees = (span["start"], span["end"], span["type"]) in proposal_keys
            if official or gliner_agrees:
                signals = list(span["signals"])
                if gliner_agrees:
                    signals.append("fold0_excluded_gliner_exact_type")
                spans.append({**span, "signals": sorted(set(signals))})
            else:
                rejection_counts["no_gliner_exact_type_agreement"] += 1
        spans = resolve_spans(spans)
        if spans:
            final_rows.append({**row, "spans": spans})
        else:
            rejection_counts["no_admitted_spans"] += 1

    span_count = sum(len(row["spans"]) for row in final_rows)
    if len(final_rows) < 512 or span_count < 1000:
        raise ValueError(
            f"silver corpus below frozen minimum: chunks={len(final_rows)} spans={span_count}"
        )
    for row in final_rows:
        for span in row["spans"]:
            if row["text"][span["start"]:span["end"]] != span["text"]:
                raise AssertionError("persisted silver span is not an exact substring")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = args.output_dir / "corpus.jsonl"
    corpus_path.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in final_rows
    ))
    corpus_files_hash = hashlib.sha256("\n".join(
        f"{path.relative_to(args.corpus_dir)} {digest(path)}" for path in documents
    ).encode()).hexdigest()
    summary = {
        "experiment": "task1_entity_aware_silver_fold0",
        "fold": args.fold, "seed": args.seed,
        "documents": len(documents), "candidate_chunks": len(candidates),
        "candidate_chunks_sent_to_gliner": len(repeated_candidates),
        "silver_chunks": len(final_rows), "silver_spans": span_count,
        "spans_by_type": dict(Counter(
            span["type"] for row in final_rows for span in row["spans"]
        )),
        "spans_by_signal": dict(Counter(
            signal for row in final_rows for span in row["spans"]
            for signal in span["signals"]
        )),
        "unique_surfaces": len({
            (span["canonical"].casefold(), span["type"])
            for row in final_rows for span in row["spans"]
        }),
        "held_surface_keys_excluded": len(held_keys),
        "train_inventory_surfaces": len(train_inventory),
        "official_issuer_aliases": sum(map(len, aliases_by_issuer.values())),
        "gliner_threshold": args.threshold,
        "rejections": dict(rejection_counts),
        "input_sha256": {
            "train": digest(train_path), "validation_packet": digest(validation_path),
            "ledger": digest(args.ledger), "corpus_file_set": corpus_files_hash,
            "gliner_model": digest(args.gliner_model / "pytorch_model.bin"),
        },
        "corpus_sha256": digest(corpus_path),
        "validation_answer_column_loaded": False,
        "held_labels_used_only_for_surface_exclusion_and_final_evaluation": True,
        "public_test_used": False, "hidden_test_used": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
