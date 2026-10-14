#!/usr/bin/env python3
"""Build label-free NLI and symbolic grounding features for Task 3 candidates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.polyfiqa_edits import sentences, split_answer  # noqa: E402
from fineval.polyfiqa_entailment import (  # noqa: E402
    clean_claim,
    evidence_quotes,
    fact_clauses,
    numeric_support,
    quotation_support,
)


SOURCE_TYPES = ("fact", "statement", "news")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def encode(
    texts: list[str], tokenizer, model, *, batch_size: int, max_length: int,
) -> np.ndarray:
    output = []
    for start in range(0, len(texts), batch_size):
        features = tokenizer(
            texts[start:start + batch_size], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        with torch.inference_mode():
            hidden = model(**features).last_hidden_state
            mask = features["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            output.append(F.normalize(pooled, p=2, dim=1).cpu().numpy())
    return np.concatenate(output, axis=0)


def nli_probabilities(
    pairs: list[tuple[str, str]], tokenizer, model, *, batch_size: int,
    max_length: int,
) -> np.ndarray:
    output = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        features = tokenizer(
            [pair[0] for pair in batch], [pair[1] for pair in batch],
            padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            output.append(model(**features).logits.float().softmax(-1).cpu().numpy())
    return np.concatenate(output, axis=0)


def top_sources(
    claim: str, sources: dict[str, list[str]], vectors: dict[str, np.ndarray],
    *, statement_k: int, news_k: int,
) -> list[tuple[str, str]]:
    claim_vector = vectors[claim]
    selected = [("fact", text) for text in sources["fact"]]
    for source_type, limit in (("statement", statement_k), ("news", news_k)):
        ranked = sorted(
            sources[source_type],
            key=lambda text: float(vectors[text] @ claim_vector), reverse=True,
        )
        selected.extend((source_type, text) for text in ranked[:limit])
    return list(dict.fromkeys(selected))


def aggregate_claim_scores(
    claims: list[str], selected_by_claim: dict[str, list[tuple[str, str]]],
    pair_scores: dict[tuple[str, str], dict[str, float]],
) -> dict[str, float]:
    all_claim = []
    by_type = defaultdict(list)
    for claim in claims:
        scores = [
            (source_type, pair_scores[(premise, claim)])
            for source_type, premise in selected_by_claim[claim]
        ]
        entail = max((item[1]["entailment"] for item in scores), default=0.0)
        contradiction = max(
            (item[1]["contradiction"] for item in scores), default=0.0,
        )
        # The best signed source is more meaningful than independently taking
        # the largest entailment and contradiction from unrelated premises.
        margin = max(
            (
                item[1]["entailment"] - item[1]["contradiction"]
                for item in scores
            ),
            default=-1.0,
        )
        all_claim.append((entail, contradiction, margin))
        for source_type in SOURCE_TYPES:
            typed = [item[1] for item in scores if item[0] == source_type]
            by_type[source_type].append((
                max((item["entailment"] for item in typed), default=0.0),
                max((item["contradiction"] for item in typed), default=0.0),
            ))

    entailments = np.asarray([row[0] for row in all_claim], dtype=float)
    contradictions = np.asarray([row[1] for row in all_claim], dtype=float)
    margins = np.asarray([row[2] for row in all_claim], dtype=float)
    result = {
        "nli_entail_mean": float(entailments.mean()),
        "nli_entail_min": float(entailments.min()),
        "nli_entail_max": float(entailments.max()),
        "nli_contradiction_mean": float(contradictions.mean()),
        "nli_contradiction_max": float(contradictions.max()),
        "nli_margin_mean": float(margins.mean()),
        "nli_margin_min": float(margins.min()),
        "nli_strong_entail_rate": float((entailments >= 0.50).mean()),
        "nli_strong_contradiction_rate": float((contradictions >= 0.50).mean()),
    }
    for source_type in SOURCE_TYPES:
        typed = np.asarray(by_type[source_type], dtype=float)
        result[f"{source_type}_entail_mean"] = float(typed[:, 0].mean())
        result[f"{source_type}_entail_min"] = float(typed[:, 0].min())
        result[f"{source_type}_contradiction_max"] = float(typed[:, 1].max())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lattice", type=Path,
        default=ROOT / "artifacts/polyfiqa_sparse_edit_oracle_v1/lattice.jsonl",
    )
    parser.add_argument(
        "--facts", type=Path,
        default=ROOT / "artifacts/polyfiqa_fact_graphs/facts.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_entailment_features_v1",
    )
    parser.add_argument(
        "--retriever-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument(
        "--nli-model", default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    )
    parser.add_argument("--statement-k", type=int, default=3)
    parser.add_argument("--news-k", type=int, default=6)
    parser.add_argument("--retriever-batch-size", type=int, default=32)
    parser.add_argument("--nli-batch-size", type=int, default=16)
    parser.add_argument("--retriever-max-length", type=int, default=256)
    parser.add_argument("--nli-max-length", type=int, default=384)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    diagnostic = load_jsonl(args.lattice)
    # Reference text and candidate ROUGE scores intentionally remain behind the
    # projection boundary below.  They are available only to the later evaluator.
    rows = [{
        key: row[key] for key in (
            "row_index", "task_id", "internal_id", "tier", "family", "candidates",
        )
    } for row in diagnostic]
    facts = {int(row["row_index"]): row for row in load_jsonl(args.facts)}

    row_data = []
    embedding_texts = set()
    for row in rows:
        graph = facts[int(row["row_index"])]
        sources = {
            "fact": fact_clauses(graph),
            "statement": [
                item["text"] for item in graph.get("evidence", [])
                if item.get("source") == "financial_statements" and item.get("text")
            ],
            "news": [
                item["text"] for item in graph.get("evidence", [])
                if item.get("source") == "news" and item.get("text")
            ],
        }
        candidates = []
        for operation, answer in row["candidates"].items():
            core, evidence = split_answer(answer)
            claims = [clean_claim(item) for item in (sentences(core) or [core])]
            claims = [item for item in claims if item]
            candidates.append({
                "operation": operation, "answer": answer, "core": core,
                "evidence": evidence, "claims": claims,
            })
            embedding_texts.update(claims)
        for values in sources.values():
            embedding_texts.update(values)
        row_data.append({
            **{key: row[key] for key in (
                "row_index", "task_id", "internal_id", "tier", "family",
            )},
            "graph": graph, "sources": sources, "candidates": candidates,
        })

    ordered_texts = sorted(embedding_texts)
    retriever_tokenizer = AutoTokenizer.from_pretrained(
        args.retriever_model, local_files_only=True,
    )
    retriever = AutoModel.from_pretrained(
        args.retriever_model, local_files_only=True,
    ).eval().cpu()
    matrix = encode(
        ordered_texts, retriever_tokenizer, retriever,
        batch_size=args.retriever_batch_size, max_length=args.retriever_max_length,
    )
    vectors = dict(zip(ordered_texts, matrix, strict=True))
    del retriever, retriever_tokenizer, matrix

    selected_by_row_claim = {}
    pair_types = defaultdict(set)
    for row in row_data:
        unique_claims = sorted({
            claim for candidate in row["candidates"] for claim in candidate["claims"]
        })
        for claim in unique_claims:
            selected = top_sources(
                claim, row["sources"], vectors,
                statement_k=args.statement_k, news_k=args.news_k,
            )
            selected_by_row_claim[(row["row_index"], claim)] = selected
            for source_type, premise in selected:
                pair_types[(premise, claim)].add(source_type)

    ordered_pairs = sorted(pair_types)
    nli_tokenizer = AutoTokenizer.from_pretrained(args.nli_model, local_files_only=True)
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        args.nli_model, local_files_only=True,
    ).eval().cpu()
    probabilities = nli_probabilities(
        ordered_pairs, nli_tokenizer, nli_model,
        batch_size=args.nli_batch_size, max_length=args.nli_max_length,
    )
    labels = {str(value).casefold(): int(key) for key, value in nli_model.config.id2label.items()}
    required = {"entailment", "neutral", "contradiction"}
    if not required <= set(labels):
        raise ValueError(f"unexpected NLI label map: {nli_model.config.id2label}")
    pair_scores = {
        pair: {label: float(probabilities[index, labels[label]]) for label in required}
        for index, pair in enumerate(ordered_pairs)
    }

    feature_names = [
        "nli_entail_mean", "nli_entail_min", "nli_entail_max",
        "nli_contradiction_mean", "nli_contradiction_max",
        "nli_margin_mean", "nli_margin_min", "nli_strong_entail_rate",
        "nli_strong_contradiction_rate",
        *[
            f"{source_type}_{suffix}"
            for source_type in SOURCE_TYPES
            for suffix in ("entail_mean", "entail_min", "contradiction_max")
        ],
        "quote_exact_support", "quote_token_support", "quote_count",
        "has_evidence", "core_numeric_support", "core_numeric_supported",
        "core_numeric_unsupported", "answer_numeric_support",
        "answer_numeric_supported", "answer_numeric_unsupported",
        "claim_count", "answer_words", "core_words", "evidence_words",
    ]
    metadata = []
    features = []
    for row in row_data:
        immutable_sources = [
            item["text"] for item in row["graph"].get("evidence", [])
            if item.get("text")
        ]
        selected = {
            claim: selected_by_row_claim[(row["row_index"], claim)]
            for candidate in row["candidates"] for claim in candidate["claims"]
        }
        for candidate in row["candidates"]:
            nli = aggregate_claim_scores(candidate["claims"], selected, pair_scores)
            quote_exact, quote_tokens = quotation_support(
                candidate["evidence"], immutable_sources,
            )
            core_numeric = numeric_support(candidate["core"], row["graph"])
            answer_numeric = numeric_support(candidate["answer"], row["graph"])
            values = {
                **nli,
                "quote_exact_support": quote_exact,
                "quote_token_support": quote_tokens,
                "quote_count": float(len(evidence_quotes(candidate["evidence"]))),
                "has_evidence": float(bool(candidate["evidence"])),
                "core_numeric_support": core_numeric[0],
                "core_numeric_supported": float(core_numeric[1]),
                "core_numeric_unsupported": float(core_numeric[2]),
                "answer_numeric_support": answer_numeric[0],
                "answer_numeric_supported": float(answer_numeric[1]),
                "answer_numeric_unsupported": float(answer_numeric[2]),
                "claim_count": float(len(candidate["claims"])),
                "answer_words": len(candidate["answer"].split()) / 100.0,
                "core_words": len(candidate["core"].split()) / 100.0,
                "evidence_words": len(candidate["evidence"].split()) / 100.0,
            }
            features.append([values[name] for name in feature_names])
            metadata.append({
                **{key: row[key] for key in (
                    "row_index", "task_id", "internal_id", "tier", "family",
                )},
                "operation": candidate["operation"], "answer": candidate["answer"],
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "features.npy", np.asarray(features, dtype=np.float32))
    (args.output_dir / "metadata.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in metadata
    ))
    manifest = {
        "rows": len(row_data), "candidate_examples": len(metadata),
        "unique_embedding_texts": len(ordered_texts),
        "unique_nli_pairs": len(ordered_pairs),
        "retriever_model": args.retriever_model, "nli_model": args.nli_model,
        "device": "cpu", "threads": args.threads,
        "statement_k": args.statement_k, "news_k": args.news_k,
        "feature_names": feature_names, "feature_dimensions": len(feature_names),
        "nli_direction": "input evidence premise -> candidate answer clause hypothesis",
        "symbolic_signals": "exact quotation provenance and unit-aware numeric support",
        "reference_field_accessed": False, "score_field_accessed": False,
        "current_or_hidden_test_used": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
