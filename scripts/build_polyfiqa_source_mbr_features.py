#!/usr/bin/env python3
"""Build label-free semantic source-MBR features for sparse Task 3 edits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.polyfiqa_edits import sentences, split_answer  # noqa: E402
from fineval.polyfiqa_source_mbr import (  # noqa: E402
    bidirectional_support, fact_verbalization, request_description,
)


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
        default=ROOT / "artifacts/polyfiqa_source_mbr_features_v1",
    )
    parser.add_argument(
        "--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    # `reference` and `scores` coexist in the diagnostic lattice but are never
    # accessed here. Only immutable candidate text and input-derived metadata
    # are copied into the label-free working representation.
    diagnostic = load_jsonl(args.lattice)
    rows = [{
        key: row[key] for key in (
            "row_index", "task_id", "internal_id", "tier", "family", "candidates",
        )
    } for row in diagnostic]
    facts = {int(row["row_index"]): row for row in load_jsonl(args.facts)}

    row_data = []
    all_texts: set[str] = set()
    for row in rows:
        graph = facts[int(row["row_index"])]
        question = graph["question"]
        request = request_description(row["family"])
        fact_text = fact_verbalization(graph)
        source_texts = [fact_text] + [
            item["text"] for item in graph.get("evidence", []) if item.get("text")
        ]
        candidate_data = []
        for operation, answer in row["candidates"].items():
            core, evidence = split_answer(answer)
            parts = sentences(core) or [core]
            if evidence:
                parts.append(evidence)
            item = {
                "operation": operation, "answer": answer, "core": core,
                "evidence": evidence, "parts": parts,
            }
            candidate_data.append(item)
            all_texts.update([answer, core, *parts])
            if evidence:
                all_texts.add(evidence)
        all_texts.update([question, request, fact_text, *source_texts])
        row_data.append({
            **{key: row[key] for key in (
                "row_index", "task_id", "internal_id", "tier", "family",
            )},
            "question": question, "request": request, "fact_text": fact_text,
            "source_texts": source_texts, "graph": graph,
            "candidates": candidate_data,
        })

    ordered_texts = sorted(all_texts)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True).eval().cpu()
    matrix = encode(
        ordered_texts, tokenizer, model, batch_size=args.batch_size,
        max_length=args.max_length,
    )
    vectors = dict(zip(ordered_texts, matrix, strict=True))

    metadata = []
    dense_features = []
    feature_names = [
        "core_question", "answer_question", "core_request", "answer_request",
        "core_facts", "answer_facts", "evidence_question", "evidence_facts",
        "source_precision", "source_recall", "source_f1",
        "statement_support", "news_support", "answer_consensus",
        "core_consensus", "baseline_similarity", "core_baseline_similarity",
        "candidate_parts", "candidate_words", "core_words", "evidence_words",
        "fixed_source_mbr",
    ]
    for row in row_data:
        question_vector = vectors[row["question"]]
        request_vector = vectors[row["request"]]
        fact_vector = vectors[row["fact_text"]]
        graph = row["graph"]
        # Retain all exact statement lines plus the five news sentences most
        # related to the question. This is a deterministic quasi-source view.
        statements = [
            item["text"] for item in graph.get("evidence", [])
            if item.get("source") == "financial_statements"
        ]
        news = [
            item["text"] for item in graph.get("evidence", [])
            if item.get("source") == "news"
        ]
        news = sorted(
            news, key=lambda text: float(vectors[text] @ question_vector), reverse=True,
        )[:5]
        relevant_sources = list(dict.fromkeys([row["fact_text"], *statements, *news]))
        source_matrix = np.stack([vectors[text] for text in relevant_sources])
        statement_matrix = (
            np.stack([vectors[text] for text in statements]) if statements
            else np.empty((0, len(question_vector)))
        )
        news_matrix = (
            np.stack([vectors[text] for text in news]) if news
            else np.empty((0, len(question_vector)))
        )
        answer_matrix = np.stack([vectors[item["answer"]] for item in row["candidates"]])
        core_matrix = np.stack([vectors[item["core"]] for item in row["candidates"]])
        baseline_index = next(
            index for index, item in enumerate(row["candidates"])
            if item["operation"] == "baseline"
        )
        baseline_vector = answer_matrix[baseline_index]
        baseline_core_vector = core_matrix[baseline_index]
        answer_consensus = answer_matrix @ answer_matrix.T
        core_consensus = core_matrix @ core_matrix.T
        for index, item in enumerate(row["candidates"]):
            answer_vector = answer_matrix[index]
            core_vector = core_matrix[index]
            evidence_vector = vectors[item["evidence"]] if item["evidence"] else np.zeros_like(core_vector)
            part_matrix = np.stack([vectors[text] for text in item["parts"]])
            precision, recall, support_f1 = bidirectional_support(
                part_matrix, source_matrix,
            )
            statement_support = (
                float((statement_matrix @ evidence_vector).max())
                if item["evidence"] and len(statement_matrix) else 0.0
            )
            news_support = (
                float((news_matrix @ evidence_vector).max())
                if item["evidence"] and len(news_matrix) else 0.0
            )
            consensus = float((answer_consensus[index].sum() - 1) / max(len(answer_matrix) - 1, 1))
            core_medoid = float((core_consensus[index].sum() - 1) / max(len(core_matrix) - 1, 1))
            core_question = float(core_vector @ question_vector)
            answer_question = float(answer_vector @ question_vector)
            core_request = float(core_vector @ request_vector)
            answer_request = float(answer_vector @ request_vector)
            core_facts = float(core_vector @ fact_vector)
            answer_facts = float(answer_vector @ fact_vector)
            evidence_question = float(evidence_vector @ question_vector) if item["evidence"] else 0.0
            evidence_facts = float(evidence_vector @ fact_vector) if item["evidence"] else 0.0
            fixed = (
                0.20 * core_question + 0.20 * core_request
                + 0.15 * answer_facts + 0.25 * support_f1
                + 0.10 * consensus + 0.05 * statement_support
                + 0.05 * news_support
            )
            scalar = [
                core_question, answer_question, core_request, answer_request,
                core_facts, answer_facts, evidence_question, evidence_facts,
                precision, recall, support_f1, statement_support, news_support,
                consensus, core_medoid, float(answer_vector @ baseline_vector),
                float(core_vector @ baseline_core_vector), float(len(item["parts"])),
                len(item["answer"].split()) / 100.0,
                len(item["core"].split()) / 100.0,
                len(item["evidence"].split()) / 100.0, fixed,
            ]
            # Difference embeddings let the nested verifier learn semantic edit
            # direction while remaining invariant to the target company's name.
            dense_features.append(np.concatenate([
                np.asarray(scalar, dtype=np.float32),
                answer_vector - baseline_vector,
                core_vector - baseline_core_vector,
            ]))
            metadata.append({
                "row_index": int(row["row_index"]), "task_id": row["task_id"],
                "internal_id": row["internal_id"], "tier": row["tier"],
                "family": row["family"], "operation": item["operation"],
                "answer": item["answer"], "fixed_source_mbr": fixed,
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "features.npy", np.stack(dense_features).astype(np.float32))
    (args.output_dir / "metadata.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in metadata
    ))
    manifest = {
        "rows": len(row_data), "candidate_examples": len(metadata),
        "model": args.model, "device": "cpu", "max_length": args.max_length,
        "scalar_features": feature_names,
        "embedding_features": "answer-minus-baseline and core-minus-baseline",
        "feature_dimensions": int(len(dense_features[0])),
        "quasi_sources": "question, fixed requested-slot description, typed facts, exact statement lines, top-5 question-related news sentences",
        "reference_field_accessed": False, "score_field_accessed": False,
        "current_or_hidden_test_used": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
