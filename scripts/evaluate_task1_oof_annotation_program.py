#!/usr/bin/env python3
"""Learn add/replace/nesting decisions on genuine fold-pure disagreements.

GLiNER and the nested-BERT incumbent must both be out-of-fold on the official
training rows.  The confidence gate is selected from nested entity-disjoint
OOF predictions, then frozen before the 80-row validation split is evaluated.
The public test is never read.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.task1_annotation_ops import (  # noqa: E402
    apply_operation,
    apply_ranked_operations,
    candidate_operations,
)
from task1_numeric_baseline import parse_answer, score  # noqa: E402
from train_task1_textual import (  # noqa: E402
    ENTITY_TYPES,
    encode_release_entity,
    repair_embedded_label,
)


LABEL_TO_TYPE = {
    "person": "ΠΡΟΣΩΠΟ",
    "organization": "ΟΡΓΑΝΙΣΜΟΣ",
    "location": "ΤΟΠΟΘΕΣΙΑ",
}
LEGAL_RE = re.compile(
    r"(?i)(?:\bA\.?E\.?\b|\bS\.?A\.?\b|\bLTD\b|\bPLC\b|\bBANK\b|"
    r"\bCOMPANY\b|\bHOLDINGS?\b|Α\.?Ε\.?|ΤΡΑΠΕΖ|ΑΝΩΝΥΜ)"
)
LEADING_ARTICLE_RE = re.compile(r"(?i)^(?:η|ο|οι|the)\s+")


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def gliner_candidates(frame: pd.DataFrame, path: Path):
    records = load_jsonl(path)
    if len(records) != len(frame):
        raise ValueError(f"{path}: expected {len(frame)} rows, found {len(records)}")
    by_row = [[] for _ in range(len(frame))]
    for fallback_index, record in enumerate(records):
        row_index = int(record.get("row_index", fallback_index))
        text = str(frame.iloc[row_index].text)
        deduplicated = {}
        for raw in record["entities"]:
            kind = LABEL_TO_TYPE[str(raw["label"])]
            start, end = int(raw["start"]), int(raw["end"])
            raw_surface = text[start:end]
            candidate = {
                "row_index": row_index, "start": start, "end": end,
                "type": kind, "surface_text": raw_surface,
                "entity": encode_release_entity(raw_surface, kind),
                "score": float(raw["score"]),
            }
            key = start, end, kind
            if key not in deduplicated or candidate["score"] > deduplicated[key]["score"]:
                deduplicated[key] = candidate
        by_row[row_index] = list(deduplicated.values())
    return by_row


def load_oof_incumbent(path: Path, row_count: int):
    records = load_jsonl(path)
    if len(records) != row_count:
        raise ValueError(f"{path}: expected {row_count} rows, found {len(records)}")
    items, occurrences = [set() for _ in range(row_count)], [[] for _ in range(row_count)]
    for fallback_index, record in enumerate(records):
        row_index = int(record.get("row_index", fallback_index))
        for entity in record["entities"]:
            item = str(entity["entity"]), str(entity["type"])
            items[row_index].add(item)
            occurrences[row_index].append({
                "entity": item[0], "type": item[1],
                "surface_text": str(entity.get("surface_text", entity["entity"])),
                "start": int(entity["start"]), "end": int(entity["end"]),
                "confidence": float(entity.get("confidence", 1.0)),
            })
    return items, occurrences


def load_validation_incumbent(frame: pd.DataFrame, paths: list[Path]):
    items = [set() for _ in range(len(frame))]
    for path in paths:
        records = load_jsonl(path)
        if len(records) != len(frame):
            raise ValueError(f"{path}: validation row count mismatch")
        for fallback_index, record in enumerate(records):
            row_index = int(record.get("row_index", fallback_index))
            items[row_index].update(parse_answer(record.get("answer", "")))
    occurrences = [[] for _ in range(len(frame))]
    for row_index, row_items in enumerate(items):
        text = str(frame.iloc[row_index].text)
        for entity, kind in row_items:
            forms = tuple(dict.fromkeys((entity, repair_embedded_label(entity))))
            seen_spans = set()
            for form in forms:
                for match in re.finditer(re.escape(form), text):
                    if match.span() in seen_spans:
                        continue
                    seen_spans.add(match.span())
                    occurrences[row_index].append({
                        "entity": entity, "type": kind, "surface_text": form,
                        "start": match.start(), "end": match.end(), "confidence": 1.0,
                    })
    return items, occurrences


def make_operations(frame, gold, candidates_by_row, incumbent, occurrences, folds=None):
    # Raw span scores can shift across independently trained fold models.
    # Convert them to input-only fold/type percentiles and within-row margins;
    # the same calibration is recomputed on validation/test without labels.
    score_buckets = {}
    for row_index, candidates in enumerate(candidates_by_row):
        fold = -1 if folds is None else int(folds[row_index])
        for candidate in candidates:
            score_buckets.setdefault((fold, candidate["type"]), []).append(
                float(candidate["score"])
            )
    for values in score_buckets.values():
        values.sort()

    operations = []
    for row_index, candidates in enumerate(candidates_by_row):
        text = str(frame.iloc[row_index].text)
        fold = -1 if folds is None else int(folds[row_index])
        row_top_score = max((candidate["score"] for candidate in candidates), default=0.0)
        row_type_top = {
            kind: max(
                (candidate["score"] for candidate in candidates if candidate["type"] == kind),
                default=0.0,
            )
            for kind in ENTITY_TYPES
        }
        baseline_f1 = (
            score(list(gold[row_index]), list(incumbent[row_index]))[2]
            if gold is not None else 0.0
        )
        confidence_by_item = {}
        for occurrence in occurrences[row_index]:
            item = occurrence["entity"], occurrence["type"]
            confidence_by_item[item] = max(
                confidence_by_item.get(item, 0.0), float(occurrence.get("confidence", 1.0))
            )
        for candidate in candidates:
            variants = candidate_operations(
                candidate, occurrences[row_index], incumbent[row_index],
            )
            surface = candidate["entity"]
            letters = [character for character in surface if character.isalpha()]
            left = text[max(0, candidate["start"] - 80):candidate["start"]]
            right = text[candidate["end"]:candidate["end"] + 80]
            for operation in variants:
                changed = apply_operation(incumbent[row_index], operation)
                changed_f1 = (
                    score(list(gold[row_index]), list(changed))[2]
                    if gold is not None else 0.0
                )
                removed_confidences = [
                    confidence_by_item.get(tuple(item), 0.0) for item in operation["remove"]
                ]
                bucket = score_buckets[(fold, candidate["type"])]
                score_percentile = bisect_right(bucket, candidate["score"]) / len(bucket)
                features = {
                    "gliner_score": candidate["score"],
                    "gliner_score_squared": candidate["score"] ** 2,
                    "gliner_fold_type_percentile": score_percentile,
                    "margin_from_row_top": row_top_score - candidate["score"],
                    "margin_from_row_type_top": (
                        row_type_top[candidate["type"]] - candidate["score"]
                    ),
                    "characters": len(surface),
                    "words": len(surface.split()),
                    "incumbent_entities": len(incumbent[row_index]),
                    "removed_entities": len(operation["remove"]),
                    "removed_confidence_max": max(removed_confidences, default=0.0),
                    "removed_confidence_mean": (
                        sum(removed_confidences) / len(removed_confidences)
                        if removed_confidences else 0.0
                    ),
                    "same_type_overlap_count": operation["same_type_overlap_count"],
                    "cross_type_overlap_count": operation["cross_type_overlap_count"],
                    "candidate_contains_count": operation["candidate_contains_count"],
                    "candidate_contained_by_count": operation["candidate_contained_by_count"],
                    "has_digit": int(any(character.isdigit() for character in surface)),
                    "has_parenthesis": int("(" in surface or ")" in surface),
                    "legal_designator": int(bool(LEGAL_RE.search(surface))),
                    "leading_article": int(bool(LEADING_ARTICLE_RE.match(surface))),
                    "uppercase_ratio": (
                        sum(character.isupper() for character in letters) / max(1, len(letters))
                    ),
                    "latin_ratio": (
                        sum("A" <= character.upper() <= "Z" for character in letters)
                        / max(1, len(letters))
                    ),
                    f"type={candidate['type']}": 1,
                    f"operation={operation['operation']}": 1,
                }
                operations.append(operation | {
                    "fold": None if folds is None else int(folds[row_index]),
                    "features": features,
                    "text_features": (
                        f"{candidate['type']} [OP] {operation['operation']} "
                        f"[ENTITY] {surface} [LEFT] {left} [RIGHT] {right}"
                    ),
                    "target_delta": changed_f1 - baseline_f1,
                    "beneficial": int(
                        gold is not None and changed_f1 > baseline_f1 + 1e-12
                    ),
                })
    return operations


def fit_components(operations, indices, max_features):
    text_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=2,
        max_features=max_features, sublinear_tf=True,
    )
    numeric_vectorizer = DictVectorizer()
    scaler = StandardScaler(with_mean=False)
    text_matrix = text_vectorizer.fit_transform(
        [operations[index]["text_features"] for index in indices]
    )
    numeric_matrix = scaler.fit_transform(numeric_vectorizer.fit_transform(
        [operations[index]["features"] for index in indices]
    ))
    matrix = hstack([text_matrix, numeric_matrix], format="csr")
    labels = [operations[index]["beneficial"] for index in indices]
    if len(set(labels)) < 2:
        raise ValueError("operation training subset has only one class")
    classifier = LogisticRegression(
        C=0.5, max_iter=1500, solver="liblinear", random_state=7,
    ).fit(matrix, labels)
    return text_vectorizer, numeric_vectorizer, scaler, classifier


def transform(operations, indices, components):
    text_vectorizer, numeric_vectorizer, scaler, _ = components
    text_matrix = text_vectorizer.transform(
        [operations[index]["text_features"] for index in indices]
    )
    numeric_matrix = scaler.transform(numeric_vectorizer.transform(
        [operations[index]["features"] for index in indices]
    ))
    return hstack([text_matrix, numeric_matrix], format="csr")


def type_f1(gold, predicted):
    output = {}
    for kind in ENTITY_TYPES:
        true_positive = predicted_count = gold_count = 0
        for target, guess in zip(gold, predicted):
            target_type = {item for item in target if item[1] == kind}
            guess_type = {item for item in guess if item[1] == kind}
            true_positive += len(target_type & guess_type)
            predicted_count += len(guess_type)
            gold_count += len(target_type)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / gold_count if gold_count else 0.0
        output[kind] = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return output


def evaluate_policy(gold, baselines, operations, probabilities, threshold, folds, seed=7):
    selected, accepted = apply_ranked_operations(
        baselines, operations, probabilities, threshold,
    )
    baseline_scores = np.asarray([
        score(list(target), list(prediction))[2]
        for target, prediction in zip(gold, baselines)
    ])
    selected_scores = np.asarray([
        score(list(target), list(prediction))[2]
        for target, prediction in zip(gold, selected)
    ])
    deltas = selected_scores - baseline_scores
    rng = np.random.default_rng(seed)
    bootstrap = deltas[
        rng.integers(0, len(deltas), size=(10000, len(deltas)))
    ].mean(axis=1)
    base_type = type_f1(gold, baselines)
    selected_type = type_f1(gold, selected)
    type_deltas = {kind: selected_type[kind] - base_type[kind] for kind in ENTITY_TYPES}
    fold_deltas = {}
    if folds is not None:
        fold_array = np.asarray(folds)
        for fold in sorted(set(map(int, folds))):
            fold_deltas[str(fold)] = float(deltas[fold_array == fold].mean())
    return selected, accepted, {
        "threshold": float(threshold),
        "selected_operations": len(accepted),
        "baseline_macro_entity_f1": float(baseline_scores.mean()),
        "policy_macro_entity_f1": float(selected_scores.mean()),
        "delta": float(deltas.mean()),
        "rows_improved": int((deltas > 0).sum()),
        "rows_harmed": int((deltas < 0).sum()),
        "fold_deltas": fold_deltas,
        "improved_folds": sum(value > 1e-12 for value in fold_deltas.values()),
        "regressed_folds": sum(value < -1e-12 for value in fold_deltas.values()),
        "type_deltas": type_deltas,
        "bootstrap_p05": float(np.quantile(bootstrap, 0.05)),
        "bootstrap_p50": float(np.quantile(bootstrap, 0.50)),
        "bootstrap_p95": float(np.quantile(bootstrap, 0.95)),
        "bootstrap_probability_gain": float((bootstrap > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-gliner-raw", type=Path,
        default=ROOT / "artifacts/task1_gliner_oof_seed7_width16_ensemble/oof_raw_predictions.jsonl",
    )
    parser.add_argument(
        "--train-incumbent", type=Path,
        default=ROOT / "artifacts/task1_multilabel_oof_legal_seed7/oof_predictions.jsonl",
    )
    parser.add_argument(
        "--folds", type=Path,
        default=ROOT / "artifacts/task1_gliner_oof_seed7_width16_ensemble/folds.csv",
    )
    parser.add_argument(
        "--validation-gliner-raw", type=Path,
        default=ROOT / "artifacts/task1_gliner_finetuned_seed7_cpu80/validation_raw.jsonl",
    )
    parser.add_argument("--validation-incumbent", type=Path, nargs="+", default=[
        ROOT / "artifacts/task1_textual/validation_flat_ensemble_3model_release_encoded.jsonl",
        ROOT / "artifacts/task1_textual/validation_nested_release_encoded.jsonl",
    ])
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--minimum-oof-operations", type=int, default=3)
    parser.add_argument("--minimum-oof-probability-gain", type=float, default=0.90)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/task1_oof_annotation_program_seed7")
    args = parser.parse_args()

    train = pd.read_parquet(
        ROOT / "data/external/plutus_finner_text/train-00000-of-00001.parquet"
    )
    validation = pd.read_parquet(
        ROOT / "data/external/plutus_finner_text/validation-00000-of-00001.parquet"
    )
    folds_frame = pd.read_csv(args.folds).sort_values("row_index")
    folds = folds_frame["fold"].astype(int).tolist()
    if folds_frame["row_index"].astype(int).tolist() != list(range(len(train))):
        raise ValueError("fold file does not cover training rows in order")

    train_gold = [set(parse_answer(answer)) for answer in train["answer"]]
    validation_gold = [set(parse_answer(answer)) for answer in validation["answer"]]
    train_candidates = gliner_candidates(train, args.train_gliner_raw)
    validation_candidates = gliner_candidates(validation, args.validation_gliner_raw)
    train_incumbent, train_occurrences = load_oof_incumbent(
        args.train_incumbent, len(train),
    )
    validation_incumbent, validation_occurrences = load_validation_incumbent(
        validation, args.validation_incumbent,
    )
    train_operations = make_operations(
        train, train_gold, train_candidates, train_incumbent, train_occurrences, folds,
    )
    validation_operations = make_operations(
        validation, validation_gold, validation_candidates,
        validation_incumbent, validation_occurrences,
    )
    if not train_operations:
        raise RuntimeError("no training disagreements were generated")

    oof_probabilities = np.zeros(len(train_operations), dtype=float)
    all_indices = np.arange(len(train_operations))
    operation_folds = np.asarray([operation["fold"] for operation in train_operations])
    for fold in sorted(set(folds)):
        fit_indices = all_indices[operation_folds != fold]
        held_indices = all_indices[operation_folds == fold]
        components = fit_components(train_operations, fit_indices, args.max_features)
        oof_probabilities[held_indices] = components[-1].predict_proba(
            transform(train_operations, held_indices, components)
        )[:, 1]

    frontier = []
    # Beneficial boundary repairs are a minority, so an unweighted logistic
    # model may be conservatively calibrated below 0.5.  Sweep broadly; the
    # nested OOF robustness gates, not an arbitrary 0.5 cutoff, control risk.
    for threshold in np.linspace(0.05, 0.99, 95):
        _, _, metrics = evaluate_policy(
            train_gold, train_incumbent, train_operations,
            oof_probabilities, float(threshold), folds,
        )
        frontier.append(metrics)
    admissible = [
        metrics for metrics in frontier
        if metrics["selected_operations"] >= args.minimum_oof_operations
        and metrics["delta"] > 0
        and metrics["bootstrap_probability_gain"] >= args.minimum_oof_probability_gain
        and metrics["improved_folds"] >= 3
        and metrics["regressed_folds"] <= 1
        and min(metrics["type_deltas"].values()) >= -0.01
    ]
    chosen = max(
        admissible,
        key=lambda metrics: (
            metrics["delta"], metrics["bootstrap_p05"], metrics["threshold"],
        ),
        default=None,
    )
    gate = chosen["threshold"] if chosen is not None else 1.01

    final_components = fit_components(train_operations, all_indices, args.max_features)
    validation_indices = np.arange(len(validation_operations))
    validation_probabilities = final_components[-1].predict_proba(
        transform(validation_operations, validation_indices, final_components)
    )[:, 1] if len(validation_operations) else np.asarray([])
    predictions, accepted, validation_metrics = evaluate_policy(
        validation_gold, validation_incumbent, validation_operations,
        validation_probabilities, gate, None,
    )
    validation_metrics["single_run_promotion_gate"] = bool(
        chosen is not None
        and validation_metrics["delta"] > 0
        and validation_metrics["bootstrap_probability_gain"] >= 0.90
        and min(validation_metrics["type_deltas"].values()) >= -0.01
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frontier).to_json(
        args.output_dir / "oof_gate_frontier.jsonl", orient="records", lines=True,
        force_ascii=False,
    )
    (args.output_dir / "validation_predictions.jsonl").write_text("".join(
        json.dumps({
            "row_index": row_index,
            "answer": "\n".join(
                f"{entity}, {kind}" for entity, kind in sorted(row)
            ),
        }, ensure_ascii=False) + "\n"
        for row_index, row in enumerate(predictions)
    ))
    (args.output_dir / "validation_accepted_operations.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in accepted
    ))
    summary = {
        "train_rows": len(train),
        "train_operations": len(train_operations),
        "train_beneficial_operations": sum(
            operation["beneficial"] for operation in train_operations
        ),
        "validation_rows": len(validation),
        "validation_operations": len(validation_operations),
        "oof_gate_status": "accepted" if chosen is not None else "abstain",
        "chosen_oof_gate": chosen,
        "validation": validation_metrics,
        "two_seed_confirmation_required": True,
        "public_test_used": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
