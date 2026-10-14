#!/usr/bin/env python3
"""Learn a train-only policy for adding GLiNER proposals to Task 1.

The candidate classifier and its precision gate are selected exclusively from
row-held-out predictions on the 320-row training split.  The 80-row validation
split is opened once afterward to report whether conservative additions help.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from task1_numeric_baseline import parse_answer, score  # noqa: E402
from train_task1_textual import encode_release_entity  # noqa: E402


LABEL_TO_TYPE = {
    "person": "ΠΡΟΣΩΠΟ",
    "organization": "ΟΡΓΑΝΙΣΜΟΣ",
    "location": "ΤΟΠΟΘΕΣΙΑ",
}
LEGAL_RE = re.compile(
    r"(?i)(?:\bA\.?E\.?\b|\bS\.?A\.?\b|\bLTD\b|\bPLC\b|\bBANK\b|"
    r"\bCOMPANY\b|\bHOLDINGS?\b|Α\.?\u0395\.?|ΤΡΑΠΕΖ)"
)
GENERIC_RE = re.compile(
    r"(?i)(?:η |ο |του |της )?(?:εταιρ(?:εία|ία)|όμιλος|ομίλου|τράπεζα|"
    r"διοικητικό συμβούλιο|δ\.?\u03c3\.?|επιτροπή(?:ς)? ελέγχου)"
)


def load_raw(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def make_candidates(frame: pd.DataFrame, raw_records):
    candidates = []
    for record, row in zip(raw_records, frame.itertuples()):
        row_index = int(record["row_index"])
        text = str(row.text)
        gold = set(parse_answer(row.answer))
        entities = []
        for value in record["entities"]:
            kind = LABEL_TO_TYPE[str(value["label"])]
            surface = encode_release_entity(str(value["text"]), kind)
            entities.append({
                "start": int(value["start"]), "end": int(value["end"]),
                "type": kind, "entity": surface, "gliner_score": float(value["score"]),
            })
        for entity in entities:
            start, end = entity["start"], entity["end"]
            same_overlaps = sum(
                other["type"] == entity["type"]
                and (other["start"], other["end"]) != (start, end)
                and other["start"] < end and other["end"] > start
                for other in entities
            )
            contains = sum(
                (other["start"], other["end"]) != (start, end)
                and other["start"] >= start and other["end"] <= end
                for other in entities
            )
            contained = sum(
                (other["start"], other["end"]) != (start, end)
                and other["start"] <= start and other["end"] >= end
                for other in entities
            )
            surface = entity["entity"]
            letters = [char for char in surface if char.isalpha()]
            features = {
                "score": entity["gliner_score"],
                "score_squared": entity["gliner_score"] ** 2,
                "characters": len(surface),
                "words": len(surface.split()),
                "same_type_overlaps": same_overlaps,
                "contains_candidates": contains,
                "contained_by_candidates": contained,
                f"type={entity['type']}": 1,
                "has_digit": int(any(char.isdigit() for char in surface)),
                "has_parenthesis": int("(" in surface or ")" in surface),
                "uppercase_ratio": sum(char.isupper() for char in letters) / max(1, len(letters)),
                "latin_ratio": sum("A" <= char.upper() <= "Z" for char in letters) / max(1, len(letters)),
                "legal_suffix": int(bool(LEGAL_RE.search(surface))),
                "generic_surface": int(bool(GENERIC_RE.fullmatch(surface.strip()))),
            }
            left, right = text[max(0, start - 60):start], text[end:end + 60]
            candidate = dict(entity)
            candidate.update({
                "row_index": row_index,
                "features": features,
                "text_features": (
                    f"{entity['type']} [ENTITY] {surface} [LEFT] {left} [RIGHT] {right}"
                ),
                "correct": int((surface, entity["type"]) in gold),
            })
            candidates.append(candidate)
    return candidates


def fit_components(candidates, indices, *, max_features: int):
    text_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=2,
        max_features=max_features, sublinear_tf=True,
    )
    numeric_vectorizer = DictVectorizer()
    scaler = StandardScaler(with_mean=False)
    text_matrix = text_vectorizer.fit_transform(
        [candidates[index]["text_features"] for index in indices]
    )
    numeric_matrix = scaler.fit_transform(numeric_vectorizer.fit_transform(
        [candidates[index]["features"] for index in indices]
    ))
    matrix = hstack([text_matrix, numeric_matrix], format="csr")
    classifier = LogisticRegression(
        C=1.0, max_iter=1000, solver="liblinear", random_state=7,
    ).fit(matrix, [candidates[index]["correct"] for index in indices])
    return text_vectorizer, numeric_vectorizer, scaler, classifier


def transform(candidates, indices, components):
    text_vectorizer, numeric_vectorizer, scaler, _ = components
    text_matrix = text_vectorizer.transform(
        [candidates[index]["text_features"] for index in indices]
    )
    numeric_matrix = scaler.transform(numeric_vectorizer.transform(
        [candidates[index]["features"] for index in indices]
    ))
    return hstack([text_matrix, numeric_matrix], format="csr")


def wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    p = successes / total
    denominator = 1 + z * z / total
    center = p + z * z / (2 * total)
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (center - radius) / denominator


def select_gate(candidates, probabilities, *, minimum_lcb: float, minimum_count: int):
    labels = np.asarray([candidate["correct"] for candidate in candidates])
    frontier = []
    for threshold in np.linspace(0.50, 0.99, 99):
        selected = probabilities >= threshold
        total = int(selected.sum())
        successes = int(labels[selected].sum())
        frontier.append({
            "threshold": float(threshold), "selected": total, "correct": successes,
            "precision": successes / total if total else 0.0,
            "wilson_precision_lcb": wilson_lower(successes, total),
        })
    admissible = [
        item for item in frontier
        if item["selected"] >= minimum_count
        and item["wilson_precision_lcb"] >= minimum_lcb
    ]
    if not admissible:
        return 1.0, frontier
    chosen = max(admissible, key=lambda item: (item["selected"], -item["threshold"]))
    return chosen["threshold"], frontier


def load_incumbent(paths: list[Path], row_count: int):
    result = [set() for _ in range(row_count)]
    for path in paths:
        records = load_raw(path)
        if len(records) != row_count:
            raise ValueError(f"{path}: row count mismatch")
        for index, record in enumerate(records):
            result[int(record.get("row_index", index))].update(
                parse_answer(record.get("answer", ""))
            )
    return result


def locate_surfaces(text: str, entities):
    spans = []
    for surface, kind in entities:
        for match in re.finditer(re.escape(surface), text):
            spans.append((match.start(), match.end(), kind))
    return spans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-raw", type=Path,
                        default=ROOT / "artifacts/task1_gliner_proposals_train/raw_predictions.jsonl")
    parser.add_argument("--validation-raw", type=Path,
                        default=ROOT / "artifacts/task1_gliner_proposals/raw_predictions.jsonl")
    parser.add_argument("--minimum-precision-lcb", type=float, default=0.80)
    parser.add_argument("--minimum-oof-count", type=int, default=20)
    parser.add_argument("--max-features", type=int, default=15000)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/task1_annotation_policy_v1")
    parser.add_argument("--incumbent", type=Path, nargs="+", default=[
        ROOT / "artifacts/task1_textual/validation_flat_ensemble_3model_release_encoded.jsonl",
        ROOT / "artifacts/task1_textual/validation_nested_release_encoded.jsonl",
    ])
    args = parser.parse_args()

    train_frame = pd.read_parquet(
        ROOT / "data/external/plutus_finner_text/train-00000-of-00001.parquet"
    )
    validation_frame = pd.read_parquet(
        ROOT / "data/external/plutus_finner_text/validation-00000-of-00001.parquet"
    )
    train_candidates = make_candidates(train_frame, load_raw(args.train_raw))
    validation_candidates = make_candidates(validation_frame, load_raw(args.validation_raw))

    oof = np.zeros(len(train_candidates), dtype=float)
    groups = np.asarray([candidate["row_index"] for candidate in train_candidates])
    splitter = GroupKFold(n_splits=5)
    all_indices = np.arange(len(train_candidates))
    for fit_indices, held_indices in splitter.split(all_indices, groups=groups):
        components = fit_components(
            train_candidates, fit_indices, max_features=args.max_features,
        )
        oof[held_indices] = components[-1].predict_proba(
            transform(train_candidates, held_indices, components)
        )[:, 1]
    gate, frontier = select_gate(
        train_candidates, oof, minimum_lcb=args.minimum_precision_lcb,
        minimum_count=args.minimum_oof_count,
    )

    components = fit_components(
        train_candidates, all_indices, max_features=args.max_features,
    )
    validation_indices = np.arange(len(validation_candidates))
    validation_probabilities = components[-1].predict_proba(
        transform(validation_candidates, validation_indices, components)
    )[:, 1]
    incumbent = load_incumbent(args.incumbent, len(validation_frame))
    selected = [set(row) for row in incumbent]
    selected_spans = [
        locate_surfaces(str(row.text), entities)
        for row, entities in zip(validation_frame.itertuples(), selected)
    ]
    additions = []
    for probability, candidate in sorted(
        zip(validation_probabilities, validation_candidates), reverse=True,
        key=lambda item: item[0],
    ):
        if probability < gate:
            continue
        row_index = candidate["row_index"]
        item = (candidate["entity"], candidate["type"])
        if item in selected[row_index]:
            continue
        if any(
            kind == candidate["type"] and left < candidate["end"] and right > candidate["start"]
            for left, right, kind in selected_spans[row_index]
        ):
            continue
        selected[row_index].add(item)
        selected_spans[row_index].append(
            (candidate["start"], candidate["end"], candidate["type"])
        )
        additions.append({
            "row_index": row_index, "entity": candidate["entity"],
            "type": candidate["type"], "policy_probability": float(probability),
            "gliner_score": candidate["gliner_score"],
            "validation_correct": bool(candidate["correct"]),
        })

    gold = [set(parse_answer(answer)) for answer in validation_frame["answer"]]
    base_scores = np.asarray([
        score(list(target), list(prediction))[2]
        for target, prediction in zip(gold, incumbent)
    ])
    policy_scores = np.asarray([
        score(list(target), list(prediction))[2]
        for target, prediction in zip(gold, selected)
    ])
    deltas = policy_scores - base_scores
    rng = np.random.default_rng(7)
    bootstrap = deltas[rng.integers(0, len(deltas), size=(10000, len(deltas)))].mean(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions.jsonl").write_text("".join(
        json.dumps({
            "row_index": index,
            "answer": "\n".join(f"{entity}, {kind}" for entity, kind in sorted(row)),
        }, ensure_ascii=False) + "\n"
        for index, row in enumerate(selected)
    ))
    pd.DataFrame(frontier).to_csv(args.output_dir / "oof_precision_frontier.csv", index=False)
    (args.output_dir / "additions.jsonl").write_text("".join(
        json.dumps(item, ensure_ascii=False) + "\n" for item in additions
    ))
    summary = {
        "gate_selected_from_train_oof_only": gate,
        "minimum_train_oof_precision_lcb": args.minimum_precision_lcb,
        "minimum_train_oof_count": args.minimum_oof_count,
        "train_candidates": len(train_candidates),
        "validation_candidates": len(validation_candidates),
        "validation_additions": len(additions),
        "validation_correct_additions": sum(item["validation_correct"] for item in additions),
        "validation_incorrect_additions": sum(not item["validation_correct"] for item in additions),
        "incumbent_macro_entity_f1": float(base_scores.mean()),
        "policy_macro_entity_f1": float(policy_scores.mean()),
        "delta": float(deltas.mean()),
        "rows_improved": int((deltas > 0).sum()),
        "rows_harmed": int((deltas < 0).sum()),
        "bootstrap_delta_p05": float(np.quantile(bootstrap, 0.05)),
        "bootstrap_delta_p50": float(np.quantile(bootstrap, 0.50)),
        "bootstrap_delta_p95": float(np.quantile(bootstrap, 0.95)),
        "bootstrap_probability_gain": float((bootstrap > 0).mean()),
        "public_test_used": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for item in additions:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
