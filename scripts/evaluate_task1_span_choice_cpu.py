#!/usr/bin/env python3
"""Entity-linked CPU diagnostic for counterfactual Task 1 span choices."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

from task1_numeric_baseline import parse_answer  # noqa: E402
from train_task1_textual import encode_release_entity  # noqa: E402
from train_task1_textual_multilabel import align_overlapping  # noqa: E402


LEGAL_RE = re.compile(
    r"(?i)(?:Α\s*\.\s*Ε\s*\.?|A\s*\.\s*E\s*\.?|S\s*\.\s*A\s*\.?|"
    r"LTD\.?|LIMITED|PLC|INC\.?|CORP\.?|COMPANY|HOLDINGS?|BANK|ΤΡΑΠΕΖ)"
)
ARTICLE_RE = re.compile(r"(?i)^(?:η|ο|οι|the)\b")
LABEL_TO_TYPE = {
    "person": "ΠΡΟΣΩΠΟ",
    "organization": "ΟΡΓΑΝΙΣΜΟΣ",
    "location": "ΤΟΠΟΘΕΣΙΑ",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def group_pairs(pairs: list[dict]) -> list[dict]:
    grouped = {}
    for pair in pairs:
        preferred = pair["preferred"]
        key = (
            int(pair["row_index"]), int(pair["gold_index"]),
            int(preferred["start"]), int(preferred["end"]),
            str(preferred["entity_type"]),
        )
        if key not in grouped:
            grouped[key] = {
                "row_index": int(pair["row_index"]), "fold": int(pair["fold"]),
                "gold_index": int(pair["gold_index"]),
                "context_start": int(pair["context_start"]),
                "context": str(pair["context"]),
                "preferred": preferred,
                "rejected": {},
            }
        rejected = pair["rejected"]
        rejected_key = (
            int(rejected["start"]), int(rejected["end"]),
            str(rejected["entity_type"]),
        )
        grouped[key]["rejected"][rejected_key] = rejected
    return [
        record | {"rejected": list(record["rejected"].values())}
        for _, record in sorted(grouped.items())
    ]


def actual_gliner_groups(
    frame: pd.DataFrame, raw_predictions: list[dict], folds: np.ndarray,
) -> tuple[list[dict], dict]:
    """Build held-fold groups where GLiNER proposes the exact gold plus rivals."""
    predictions = {int(record["row_index"]): record for record in raw_predictions}
    groups = []
    aligned = recalled = with_rivals = 0
    for row_index, row in frame.iterrows():
        text = str(row.text)
        gold_items = set(parse_answer(row.answer))
        annotations = align_overlapping(text, list(gold_items))
        aligned += len(annotations)
        proposals = {}
        for raw in predictions[row_index]["entities"]:
            entity_type = LABEL_TO_TYPE[str(raw["label"])]
            start, end = int(raw["start"]), int(raw["end"])
            candidate = {
                "start": start, "end": end, "entity_type": entity_type,
                "surface": text[start:end], "proposal_score": float(raw["score"]),
            }
            key = start, end, entity_type
            if key not in proposals or candidate["proposal_score"] > proposals[key]["proposal_score"]:
                proposals[key] = candidate
        candidates = list(proposals.values())
        for gold_index, (start, end, entity_type) in enumerate(annotations):
            key = start, end, entity_type
            if key not in proposals:
                continue
            recalled += 1
            rejected = []
            for candidate in candidates:
                if not (
                    int(candidate["start"]) < end and int(candidate["end"]) > start
                ) or (
                    int(candidate["start"]), int(candidate["end"]),
                    str(candidate["entity_type"]),
                ) == key:
                    continue
                candidate_item = (
                    encode_release_entity(
                        str(candidate["surface"]), str(candidate["entity_type"]),
                    ),
                    str(candidate["entity_type"]),
                )
                if candidate_item not in gold_items:
                    rejected.append(candidate)
            if not rejected:
                continue
            with_rivals += 1
            all_candidates = [proposals[key], *rejected]
            context_start = max(0, min(int(item["start"]) for item in all_candidates) - 100)
            context_end = min(
                len(text), max(int(item["end"]) for item in all_candidates) + 100,
            )
            groups.append({
                "row_index": int(row_index), "fold": int(folds[row_index]),
                "gold_index": int(gold_index), "context_start": context_start,
                "context": text[context_start:context_end],
                "preferred": proposals[key], "rejected": rejected,
            })
    return groups, {
        "aligned_gold_occurrences": aligned,
        "exact_gold_recalled_by_gliner": recalled,
        "exact_recall": recalled / aligned,
        "recalled_gold_with_overlapping_nongold_rival": with_rivals,
    }


def intrinsic_features(candidate: dict, group: dict, candidates: list[dict]) -> dict:
    surface = str(candidate["surface"])
    letters = [character for character in surface if character.isalpha()]
    left, right = int(candidate["start"]), int(candidate["end"])
    return {
        "character_length": len(surface),
        "token_count": len(re.findall(r"\w+|[^\w\s]", surface)),
        "uppercase_ratio": sum(character.isupper() for character in letters) / max(1, len(letters)),
        "latin_ratio": sum("A" <= character.upper() <= "Z" for character in letters) / max(1, len(letters)),
        "has_digit": int(any(character.isdigit() for character in surface)),
        "has_legal_designator": int(bool(LEGAL_RE.search(surface))),
        "starts_article": int(bool(ARTICLE_RE.match(surface))),
        "starts_punctuation": int(bool(surface and not surface[0].isalnum())),
        "ends_punctuation": int(bool(surface and not surface[-1].isalnum())),
        "same_type_contains": sum(
            other is not candidate
            and other["entity_type"] == candidate["entity_type"]
            and left <= int(other["start"]) and right >= int(other["end"])
            for other in candidates
        ),
        "same_type_contained_by": sum(
            other is not candidate
            and other["entity_type"] == candidate["entity_type"]
            and int(other["start"]) <= left and int(other["end"]) >= right
            for other in candidates
        ),
        f"type={candidate['entity_type']}": 1,
    }


def candidate_text(candidate: dict, group: dict) -> str:
    context = group["context"]
    relative_left = int(candidate["start"]) - int(group["context_start"])
    relative_right = int(candidate["end"]) - int(group["context_start"])
    left = context[max(0, relative_left - 80):relative_left]
    right = context[relative_right:relative_right + 80]
    return (
        f"{candidate['entity_type']} [ENTITY] {candidate['surface']} "
        f"[LEFT] {left} [RIGHT] {right}"
    )


def flatten(groups: list[dict]) -> list[dict]:
    row_group_counts = Counter(group["row_index"] for group in groups)
    output = []
    for group_index, group in enumerate(groups):
        candidates = [group["preferred"], *group["rejected"]]
        group_weight = 1.0 / row_group_counts[group["row_index"]]
        for candidate_index, candidate in enumerate(candidates):
            is_gold = int(candidate_index == 0)
            weight = (
                0.5 * group_weight if is_gold
                else 0.5 * group_weight / len(group["rejected"])
            )
            output.append({
                "group_index": group_index, "is_gold": is_gold, "weight": weight,
                "text": candidate_text(candidate, group),
                "features": intrinsic_features(candidate, group, candidates),
                "candidate": candidate,
            })
    return output


def fit_model(records: list[dict]):
    text_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=2,
        max_features=50_000, sublinear_tf=True,
    )
    dict_vectorizer = DictVectorizer()
    scaler = StandardScaler(with_mean=False)
    text = text_vectorizer.fit_transform([record["text"] for record in records])
    numeric = scaler.fit_transform(dict_vectorizer.fit_transform([
        record["features"] for record in records
    ]))
    matrix = hstack([text, numeric], format="csr")
    model = LogisticRegression(
        C=0.5, max_iter=2000, solver="liblinear", random_state=7,
    ).fit(
        matrix, [record["is_gold"] for record in records],
        sample_weight=[record["weight"] for record in records],
    )
    return text_vectorizer, dict_vectorizer, scaler, model


def score_records(records: list[dict], components) -> np.ndarray:
    text_vectorizer, dict_vectorizer, scaler, model = components
    text = text_vectorizer.transform([record["text"] for record in records])
    numeric = scaler.transform(dict_vectorizer.transform([
        record["features"] for record in records
    ]))
    return model.predict_proba(hstack([text, numeric], format="csr"))[:, 1]


def ranking_metrics(
    groups: list[dict], records: list[dict], probabilities: np.ndarray,
) -> tuple[dict, list[dict]]:
    by_group = defaultdict(list)
    for record, probability in zip(records, probabilities, strict=True):
        by_group[record["group_index"]].append((record, float(probability)))
    pair_correct = pair_count = top1 = reciprocal_rank = 0.0
    longest_top1 = shortest_top1 = 0
    proposal_pair_correct = proposal_top1 = 0.0
    has_proposal_scores = all(
        "proposal_score" in group["preferred"]
        for group in groups
    )
    predictions = []
    for group_index, group in enumerate(groups):
        scored = by_group[group_index]
        gold_score = next(score for record, score in scored if record["is_gold"])
        negative_scores = [score for record, score in scored if not record["is_gold"]]
        pair_correct += sum(gold_score > score for score in negative_scores)
        pair_count += len(negative_scores)
        ranking = sorted(scored, key=lambda item: (-item[1], -item[0]["is_gold"]))
        rank = next(
            index + 1 for index, (record, _) in enumerate(ranking)
            if record["is_gold"]
        )
        top1 += int(rank == 1)
        reciprocal_rank += 1.0 / rank
        candidates = [group["preferred"], *group["rejected"]]
        gold_geometry = (
            group["preferred"]["start"], group["preferred"]["end"],
            group["preferred"]["entity_type"],
        )
        longest = max(candidates, key=lambda row: (len(row["surface"]), row["surface"]))
        shortest = min(candidates, key=lambda row: (len(row["surface"]), row["surface"]))
        longest_top1 += int((longest["start"], longest["end"], longest["entity_type"]) == gold_geometry)
        shortest_top1 += int((shortest["start"], shortest["end"], shortest["entity_type"]) == gold_geometry)
        if has_proposal_scores:
            gold_proposal_score = float(group["preferred"]["proposal_score"])
            rival_proposal_scores = [float(item["proposal_score"]) for item in group["rejected"]]
            proposal_pair_correct += sum(
                gold_proposal_score > value for value in rival_proposal_scores
            )
            proposal_top1 += int(gold_proposal_score > max(rival_proposal_scores))
        predictions.append({
            "row_index": group["row_index"], "gold_index": group["gold_index"],
            "fold": group["fold"], "gold_rank": rank, "gold_score": gold_score,
            "candidate_count": len(scored),
            "best_candidate": ranking[0][0]["candidate"],
            "best_score": ranking[0][1],
        })
    metrics = {
        "groups": len(groups), "pairwise_choices": int(pair_count),
        "pairwise_accuracy": pair_correct / pair_count,
        "gold_top1_accuracy": top1 / len(groups),
        "mean_reciprocal_rank": reciprocal_rank / len(groups),
        "longest_top1_accuracy": longest_top1 / len(groups),
        "shortest_top1_accuracy": shortest_top1 / len(groups),
    }
    if has_proposal_scores:
        metrics["gliner_score_pairwise_accuracy"] = proposal_pair_correct / pair_count
        metrics["gliner_score_gold_top1_accuracy"] = proposal_top1 / len(groups)
    return metrics, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs", type=Path,
        default=ROOT / "artifacts/task1_span_surgery_v1/span_choice_pairs.jsonl",
    )
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "data/external/plutus_finner_text/train-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--gliner-raw", type=Path,
        default=ROOT / "artifacts/task1_gliner_oof_seed7_width16_ensemble/oof_raw_predictions.jsonl",
    )
    parser.add_argument(
        "--folds", type=Path,
        default=ROOT / "artifacts/task1_gliner_oof_seed7_width16_ensemble/folds.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/task1_span_choice_cpu_v1",
    )
    args = parser.parse_args()

    groups = group_pairs(load_jsonl(args.pairs))
    frame = pd.read_parquet(args.input).reset_index(drop=True)
    fold_frame = pd.read_csv(args.folds).sort_values("row_index")
    if fold_frame.row_index.astype(int).tolist() != list(range(len(frame))):
        raise ValueError("fold file is not aligned to Task 1 training rows")
    actual_groups, actual_coverage = actual_gliner_groups(
        frame, load_jsonl(args.gliner_raw), fold_frame.fold.astype(int).to_numpy(),
    )
    all_predictions = []
    actual_predictions = []
    fold_summaries = {}
    actual_fold_summaries = {}
    for fold in range(5):
        train_groups = [group for group in groups if group["fold"] != fold]
        valid_groups = [group for group in groups if group["fold"] == fold]
        train_records = flatten(train_groups)
        valid_records = flatten(valid_groups)
        components = fit_model(train_records)
        probabilities = score_records(valid_records, components)
        fold_summaries[str(fold)], predictions = ranking_metrics(
            valid_groups, valid_records, probabilities,
        )
        all_predictions.extend(predictions)

        held_actual_groups = [group for group in actual_groups if group["fold"] == fold]
        held_actual_records = flatten(held_actual_groups)
        held_actual_probabilities = score_records(held_actual_records, components)
        actual_fold_summaries[str(fold)], predictions = ranking_metrics(
            held_actual_groups, held_actual_records, held_actual_probabilities,
        )
        actual_predictions.extend(predictions)

    total_groups = sum(item["groups"] for item in fold_summaries.values())
    summary = {
        "evaluation": "five entity-linked held folds",
        "span_choice_groups": total_groups,
        "pairwise_choices": sum(len(group["rejected"]) for group in groups),
        "weighted_pairwise_accuracy": sum(
            item["pairwise_accuracy"] * item["pairwise_choices"]
            for item in fold_summaries.values()
        ) / sum(item["pairwise_choices"] for item in fold_summaries.values()),
        "weighted_gold_top1_accuracy": sum(
            item["gold_top1_accuracy"] * item["groups"]
            for item in fold_summaries.values()
        ) / total_groups,
        "weighted_mean_reciprocal_rank": sum(
            item["mean_reciprocal_rank"] * item["groups"]
            for item in fold_summaries.values()
        ) / total_groups,
        "folds": fold_summaries,
        "actual_gliner_transfer": {
            **actual_coverage,
            "evaluable_groups": sum(
                item["groups"] for item in actual_fold_summaries.values()
            ),
            "pairwise_accuracy": sum(
                item["pairwise_accuracy"] * item["pairwise_choices"]
                for item in actual_fold_summaries.values()
            ) / sum(item["pairwise_choices"] for item in actual_fold_summaries.values()),
            "gold_top1_accuracy": sum(
                item["gold_top1_accuracy"] * item["groups"]
                for item in actual_fold_summaries.values()
            ) / sum(item["groups"] for item in actual_fold_summaries.values()),
            "gliner_score_gold_top1_accuracy": sum(
                item["gliner_score_gold_top1_accuracy"] * item["groups"]
                for item in actual_fold_summaries.values()
            ) / sum(item["groups"] for item in actual_fold_summaries.values()),
            "folds": actual_fold_summaries,
        },
        "features": "candidate surface/context plus intrinsic span geometry; no surgery label",
        "surgery_training_uses_incumbent_or_gliner_predictions": False,
        "actual_transfer_evaluation_uses_oof_gliner_candidates": True,
        "end_to_end_ner_claim": False,
        "validation_or_public_test_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in all_predictions
    ))
    (args.output_dir / "actual_gliner_predictions.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in actual_predictions
    ))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
