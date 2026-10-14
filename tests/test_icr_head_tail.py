import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import numpy as np
import pandas as pd

from train_icr_japanese_cv import build_oof_output, collate, encode_pair
from predict_icr_japanese_ensemble import (
    collate_queries, load_prediction_frame, uses_rinna_roberta_compat,
)


class ToyTokenizer:
    def __call__(self, first, second=None, **kwargs):
        def ids(value):
            return [int(token) for token in value.split()]
        if second is None:
            return {"input_ids": ids(first)}
        values = [101] + ids(first) + [102] + ids(second) + [102]
        return {"input_ids": values[:kwargs["max_length"]], "attention_mask": [1] * min(len(values), kwargs["max_length"])}

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2

    def prepare_for_model(self, first, second, **kwargs):
        values = [101] + list(first) + [102] + list(second) + [102]
        return {"input_ids": values, "attention_mask": [1] * len(values)}


def test_response_head_tail_preserves_both_ends():
    encoded = encode_pair(
        ToyTokenizer(), "1 2", "10 11 12 13 14 15 16 17 18 19",
        max_length=9, truncation_profile="response_head_tail",
    )
    assert encoded["input_ids"] == [101, 1, 2, 102, 10, 11, 18, 19, 102]


def test_response_head_tail_does_not_change_short_pair():
    encoded = encode_pair(
        ToyTokenizer(), "1 2", "10 11", max_length=9,
        truncation_profile="response_head_tail",
    )
    assert encoded["input_ids"] == [101, 1, 2, 102, 10, 11, 102]


class RinnaToyTokenizer:
    cls_token = "[CLS]"

    def __call__(self, first, second=None, **kwargs):
        assert first.startswith(self.cls_token)
        return {"input_ids": [4, 10, 2, 20, 2], "attention_mask": [1] * 5}

    def pad(self, features, return_tensors=None):
        return {
            key: torch.tensor([feature[key] for feature in features])
            for key in features[0]
        }


def test_rinna_compatibility_prepends_cls_and_uses_zero_based_positions():
    tokenizer = RinnaToyTokenizer()
    encoded = encode_pair(
        tokenizer, "question", "response", max_length=8, prepend_cls=True,
    )
    batch = collate(
        [encoded | {"labels": 3}], tokenizer, zero_based_position_ids=True,
    )
    assert batch["input_ids"].tolist() == [[4, 10, 2, 20, 2]]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 3, 4]]
    assert batch["labels"].tolist() == [3]


def test_rinna_inference_collation_and_saved_marker(tmp_path):
    tokenizer = RinnaToyTokenizer()
    batch = collate_queries(
        [{"input_ids": [4, 10, 2, 20, 2], "attention_mask": [1] * 5}],
        tokenizer,
        zero_based_position_ids=True,
    )
    assert batch["position_ids"].tolist() == [[0, 1, 2, 3, 4]]
    assert "labels" not in batch

    (tmp_path / "training.txt").write_text("rinna_roberta_compat=True\n")
    assert uses_rinna_roberta_compat(tmp_path)
    assert not uses_rinna_roberta_compat(tmp_path / "unmarked")


def test_partial_oof_output_contains_only_completed_rows():
    frame = pd.DataFrame({
        "id": [10, 11, 12],
        "answer": ["+1", "0", "-1"],
        "fold": [0, 1, 2],
    })
    logits = np.full((3, 5), np.nan, dtype=np.float32)
    logits[0] = [0, 0, 0, 2, 0]
    logits[2] = [0, 3, 0, 0, 0]

    output = build_oof_output(frame, logits)

    assert output.id.tolist() == [10, 12]
    assert output.gold.tolist() == ["+1", "-1"]
    assert output.prediction.tolist() == ["+1", "-1"]
    assert output.fold.tolist() == [0, 2]


def test_prediction_loader_excludes_answer_from_parquet_and_jsonl(tmp_path):
    frame = pd.DataFrame({
        "id": [17], "query": ["question"], "answer": ["hidden-label"],
    })
    parquet = tmp_path / "input.parquet"
    jsonl = tmp_path / "input.jsonl"
    frame.to_parquet(parquet, index=False)
    jsonl.write_text(frame.to_json(orient="records", lines=True) + "\n")

    for path in (parquet, jsonl):
        loaded = load_prediction_frame(path)
        assert loaded.columns.tolist() == ["query", "id"]
        assert loaded.to_dict("records") == [{"query": "question", "id": 17}]
