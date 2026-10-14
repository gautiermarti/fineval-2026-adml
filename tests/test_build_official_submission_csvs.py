import json

import pandas as pd
import pytest

from scripts.build_official_submission_csvs import build_rows


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def test_task1_uses_official_ids_and_none(tmp_path):
    source = tmp_path / "input.parquet"
    predictions = tmp_path / "predictions.jsonl"
    pd.DataFrame({
        "id": [f"row-{index}" for index in range(200)],
        "query": ["q"] * 200,
    }).to_parquet(source)
    rows = [{"row_index": index, "answer": "None"} for index in range(200)]
    rows[1]["answer"] = "ACME, ΟΡΓΑΝΙΣΜΟΣ"
    write_jsonl(predictions, rows)
    output = build_rows(1, source, predictions)
    assert output[0] == {"id": "row-0", "prediction": "None"}
    assert output[1] == {"id": "row-1", "prediction": "ACME, ΟΡΓΑΝΙΣΜΟΣ"}


def test_task2_rejects_id_drift(tmp_path):
    source = tmp_path / "input.parquet"
    predictions = tmp_path / "predictions.jsonl"
    pd.DataFrame({"id": range(50), "query": ["q"] * 50}).to_parquet(source)
    rows = [{"id": index, "answer": "0"} for index in range(50)]
    rows[7]["id"] = 99
    write_jsonl(predictions, rows)
    with pytest.raises(ValueError, match="id mismatch at row 7"):
        build_rows(2, source, predictions)


def test_task2_accepts_private_only_packet(tmp_path):
    source = tmp_path / "input.parquet"
    predictions = tmp_path / "predictions.jsonl"
    pd.DataFrame({"id": range(35), "query": ["q"] * 35}).to_parquet(source)
    rows = [{"id": index, "answer": "0"} for index in range(35)]
    write_jsonl(predictions, rows)
    assert len(build_rows(2, source, predictions)) == 35


def test_rejects_input_prediction_length_mismatch(tmp_path):
    source = tmp_path / "input.parquet"
    predictions = tmp_path / "predictions.jsonl"
    pd.DataFrame({"id": range(2), "query": ["q"] * 2}).to_parquet(source)
    write_jsonl(predictions, [{"id": 0, "answer": "0"}])
    with pytest.raises(ValueError, match="row mismatch"):
        build_rows(2, source, predictions)


def test_task3_rejects_empty_answer(tmp_path):
    source = tmp_path / "input.parquet"
    predictions = tmp_path / "predictions.jsonl"
    pd.DataFrame({
        "id": [f"p{i}" for i in range(76)],
        "task_id": ["bundle"] * 76,
        "query": ["q"] * 76,
        "question": ["question"] * 76,
    }).to_parquet(source)
    rows = [{"task_id": "bundle", "answer": "ok"} for _ in range(76)]
    rows[3]["answer"] = ""
    write_jsonl(predictions, rows)
    with pytest.raises(ValueError, match="empty Task 3 answer at row 3"):
        build_rows(3, source, predictions)
