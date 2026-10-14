import pandas as pd
import pytest

from scripts.merge_task1_textual_predictions import merge
from scripts.prepare_task1_textual_subset import INDEX_COLUMN, prepare


NUMERIC_QUERY = "Αναγνώρισε αριθμητικές οντότητες και χρηματικά ποσά."
TEXTUAL_QUERY = "Αναγνώρισε ένα πρόσωπο, έναν οργανισμό και μία τοποθεσία."


def test_extract_and_merge_preserves_noncontiguous_submission_indices():
    source = pd.DataFrame({
        "query": [NUMERIC_QUERY, TEXTUAL_QUERY, NUMERIC_QUERY, TEXTUAL_QUERY],
        "text": ["n0", "t1", "n2", "t3"],
    })
    subset = prepare(source)
    assert subset[INDEX_COLUMN].tolist() == [1, 3]
    assert subset.text.tolist() == ["t1", "t3"]

    base = [
        {"row_index": index, "answer": f"base-{index}"}
        for index in range(4)
    ]
    textual = [
        {"row_index": 0, "answer": "entity-1, ΠΡΟΣΩΠΟ"},
        {"row_index": 1, "answer": "entity-3, ΟΡΓΑΝΙΣΜΟΣ"},
    ]
    output = merge(source, base, subset, textual)
    assert [row["answer"] for row in output] == [
        "base-0", "entity-1, ΠΡΟΣΩΠΟ", "base-2", "entity-3, ΟΡΓΑΝΙΣΜΟΣ",
    ]


def test_merge_rejects_reordered_textual_subset():
    source = pd.DataFrame({
        "query": [TEXTUAL_QUERY, NUMERIC_QUERY, TEXTUAL_QUERY],
        "text": ["t0", "n1", "t2"],
    })
    subset = prepare(source).iloc[::-1].reset_index(drop=True)
    base = [{"row_index": index, "answer": ""} for index in range(3)]
    textual = [{"row_index": index, "answer": ""} for index in range(2)]
    with pytest.raises(ValueError, match="does not exactly match"):
        merge(source, base, subset, textual)


def test_merge_rejects_local_prediction_misalignment():
    source = pd.DataFrame({"query": [TEXTUAL_QUERY], "text": ["t0"]})
    subset = prepare(source)
    with pytest.raises(ValueError, match="local position"):
        merge(
            source, [{"row_index": 0, "answer": ""}], subset,
            [{"row_index": 4, "answer": "entity, ΠΡΟΣΩΠΟ"}],
        )


def test_custom_submission_indices_are_preserved_as_json_native_scalars():
    source = pd.DataFrame({
        "row_index": [101, 205],
        "query": [TEXTUAL_QUERY, NUMERIC_QUERY],
        "text": ["t101", "n205"],
    })
    subset = prepare(source)
    output = merge(
        source,
        [
            {"row_index": 101, "answer": "old"},
            {"row_index": 205, "answer": "number, ΠΟΣΟΤΗΤΕΣ"},
        ],
        subset,
        [{"row_index": 0, "answer": "entity, ΠΡΟΣΩΠΟ"}],
    )
    assert output == [
        {"row_index": 101, "answer": "entity, ΠΡΟΣΩΠΟ"},
        {"row_index": 205, "answer": "number, ΠΟΣΟΤΗΤΕΣ"},
    ]
    assert all(type(row["row_index"]) is int for row in output)
