import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.normalize_task1_submission import normalize_records


NUMERIC_QUERY = "Αναγνώρισε αριθμητικές οντότητες και χρηματικά ποσά."
TEXTUAL_QUERY = "Αναγνώρισε κάθε πρόσωπο, οργανισμό και τοποθεσία."


def test_normalize_records_changes_only_numeric_calendar_suffix():
    source = [
        {"query": NUMERIC_QUERY, "text": "Την 8ης Ιουνίου 2023."},
        {"query": TEXTUAL_QUERY, "text": "Η ACME βρίσκεται στην Αθήνα."},
    ]
    predictions = [
        {
            "row_index": 0,
            "answer": "8ης, ΧΡΟΝΙΚΑ\n8, ΧΡΟΝΙΚΑ\n2023, ΧΡΟΝΙΚΑ",
        },
        {"row_index": 1, "answer": "ACME, ΟΡΓΑΝΙΣΜΟΣ"},
    ]
    output, changes = normalize_records(source, predictions)
    assert output == [
        {"row_index": 0, "answer": "8, ΧΡΟΝΙΚΑ\n2023, ΧΡΟΝΙΚΑ"},
        {"row_index": 1, "answer": "ACME, ΟΡΓΑΝΙΣΜΟΣ"},
    ]
    assert len(changes) == 1
    assert changes[0]["removed"] == [["8ης", "ΧΡΟΝΙΚΑ"]]


def test_normalize_records_is_idempotent():
    source = [{"query": NUMERIC_QUERY, "text": "Την 8ης Ιουνίου 2023."}]
    predictions = [{
        "row_index": 0, "answer": "8, ΧΡΟΝΙΚΑ\n2023, ΧΡΟΝΙΚΑ",
    }]
    output, changes = normalize_records(source, predictions)
    assert output == predictions
    assert changes == []

