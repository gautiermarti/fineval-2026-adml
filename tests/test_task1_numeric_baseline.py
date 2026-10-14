import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from scripts.task1_numeric_baseline import predict


def test_numeric_units_are_not_in_entity_span():
    assert predict("επένδυση €5,8 εκ.") == [("5,8", "ΧΡΗΜΑΤΑ")]


def test_dates_are_component_entities():
    assert predict("στις 25.10.2016") == [
        ("25", "ΧΡΟΝΙΚΑ"), ("10", "ΧΡΟΝΙΚΑ"), ("2016", "ΧΡΟΝΙΚΑ")
    ]


def test_counts_and_years():
    assert predict("με άλλα 4 καταστήματα και 1 στην Κύπρο μέσα στο 2009") == [
        ("4", "ΠΟΣΟΤΗΤΕΣ"), ("1", "ΠΟΣΟΤΗΤΕΣ"), ("2009", "ΧΡΟΝΙΚΑ")
    ]
