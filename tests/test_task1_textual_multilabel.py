import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_task1_textual_multilabel import align_overlapping


def test_multilabel_alignment_retains_nested_location_inside_organization():
    text = "Η Bank of Greece δημοσίευσε ανακοίνωση."
    assert align_overlapping(text, [
        ("Bank of Greece", "ΟΡΓΑΝΙΣΜΟΣ"), ("Greece", "ΤΟΠΟΘΕΣΙΑ"),
    ]) == [(2, 16, "ΟΡΓΑΝΙΣΜΟΣ"), (10, 16, "ΤΟΠΟΘΕΣΙΑ")]


def test_multilabel_alignment_assigns_repeated_same_type_mentions_once():
    text = "ΟΠΑΠ και ΟΠΑΠ"
    assert align_overlapping(text, [
        ("ΟΠΑΠ", "ΟΡΓΑΝΙΣΜΟΣ"), ("ΟΠΑΠ", "ΟΡΓΑΝΙΣΜΟΣ"),
    ]) == [(0, 4, "ΟΡΓΑΝΙΣΜΟΣ"), (9, 13, "ΟΡΓΑΝΙΣΜΟΣ")]


def test_multilabel_alignment_keeps_outer_span_for_same_type_nesting():
    text = "Η EPSILON SINGULARLOGIC Α.Ε. ανακοίνωσε αποτελέσματα."
    assert align_overlapping(text, [
        ("EPSILON SINGULARLOGIC Α.Ε.", "ΟΡΓΑΝΙΣΜΟΣ"),
        ("SINGULARLOGIC Α.Ε.", "ΟΡΓΑΝΙΣΜΟΣ"),
    ]) == [(2, 28, "ΟΡΓΑΝΙΣΜΟΣ")]
