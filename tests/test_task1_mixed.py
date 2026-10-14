import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from predict_task1_mixed import add_nested_locations


def test_nested_multilabel_adds_location_inside_flat_organization():
    flat = [("Bank of Greece", "ΟΡΓΑΝΙΣΜΟΣ")]
    multi = [("Bank of Greece", "ΟΡΓΑΝΙΣΜΟΣ"), ("Greece", "ΤΟΠΟΘΕΣΙΑ")]
    assert add_nested_locations(flat, multi) == [
        ("Bank of Greece", "ΟΡΓΑΝΙΣΜΟΣ"), ("Greece", "ΤΟΠΟΘΕΣΙΑ"),
    ]


def test_nested_multilabel_does_not_add_unanchored_location():
    flat = [("Alpha Bank", "ΟΡΓΑΝΙΣΜΟΣ")]
    multi = [("Αθήνα", "ΤΟΠΟΘΕΣΙΑ")]
    assert add_nested_locations(flat, multi) == flat
