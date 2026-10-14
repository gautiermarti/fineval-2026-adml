import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from predict_task1_lookup_safe import normalize, task_kind


def test_task1_prompt_router_distinguishes_numeric_and_textual():
    assert task_kind("αναγνώρισε αριθμητικές οντότητες και χρηματικά ποσά") == "numeric"
    assert task_kind("οντότητες: ένα πρόσωπο, έναν οργανισμό ή μία τοποθεσία") == "textual"


def test_task1_lookup_normalization_is_conservative():
    assert normalize("Αθήνα\n  Α.Ε.") == "Αθήνα Α.Ε."
    assert normalize("ΟΠΑΠ") != normalize("ΟΠΑΠ Α.Ε.")
