import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_task1_textual import (
    align_annotations, encode_release_entity, entity_type_f1,
    repair_embedded_label,
)


def test_textual_alignment_prefers_outer_longer_entity():
    text = "Η Ευρωπαϊκή Κεντρική Τράπεζα εξέδωσε ανακοίνωση."
    spans = [
        ("Ευρωπαϊκή", "ΤΟΠΟΘΕΣΙΑ"),
        ("Ευρωπαϊκή Κεντρική Τράπεζα", "ΟΡΓΑΝΙΣΜΟΣ"),
    ]
    assert align_annotations(text, spans) == [(2, 28, "ΟΡΓΑΝΙΣΜΟΣ")]


def test_textual_alignment_assigns_duplicate_mentions_once_each():
    text = "Η ΟΠΑΠ συνεργάζεται με την ΟΠΑΠ."
    spans = [("ΟΠΑΠ", "ΟΡΓΑΝΙΣΜΟΣ"), ("ΟΠΑΠ", "ΟΡΓΑΝΙΣΜΟΣ")]
    assert align_annotations(text, spans) == [
        (2, 6, "ΟΡΓΑΝΙΣΜΟΣ"), (27, 31, "ΟΡΓΑΝΙΣΜΟΣ"),
    ]


def test_released_embedded_person_label_corruption_is_repaired_for_alignment():
    assert repair_embedded_label("LYKOS PAΠΡΟΣΩΠΟLESS") == "LYKOS PAPERLESS"
    assert align_annotations(
        "Η LYKOS PAPERLESS SOLUTIONS A.E. συγχωνεύθηκε.",
        [("LYKOS PAΠΡΟΣΩΠΟLESS SOLUTIONS A.E.", "ΟΡΓΑΝΙΣΜΟΣ")],
    ) == [(2, 32, "ΟΡΓΑΝΙΣΜΟΣ")]


def test_uppercase_per_is_reencoded_only_for_organizations():
    assert encode_release_entity("SUPERVISOR Α.Ε.", "ΟΡΓΑΝΙΣΜΟΣ") == "SUΠΡΟΣΩΠΟVISOR Α.Ε."
    assert encode_release_entity("Grivalia Properties", "ΟΡΓΑΝΙΣΜΟΣ") == "Grivalia Properties"
    assert encode_release_entity("PERSON", "ΠΡΟΣΩΠΟ") == "PERSON"
    assert encode_release_entity("EUROBANK  ERGASIAS", "ΟΡΓΑΝΙΣΜΟΣ") == "EUROBANK ERGASIAS"


def test_entity_type_f1_is_computed_over_exact_typed_entities():
    gold = [
        [("Alice", "ΠΡΟΣΩΠΟ"), ("Athens", "ΤΟΠΟΘΕΣΙΑ")],
        [("Bank", "ΟΡΓΑΝΙΣΜΟΣ")],
    ]
    predicted = [
        [("Alice", "ΠΡΟΣΩΠΟ"), ("Greece", "ΤΟΠΟΘΕΣΙΑ")],
        [("Bank", "ΟΡΓΑΝΙΣΜΟΣ"), ("Fund", "ΟΡΓΑΝΙΣΜΟΣ")],
    ]
    assert entity_type_f1(gold, predicted) == {
        "ΠΡΟΣΩΠΟ": 1.0,
        "ΟΡΓΑΝΙΣΜΟΣ": 2 / 3,
        "ΤΟΠΟΘΕΣΙΑ": 0.0,
    }
