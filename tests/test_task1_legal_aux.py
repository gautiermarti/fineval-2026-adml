import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pretrain_task1_textual_legal_ner import mapped_tag


def test_legal_auxiliary_label_mapping():
    assert mapped_tag("B-ORG") == "B-ΟΡΓΑΝΙΣΜΟΣ"
    assert mapped_tag("I-PERSON") == "I-ΠΡΟΣΩΠΟ"
    assert mapped_tag("B-GPE") == "B-ΤΟΠΟΘΕΣΙΑ"
    assert mapped_tag("I-LOCATION-NAT") == "I-ΤΟΠΟΘΕΣΙΑ"


def test_legal_only_types_are_not_forced_into_financial_ontology():
    assert mapped_tag("B-LEG-REFS") == "O"
    assert mapped_tag("B-FACILITY") == "O"
