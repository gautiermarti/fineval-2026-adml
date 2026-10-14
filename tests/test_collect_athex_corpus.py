from pathlib import Path
import sys

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from collect_athex_corpus import is_greek_financial_report
from recover_athex_corpus import target_path


def test_collector_rejects_large_404_page():
    text = (
        "Title: Η σελίδα δε βρέθηκε\n"
        "Warning: Target URL returned error 404: Not Found\n"
        + "οικονομική έκθεση " * 2_000
    )
    assert not is_greek_financial_report(text)


def test_collector_accepts_substantial_greek_financial_text():
    text = (
        "Η οικονομική έκθεση περιλαμβάνει χρηματοοικονομικές καταστάσεις, "
        "ισολογισμό, έσοδα και κέρδη. " * 500
    )
    assert is_greek_financial_report(text)


def test_recovery_filename_is_stable_and_url_addressed(tmp_path):
    first = target_path(tmp_path, "846", "https://example.test/a.pdf")
    second = target_path(tmp_path, "846", "https://example.test/b.pdf")
    assert first == target_path(tmp_path, "846", "https://example.test/a.pdf")
    assert first != second
