import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_task1_entity_silver_corpus import (
    document_issuer,
    report_issuer_alias,
    resolve_spans,
    word_chunks,
)


def test_extracts_official_issuer_alias_from_report_url():
    url = (
        "https://example.test/Οικονομική%20Έκθεση%20AS%20COMPANY%20Α.Ε.%20"
        "(2025,Εξαμηνιαία,Ενοποιημένη).pdf"
    )
    assert report_issuer_alias(url) == "AS COMPANY Α.Ε"


def test_resolves_document_issuer_from_stable_paths():
    assert document_issuer(Path("athex_issuers/846_001.txt")) == "846"
    assert document_issuer(Path("athex_url_recovery/208_abcd.txt")) == "208"
    assert document_issuer(Path("bank_greece_governor_2024.txt")) is None


def test_chunks_do_not_repeat_boundary_words():
    text = " ".join(f"word{index}" for index in range(400))
    chunks = list(word_chunks(text, words_per_chunk=180))
    assert len(chunks) == 3
    assert chunks[0].split()[-1] == "word179"
    assert chunks[1].split()[0] == "word180"
    assert chunks[2].split()[0] == "word360"


def test_longest_overlap_wins():
    spans = [
        {"start": 0, "end": 4, "type": "ΟΡΓΑΝΙΣΜΟΣ", "signals": ["a"]},
        {"start": 0, "end": 8, "type": "ΟΡΓΑΝΙΣΜΟΣ", "signals": ["b"]},
    ]
    assert resolve_spans(spans) == [spans[1]]
