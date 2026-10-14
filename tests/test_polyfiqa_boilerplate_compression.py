from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_polyfiqa_boilerplate_compression import (  # noqa: E402
    normalize_sentence,
    removed_sentence,
    signature_statistics,
)


def test_normalize_sentence_masks_issuer_and_financial_values():
    assert normalize_sentence(
        "Microsoft's revenue increased by $12.5 billion, or 18%."
    ) == "<company>'s revenue increased by <num>, or <num>."


def test_removed_sentence_verifies_exact_single_deletion():
    baseline = "Answer: First fact. Generic implication.\nNews Evidence: None."
    candidate = "Answer: First fact.\nNews Evidence: None."
    assert removed_sentence(baseline, candidate, "delete_sentence:1") == (
        "Generic implication."
    )


def test_signature_statistics_requires_every_training_company():
    examples = []
    for source, delta in (("HON", 0.01), ("JNJ", 0.02), ("MSFT", 0.03)):
        examples.extend({
            "company": source, "signature": "generic.", "delta": delta,
        } for _ in range(3))
    statistics = signature_statistics(examples, held="UVV")
    assert statistics["generic."]["eligible"] is True
    assert statistics["generic."]["worst_company_mean"] == 0.01


def test_signature_statistics_rejects_one_negative_training_company():
    examples = []
    for source, delta in (("HON", 0.01), ("JNJ", -0.001), ("MSFT", 0.03)):
        examples.extend({
            "company": source, "signature": "generic.", "delta": delta,
        } for _ in range(3))
    assert signature_statistics(examples, held="UVV")["generic."]["eligible"] is False
