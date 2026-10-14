from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from predict_polyfiqa_lookup_exact import cap_answer, normalize


def test_normalize_is_whitespace_and_nfkc_only():
    assert normalize("  Ａ\n B  ") == "A B"


def test_cap_answer_preserves_evidence_heading():
    answer = "Answer: " + " ".join(["body"] * 97) + "\nEvidence: one two three four"
    capped = cap_answer(answer)
    assert len(capped.split()) == 100
    assert "\nEvidence:" in capped


def test_cap_answer_leaves_short_answer_unchanged():
    answer = "Answer: Yes.\nEvidence: filing."
    assert cap_answer(answer) == answer
