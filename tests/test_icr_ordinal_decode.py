import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from predict_icr_japanese_ensemble import decode_probabilities


def test_ordinal_decoder_uses_expected_score_instead_of_mode():
    probabilities = np.asarray([[0.0, 0.0, 0.40, 0.35, 0.25]])
    assert decode_probabilities(probabilities, ordinal_mean=False).tolist() == ["0"]
    assert decode_probabilities(probabilities, ordinal_mean=True).tolist() == ["+1"]


def test_ordinal_decoder_clips_to_supported_labels():
    probabilities = np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 1.0]])
    assert decode_probabilities(probabilities, ordinal_mean=True).tolist() == ["-2", "+2"]
