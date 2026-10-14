import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.audit_ebisu_annotation_disagreement import annotation_statistics


def test_annotation_statistics_preserves_adjudicated_minority_gold():
    result = annotation_statistics(["+2", "+1", "+1"], "+2")
    assert result["disagreement"] is True
    assert result["plurality_labels"] == ["+1"]
    assert result["gold_present"] is True
    assert result["gold_is_plurality"] is False
    assert result["gold_agreement_fraction"] == 1 / 3
    assert result["annotator_ordinal_mean"] == 4 / 3
    assert result["gold_ordinal_value"] == 2


def test_annotation_statistics_unanimous_gold():
    result = annotation_statistics(["0", "0"], "0")
    assert result["unanimous"] is True
    assert result["disagreement"] is False
    assert result["gold_is_unique_majority"] is True
    assert result["normalized_entropy"] == 0.0

