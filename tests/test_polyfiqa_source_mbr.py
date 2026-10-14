import numpy as np

from fineval.polyfiqa_source_mbr import (
    bidirectional_support, fact_verbalization, request_description,
)


def test_fact_verbalization_preserves_metric_values_and_units():
    text = fact_verbalization({
        "family": "revenue",
        "numeric": [{
            "metric": "revenue", "values": [62_020, 52_747],
            "unit": "millions", "derived": False,
        }],
    })
    assert "revenue" in text
    assert "62020" in text and "52747" in text
    assert "millions" in text


def test_bidirectional_support_is_one_for_identical_unit_vectors():
    vectors = np.eye(2, dtype=float)
    precision, recall, f1 = bidirectional_support(vectors, vectors)
    assert (precision, recall, f1) == (1.0, 1.0, 1.0)
    assert "liabilities" in request_description("balance")

