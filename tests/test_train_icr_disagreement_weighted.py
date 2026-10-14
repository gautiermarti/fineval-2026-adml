import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.train_icr_disagreement_weighted_cv import (
    load_disagreement_ids,
    weighted_cross_entropy,
)


def test_disagreement_ids_are_frozen_fourteen_rows():
    values = load_disagreement_ids()
    assert len(values) == 14
    assert {38, 86, 87}.issubset(values)


def test_weighted_cross_entropy_normalizes_by_weight_sum():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 0])
    weights = torch.tensor([1.0, 2.0])
    individual = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    expected = (individual[0] + 2 * individual[1]) / 3
    assert torch.allclose(weighted_cross_entropy(logits, labels, weights), expected)

