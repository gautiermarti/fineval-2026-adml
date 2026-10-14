import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_icr_japanese_ordinal_cv import cumulative_targets, decode_logits


def test_cumulative_targets_are_ordered():
    labels = torch.tensor([0, 1, 2, 3, 4])
    assert cumulative_targets(labels).tolist() == [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ]


def test_decode_counts_passed_ordered_boundaries():
    logits = torch.tensor([
        [-1.0, -2.0, -3.0, -4.0],
        [1.0, -1.0, -2.0, -3.0],
        [4.0, 3.0, 2.0, 1.0],
    ])
    assert decode_logits(logits).tolist() == [0, 1, 4]
