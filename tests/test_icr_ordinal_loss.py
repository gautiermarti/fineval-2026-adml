import torch
import torch.nn.functional as F

from fineval.icr_ordinal_loss import (
    distance_aware_cross_entropy,
    ordered_cdf_loss,
)


def test_adjacent_error_has_lower_cdf_loss_than_extreme_error():
    label = torch.tensor([0])
    adjacent = torch.tensor([[0.0, 8.0, 0.0, 0.0, 0.0]])
    extreme = torch.tensor([[0.0, 0.0, 0.0, 0.0, 8.0]])
    assert ordered_cdf_loss(adjacent, label) < ordered_cdf_loss(extreme, label)


def test_zero_weight_exactly_recovers_cross_entropy_and_gradients():
    logits = torch.tensor(
        [[0.2, -0.4, 0.8, 0.1, -0.2], [-0.1, 0.3, 0.0, 0.7, -0.5]],
        requires_grad=True,
    )
    labels = torch.tensor([2, 3])
    total, cross_entropy, cdf = distance_aware_cross_entropy(
        logits, labels, distance_weight=0.0,
    )
    assert torch.equal(total, cross_entropy)
    assert torch.allclose(cross_entropy, F.cross_entropy(logits, labels))
    assert cdf.item() > 0
    total.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_perfect_ordered_distribution_has_near_zero_loss():
    logits = torch.full((5, 5), -20.0)
    logits[torch.arange(5), torch.arange(5)] = 20.0
    assert ordered_cdf_loss(logits, torch.arange(5)).item() < 1e-12
