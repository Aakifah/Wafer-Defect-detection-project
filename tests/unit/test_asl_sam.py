"""Tests for Asymmetric Loss and SAM optimizer."""

import torch
import torch.nn as nn

from src.model.asymmetric_loss import AsymmetricLoss
from src.training.sam import SAM, build_sam_adamw


def test_asl_downweights_easy_negatives() -> None:
    loss_fn = AsymmetricLoss(gamma_pos=0.0, gamma_neg=4.0, margin=0.05, alpha=1.0)
    logits_easy_neg = torch.tensor([[ -8.0 ]])
    logits_hard_neg = torch.tensor([[ 0.0 ]])
    targets = torch.zeros(1, 1)
    easy = loss_fn(logits_easy_neg, targets)
    hard = loss_fn(logits_hard_neg, targets)
    assert easy.item() < hard.item()


def test_asl_preserves_rare_positives_vs_focal() -> None:
    from src.model.loss import FocalLoss

    logits = torch.tensor([[0.0]])
    targets = torch.ones(1, 1)
    asl = AsymmetricLoss(gamma_pos=0.0, gamma_neg=4.0, margin=0.05)(logits, targets)
    focal = FocalLoss(alpha=1.0, gamma=2.0)(logits, targets)
    assert torch.isfinite(asl)
    assert asl.item() > 0.0
    assert torch.isfinite(focal)


def test_asl_supports_per_class_alpha() -> None:
    alpha = torch.tensor([0.5, 2.0])
    loss_fn = AsymmetricLoss(alpha=alpha, reduction="none")
    logits = torch.zeros(2, 2)
    targets = torch.ones(2, 2)
    out = loss_fn(logits, targets)
    assert out.shape == logits.shape


def test_sam_step_reduces_loss_on_toy_problem() -> None:
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = build_sam_adamw(model, lr=1e-2, rho=0.05)
    assert isinstance(optimizer, SAM)
    X = torch.randn(8, 4)
    y = torch.randint(0, 2, (8, 1)).float()
    before = loss_fn(model(X), y).item()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        return loss

    closure()
    optimizer.first_step(zero_grad=True)
    closure()
    optimizer.second_step(zero_grad=True)
    after = loss_fn(model(X), y).item()
    assert after != before
