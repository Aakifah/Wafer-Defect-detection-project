import pytest
import torch
import torch.nn.functional as F

from src.model.loss import FocalLoss


def test_focal_loss_returns_scalar_for_mean_reduction() -> None:
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    logits = torch.randn(4, 8)
    targets = torch.randint(0, 2, (4, 8)).float()

    loss = loss_fn(logits, targets)

    assert loss.dim() == 0
    assert loss.item() >= 0


def test_focal_loss_matches_bce_with_logits_formula() -> None:
    logits = torch.tensor([[0.0, 2.0], [-1.0, 3.0]])
    targets = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    alpha = 0.5
    gamma = 2.0

    actual = FocalLoss(alpha=alpha, gamma=gamma, reduction="none")(logits, targets)

    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce_loss)
    expected = alpha * (1 - pt) ** gamma * bce_loss

    torch.testing.assert_close(actual, expected)


def test_focal_loss_supports_per_class_alpha() -> None:
    logits = torch.tensor([[0.0, 2.0, -1.0], [1.0, -2.0, 3.0]])
    targets = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    class_alpha = torch.tensor([0.1, 0.5, 2.0])

    actual = FocalLoss(alpha=class_alpha, gamma=1.5, reduction="none")(logits, targets)

    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce_loss)
    expected = class_alpha.view(1, 3) * (1 - pt) ** 1.5 * bce_loss

    assert actual.shape == logits.shape
    torch.testing.assert_close(actual, expected)


def test_focal_loss_supports_sum_reduction() -> None:
    logits = torch.randn(4, 8)
    targets = torch.randint(0, 2, (4, 8)).float()

    none_loss = FocalLoss(alpha=0.75, gamma=2.0, reduction="none")(logits, targets)
    sum_loss = FocalLoss(alpha=0.75, gamma=2.0, reduction="sum")(logits, targets)

    torch.testing.assert_close(sum_loss, none_loss.sum())


def test_focal_loss_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="gamma"):
        FocalLoss(gamma=-1.0)

    with pytest.raises(ValueError, match="alpha"):
        FocalLoss(alpha=-0.1)

    with pytest.raises(ValueError, match="alpha tensor"):
        FocalLoss(alpha=torch.tensor([1.0, -1.0]))

    with pytest.raises(ValueError, match="reduction"):
        FocalLoss(reduction="median")  # type: ignore[arg-type]


def test_focal_loss_rejects_mismatched_shapes() -> None:
    loss_fn = FocalLoss()
    logits = torch.randn(4, 8)
    targets = torch.randn(4, 7)

    with pytest.raises(ValueError, match="same shape"):
        loss_fn(logits, targets)


def test_focal_loss_rejects_wrong_per_class_alpha_shape() -> None:
    loss_fn = FocalLoss(alpha=torch.ones(7))
    logits = torch.randn(4, 8)
    targets = torch.randint(0, 2, (4, 8)).float()

    with pytest.raises(ValueError, match="one value per class"):
        loss_fn(logits, targets)


def test_focal_loss_is_stable_for_extreme_logits() -> None:
    loss_fn = FocalLoss(alpha=torch.ones(8), gamma=5.0)
    logits = torch.tensor([[100.0, -100.0, 50.0, -50.0, 20.0, -20.0, 0.0, 1.0]])
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]])

    loss = loss_fn(logits, targets)

    assert torch.isfinite(loss)
    assert loss.item() >= 0
