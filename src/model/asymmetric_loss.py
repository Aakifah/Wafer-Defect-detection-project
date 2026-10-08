"""Asymmetric Loss for multi-label classification."""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

Reduction = Literal["mean", "sum", "none"]


class AsymmetricLoss(nn.Module):
    """Asymmetric loss decoupling positive and negative focusing.

    Down-weights easy negatives aggressively via gamma_neg and probability
    margin while preserving rare positive signal via gamma_pos. Combines with
    optional per-class alpha weighting for minority defects.

    Args:
        gamma_pos: Focusing factor for positive samples.
        gamma_neg: Focusing factor for negative samples.
        margin: Probability margin shifting negative logits.
        alpha: Scalar or per-class weighting tensor.
        reduction: Reduction mode applied to output.
        eps: Numerical stability term.

    Raises:
        ValueError: If gamma values are negative or shapes are invalid.
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        margin: float = 0.05,
        alpha: float | torch.Tensor = 1.0,
        reduction: Reduction = "mean",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if gamma_pos < 0:
            raise ValueError("gamma_pos must be non-negative")
        if gamma_neg < 0:
            raise ValueError("gamma_neg must be non-negative")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: 'mean', 'sum', or 'none'")
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.margin = float(margin)
        self.eps = float(eps)
        self.reduction = reduction
        self.alpha: float | None
        if isinstance(alpha, torch.Tensor):
            if torch.any(alpha < 0):
                raise ValueError("alpha tensor values must be non-negative")
            self.alpha = None
            self.register_buffer("alpha_tensor", alpha.detach().clone().float())
        else:
            if alpha < 0:
                raise ValueError("alpha must be non-negative")
            self.alpha = float(alpha)
            self.register_buffer("alpha_tensor", None)

    def _alpha_factor(self, inputs: torch.Tensor) -> torch.Tensor | float:
        """Broadcast stored alpha to input shape.

        Args:
            inputs: Logit tensor of shape (B, num_classes).

        Returns:
            Broadcastable alpha weight.

        Raises:
            RuntimeError: If alpha configuration is invalid.
            ValueError: If per-class alpha shape is incompatible.
        """
        alpha_buffer = self._buffers["alpha_tensor"]
        if alpha_buffer is None:
            if self.alpha is None:
                raise RuntimeError("AsymmetricLoss alpha configuration is invalid")
            return self.alpha
        alpha = alpha_buffer.to(device=inputs.device, dtype=inputs.dtype)
        if alpha.ndim == 0:
            return alpha
        if alpha.shape == inputs.shape:
            return alpha
        if inputs.ndim >= 2 and alpha.ndim == 1 and alpha.numel() == inputs.shape[1]:
            view_shape = [1, alpha.numel(), *([1] * (inputs.ndim - 2))]
            return alpha.view(*view_shape)
        raise ValueError("alpha tensor must be scalar, match logits shape, or be 1D per-class")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute asymmetric loss from raw logits.

        Args:
            inputs: Raw logits of shape (B, num_classes).
            targets: Binary targets of shape (B, num_classes).

        Returns:
            Reduced loss scalar or per-element tensor.

        Raises:
            ValueError: If inputs and targets shapes differ.
        """
        if inputs.shape != targets.shape:
            raise ValueError(
                f"inputs and targets must have the same shape; got "
                f"{tuple(inputs.shape)} and {tuple(targets.shape)}"
            )
        targets = targets.to(device=inputs.device, dtype=inputs.dtype)
        probs = torch.sigmoid(inputs)
        probs_pos = probs.clamp(min=self.eps, max=1.0 - self.eps)
        probs_neg = (1.0 - probs).clamp(min=self.eps, max=1.0 - self.eps)
        shifted = (probs - self.margin).clamp(min=0.0, max=1.0)
        pos_loss = -targets * torch.pow(1.0 - probs_pos, self.gamma_pos) * torch.log(probs_pos)
        neg_loss = (
            -(1.0 - targets)
            * torch.pow(shifted, self.gamma_neg)
            * torch.log(probs_neg.clamp(min=self.eps))
        )
        loss = self._alpha_factor(inputs) * (pos_loss + neg_loss)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def as_focal_equivalent(self) -> dict[str, float]:
        """Describe symmetric focal parameters approximating this instance.

        Returns:
            Mapping with averaged gamma and margin metadata.
        """
        return {
            "gamma": (self.gamma_pos + self.gamma_neg) / 2.0,
            "gamma_pos": self.gamma_pos,
            "gamma_neg": self.gamma_neg,
            "margin": self.margin,
        }


def binary_cross_entropy_with_margin(
    inputs: torch.Tensor, targets: torch.Tensor, margin: float = 0.05
) -> torch.Tensor:
    """Compute BCE with negative probability margin for diagnostics.

    Args:
        inputs: Raw logits of shape (B, num_classes).
        targets: Binary targets of shape (B, num_classes).
        margin: Margin subtracted from negative probabilities.

    Returns:
        Elementwise loss tensor.
    """
    probs = torch.sigmoid(inputs).clamp(min=1e-8, max=1.0 - 1e-8)
    shifted_neg = (probs - margin).clamp(min=0.0, max=1.0)
    pos_term = -targets * torch.log(probs)
    neg_term = -(1.0 - targets) * torch.log((1.0 - shifted_neg).clamp(min=1e-8))
    return pos_term + neg_term + F.binary_cross_entropy_with_logits(inputs, targets) * 0.0
