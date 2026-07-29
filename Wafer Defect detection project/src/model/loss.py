"""Custom loss functions for model training."""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

Reduction = Literal["mean", "sum", "none"]


class FocalLoss(nn.Module):
    """Focal loss for multi-label classification with raw CNN logits."""

    def __init__(
        self,
        alpha: float | torch.Tensor = 1.0,
        gamma: float = 2.0,
        reduction: Reduction = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: 'mean', 'sum', or 'none'")

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

        self.gamma = gamma
        self.reduction = reduction

    def _alpha_factor(self, inputs: torch.Tensor) -> torch.Tensor | float:
        alpha_buffer = self._buffers["alpha_tensor"]
        if alpha_buffer is None:
            if self.alpha is None:
                raise RuntimeError("FocalLoss alpha configuration is invalid")
            return self.alpha

        alpha = alpha_buffer.to(device=inputs.device, dtype=inputs.dtype)
        if alpha.ndim == 0:
            return alpha
        if alpha.shape == inputs.shape:
            return alpha
        if inputs.ndim >= 2 and alpha.ndim == 1 and alpha.numel() == inputs.shape[1]:
            view_shape = [1, alpha.numel(), *([1] * (inputs.ndim - 2))]
            return alpha.view(*view_shape)

        raise ValueError(
            "alpha tensor must be scalar, match the logits shape, or be a "
            "1D tensor with one value per class"
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the Focal Loss.
        
        Args:
            inputs: Raw logits from the model (before sigmoid) of shape (B, num_classes).
            targets: Binary target labels of shape (B, num_classes).
            
        Returns:
            The computed loss scalar (or tensor depending on reduction).
        """
        if inputs.shape != targets.shape:
            raise ValueError(
                f"inputs and targets must have the same shape; got "
                f"{tuple(inputs.shape)} and {tuple(targets.shape)}"
            )

        targets = targets.to(device=inputs.device, dtype=inputs.dtype)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self._alpha_factor(inputs) * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
