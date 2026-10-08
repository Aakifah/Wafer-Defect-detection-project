"""Sharpness-Aware Minimization wrapper for flat-minima optimization."""

from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class SAM(Optimizer):
    """Sharpness-Aware Minimization optimizer wrapper.

    Performs a two-step update seeking parameters whose neighborhood has
    uniformly low loss, improving generalization on noisy wafer maps.

    Args:
        params: Model parameters to optimize.
        base_optimizer: Optimizer class used for the underlying update.
        rho: Neighborhood radius for perturbation.
        adaptive: Scale perturbation by parameter magnitude.
        kwargs: Keyword arguments forwarded to the base optimizer.
    """

    def __init__(
        self,
        params: Any,
        base_optimizer: type[Optimizer],
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs: Any,
    ) -> None:
        if rho < 0.0:
            raise ValueError("rho must be non-negative")
        defaults = {"rho": rho, "adaptive": adaptive}
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True) -> None:
        """Ascend to the worst-case neighborhood point.

        Args:
            zero_grad: Zero gradients after perturbation.
        """
        grad_norm = self._grad_norm()
        scale = self.param_groups[0]["rho"] / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                factor = torch.abs(p) if group["adaptive"] else 1.0
                e_w = p.grad * factor * scale.to(p.device)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True) -> None:
        """Descend from the perturbed point and restore parameters.

        Args:
            zero_grad: Zero gradients after the update.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """Advance one SAM step using an optional closure.

        Args:
            closure: Callable recomputing loss and gradients.

        Raises:
            ValueError: If closure is not provided.
        """
        if closure is None:
            raise ValueError("SAM requires a closure performing forward-backward")
        closure()
        self.first_step(zero_grad=True)
        closure()
        self.second_step(zero_grad=True)

    def _grad_norm(self) -> torch.Tensor:
        """Compute global gradient norm across parameter groups.

        Returns:
            Scalar gradient norm tensor on the first parameter device.
        """
        device = self.param_groups[0]["params"][0].device
        norms = [
            (torch.abs(p) * p.grad).norm(p=2).to(device)
            if self.param_groups[0]["adaptive"] and p.grad is not None
            else (p.grad.norm(p=2).to(device) if p.grad is not None else torch.tensor(0.0, device=device))
            for group in self.param_groups
            for p in group["params"]
        ]
        stacked: torch.Tensor = torch.stack(norms)
        result: torch.Tensor = torch.norm(stacked, p=2)
        return result

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state into base and wrapper.

        Args:
            state_dict: State dictionary to restore.
        """
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def sam_train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    loss_fn: nn.Module,
    sam_optimizer: SAM,
    device: torch.device,
) -> float:
    """Train one epoch with SAM two-step updates.

    Args:
        model: Network to train.
        loader: DataLoader yielding (X, y) batches.
        loss_fn: Loss module accepting (logits, targets).
        sam_optimizer: SAM-wrapped optimizer.
        device: Device for computation.

    Returns:
        Average epoch loss at perturbed weights.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)

        def closure() -> torch.Tensor:
            sam_optimizer.zero_grad()
            logits = model(X)
            raw: torch.Tensor = loss_fn(logits, y)
            raw.backward()
            return raw

        first_loss = closure()
        sam_optimizer.first_step(zero_grad=True)
        second_loss = closure()
        sam_optimizer.second_step(zero_grad=True)
        batch_size = X.size(0)
        total_loss += float(second_loss.item()) * batch_size
        total_samples += batch_size
        _ = first_loss
    return total_loss / max(total_samples, 1)


def build_sam_adamw(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    rho: float = 0.05,
) -> SAM:
    """Build SAM wrapping AdamW with wafer defaults.

    Args:
        model: Network whose parameters are optimized.
        lr: Learning rate for AdamW.
        weight_decay: Weight decay for AdamW.
        rho: SAM neighborhood radius.

    Returns:
        Configured SAM optimizer.
    """
    from torch.optim import AdamW

    return SAM(model.parameters(), AdamW, rho=rho, adaptive=False, lr=lr, weight_decay=weight_decay)
