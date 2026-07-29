"""Adaptive Singular Value Decomposition (ASVD) for CompleteCNN compression.

Operates directly on CompleteCNN instances. Applies SVD-based rank reduction
to all Linear layers, preserving the specified fraction of cumulative energy.
"""

import logging

import torch
import torch.nn as nn

from src.model.architecture import CompleteCNN

logger = logging.getLogger(__name__)


def apply_asvd_to_linear(model: nn.Module, preserve_energy: float = 0.90) -> None:
    """Compress Linear layers in-place using ASVD.

    ASVD: The algorithm takes the big massive matrix weights of the model 
    and then breaks them down into smaller matrices because matrix multiplication 
    scales cubically. This saves parameters and computation while preserving the
    specified energy.

    Args:
        model: The model to compress (modified in-place).
        preserve_energy: Fraction of cumulative energy to retain (0.0-1.0).
    """
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            U, S, V = torch.linalg.svd(module.weight.data, full_matrices=False)

            energy = torch.cumsum(S**2, dim=0) / torch.sum(S**2)
            valid_k = (energy >= preserve_energy).nonzero(as_tuple=False)

            if len(valid_k) > 0:
                k = int(valid_k[0].item()) + 1
            else:
                k = len(S)

            k = max(1, min(k, module.in_features, module.out_features))

            logger.info(
                "ASVD: %s %dx%d -> rank %d (energy=%.4f)",
                name,
                module.in_features,
                module.out_features,
                k,
                float(energy[min(k - 1, len(energy) - 1)]),
            )

            linear1 = nn.Linear(module.in_features, k, bias=False)
            linear2 = nn.Linear(k, module.out_features, bias=(module.bias is not None))

            linear1.weight.data = V[:k, :]
            linear2.weight.data = U[:, :k] * S[:k].unsqueeze(0)
            if module.bias is not None:
                linear2.bias.data = module.bias.data

            setattr(model, name, nn.Sequential(linear1, linear2))


def create_compressed_cnn(
    num_classes: int = 8,
    preserve_energy: float = 0.90,
) -> CompleteCNN:
    """Create a CompleteCNN and apply ASVD compression to its Linear layers.

    Args:
        num_classes: Number of output defect classes.
        preserve_energy: Fraction of cumulative energy to retain.

    Returns:
        ASVD-compressed CompleteCNN instance.
    """
    model = CompleteCNN(num_classes=num_classes)
    apply_asvd_to_linear(model, preserve_energy=preserve_energy)
    return model
