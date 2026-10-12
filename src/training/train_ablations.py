"""Train CompleteCNN across loss and optimizer variants.

Supports the 2x2 ablation matrix over loss functions (Focal vs Asymmetric)
and optimizers (AdamW vs SAM-wrapped AdamW). Each pairing trains with the
same protocol (epochs, SWA tail, best-val tracking) and saves its model and
evaluation metrics plus a comparison.csv row, matching other stage scripts.
"""

import argparse
import logging
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.optim import AdamW, Optimizer
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from src.ETL_data.loader import get_dataloaders
from src.model.architecture import CompleteCNN
from src.model.asymmetric_loss import AsymmetricLoss
from src.model.loss import FocalLoss
from src.training.optimizer_benchmark import compute_balanced_alpha, labels_from_loader
from src.training.sam import SAM, build_sam_adamw, sam_train_one_epoch
from src.training.utils import evaluate_model, save_evaluation_results, upsert_comparison_row

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LossName = Literal["asl", "focal"]
OptimizerName = Literal["sam", "adamw"]

STAGE_DIRS: dict[tuple[str, str], str] = {
    ("asl", "sam"): "asl_sam_cnn",
    ("focal", "sam"): "focal_sam_cnn",
    ("asl", "adamw"): "asl_adamw_cnn",
    ("focal", "adamw"): "focal_adamw_cnn",
}

STAGE_LABELS: dict[tuple[str, str], str] = {
    ("asl", "sam"): "6. + ASL+SAM",
    ("focal", "sam"): "7. + Focal+SAM",
    ("asl", "adamw"): "8. + ASL+AdamW",
    ("focal", "adamw"): "9. + Focal+AdamW",
}


def build_variant_loss(
    loss_name: LossName,
    alpha: float | torch.Tensor,
    gamma_pos: float = 0.0,
    gamma_neg: float = 4.0,
    margin: float = 0.05,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Build the loss module for a variant.

    Args:
        loss_name: Which loss to build.
        alpha: Class-balanced weighting tensor or scalar.
        gamma_pos: ASL focusing for positives.
        gamma_neg: ASL focusing for negatives.
        margin: ASL negative probability margin.
        focal_gamma: Focal Loss focusing factor.

    Returns:
        Configured loss module.
    """
    if loss_name == "focal":
        return FocalLoss(alpha=alpha, gamma=focal_gamma)
    return AsymmetricLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg, margin=margin, alpha=alpha)


def build_variant_optimizer(
    optimizer_name: OptimizerName,
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    rho: float = 0.05,
) -> Optimizer:
    """Build the optimizer for a variant.

    Args:
        optimizer_name: Which optimizer to build.
        model: Network whose parameters are optimized.
        lr: Learning rate.
        weight_decay: Weight decay.
        rho: SAM neighborhood radius.

    Returns:
        Configured optimizer (SAM-wrapped or plain AdamW).
    """
    if optimizer_name == "sam":
        return build_sam_adamw(model, lr=lr, weight_decay=weight_decay, rho=rho)
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_variant(
    loss: LossName = "asl",
    optimizer: OptimizerName = "sam",
    epochs: int = 35,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    rho: float = 0.05,
    gamma_pos: float = 0.0,
    gamma_neg: float = 4.0,
    margin: float = 0.05,
    focal_gamma: float = 2.0,
    use_swa: bool = True,
    output_dir: str | None = None,
) -> None:
    """Train one loss/optimizer variant end to end.

    Saves model weights and metrics via save_evaluation_results and upserts
    the stage row in evaluation/comparison.csv.

    Args:
        loss: Loss variant to train with.
        optimizer: Optimizer variant to train with.
        epochs: Number of training epochs.
        lr: Learning rate for the base optimizer.
        weight_decay: Weight decay for the base optimizer.
        rho: SAM neighborhood radius.
        gamma_pos: ASL focusing for positives.
        gamma_neg: ASL focusing for negatives.
        margin: ASL negative probability margin.
        focal_gamma: Focal Loss focusing factor.
        use_swa: Enable Stochastic Weight Averaging over final epochs.
        output_dir: Override output directory (defaults to stage mapping).
    """
    logger.info("Starting %s+%s training...", loss.upper(), optimizer.upper())
    try:
        torch.backends.nnpack.set_flags(False)
    except Exception:
        pass
    try:
        torch.backends.mkldnn.set_flags(False)
    except Exception:
        pass
    device = torch.device("cpu")
    model = CompleteCNN(num_classes=8).to(device)
    train_dl, val_dl, test_dl = get_dataloaders(batch_size=32)
    alpha = compute_balanced_alpha(labels_from_loader(train_dl))
    loss_fn = build_variant_loss(loss, alpha, gamma_pos, gamma_neg, margin, focal_gamma)
    opt = build_variant_optimizer(optimizer, model, lr, weight_decay, rho)
    swa_model = AveragedModel(model) if use_swa else None
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(epochs):
        if isinstance(opt, SAM):
            train_loss = sam_train_one_epoch(model, train_dl, loss_fn, opt, device)
        else:
            train_loss = _train_one_epoch_simple(model, train_dl, loss_fn, opt, device)
        val_loss, val_metrics = evaluate_model(model, val_dl, loss_fn, device)
        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | Val mAP: %.4f",
            epoch + 1,
            epochs,
            train_loss,
            val_loss,
            val_metrics["mAP"],
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if swa_model is not None and epoch >= epochs - 5:
            swa_model.update_parameters(model)
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_metrics = evaluate_model(model, test_dl, loss_fn, device)
    loss_label = "AsymmetricLoss" if loss == "asl" else "FocalLoss"
    opt_label = "SAM(AdamW)" if optimizer == "sam" else "AdamW"
    metrics = {
        "test_loss": test_loss,
        **test_metrics,
        "epochs_trained": epochs,
        "optimizer": opt_label,
        "loss_fn": loss_label,
        "rho": rho,
        "gamma_pos": gamma_pos,
        "gamma_neg": gamma_neg,
        "margin": margin,
        "focal_gamma": focal_gamma,
    }
    stage_dir = output_dir or f"evaluation/{STAGE_DIRS[(loss, optimizer)]}"
    save_evaluation_results(model, metrics, stage_dir)
    out_dir = Path(stage_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    upsert_comparison_row(
        STAGE_LABELS[(loss, optimizer)],
        float(test_metrics["micro_f1"]),
        float(test_metrics["macro_f1"]),
        float(test_metrics["mAP"]),
    )


def _train_one_epoch_simple(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    """Train one epoch with a standard single-step optimizer.

    Args:
        model: Network to train.
        loader: DataLoader yielding (X, y) batches.
        loss_fn: Loss module accepting (logits, targets).
        optimizer: Plain optimizer.
        device: Device for computation.

    Returns:
        Average epoch loss.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        batch_loss: torch.Tensor = loss_fn(model(X), y)
        batch_loss.backward()
        optimizer.step()
        batch_size = X.size(0)
        total_loss += float(batch_loss.item()) * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


def train_asl_sam(
    epochs: int = 35,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    rho: float = 0.05,
    gamma_pos: float = 0.0,
    gamma_neg: float = 4.0,
    margin: float = 0.05,
    use_swa: bool = True,
) -> None:
    """Train CompleteCNN with ASL and SAM-wrapped AdamW.

    Backwards-compatible entry preserving the original ASL+SAM behavior.

    Args:
        epochs: Number of training epochs.
        lr: Learning rate for AdamW base optimizer.
        weight_decay: Weight decay for AdamW base optimizer.
        rho: SAM neighborhood radius.
        gamma_pos: ASL focusing for positives.
        gamma_neg: ASL focusing for negatives.
        margin: ASL negative probability margin.
        use_swa: Enable Stochastic Weight Averaging over final epochs.
    """
    train_variant(
        loss="asl",
        optimizer="sam",
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        rho=rho,
        gamma_pos=gamma_pos,
        gamma_neg=gamma_neg,
        margin=margin,
        use_swa=use_swa,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for variant training.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Train a loss/optimizer variant.")
    parser.add_argument("--loss", choices=["asl", "focal"], default="asl")
    parser.add_argument("--optimizer", choices=["sam", "adamw"], default="sam")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_variant(
        loss=args.loss,
        optimizer=args.optimizer,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        rho=args.rho,
        output_dir=args.output_dir,
    )
