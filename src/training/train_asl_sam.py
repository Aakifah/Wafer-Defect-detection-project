"""Train CompleteCNN with Asymmetric Loss and SAM optimizer."""

import logging
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch.optim.swa_utils import AveragedModel

from src.ETL_data.loader import get_dataloaders
from src.model.architecture import CompleteCNN
from src.model.asymmetric_loss import AsymmetricLoss
from src.training.optimizer_benchmark import compute_balanced_alpha, labels_from_loader
from src.training.sam import build_sam_adamw, sam_train_one_epoch
from src.training.utils import evaluate_model, save_evaluation_results

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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
    logger.info("Starting ASL+SAM training...")
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
    loss_fn = AsymmetricLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg, margin=margin, alpha=alpha)
    optimizer = build_sam_adamw(model, lr=lr, weight_decay=weight_decay, rho=rho)
    swa_model = AveragedModel(model) if use_swa else None
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(epochs):
        train_loss = sam_train_one_epoch(model, train_dl, loss_fn, optimizer, device)
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
    metrics = {
        "test_loss": test_loss,
        **test_metrics,
        "epochs_trained": epochs,
        "optimizer": "SAM(AdamW)",
        "loss_fn": "AsymmetricLoss",
        "rho": rho,
        "gamma_pos": gamma_pos,
        "gamma_neg": gamma_neg,
        "margin": margin,
    }
    save_evaluation_results(model, metrics, "evaluation/asl_sam_cnn")
    out_dir = Path("evaluation/asl_sam_cnn")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")


if __name__ == "__main__":
    train_asl_sam()
