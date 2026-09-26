"""Script to train the CNN using Focal Loss."""

import logging
from pathlib import Path

import torch
import torch.optim as optim
from dotenv import load_dotenv

from src.ETL_data.loader import get_dataloaders
from src.model.architecture import CompleteCNN
from src.model.loss import FocalLoss
from src.training.optimizer_benchmark import compute_balanced_alpha, labels_from_loader
from src.training.utils import (
    evaluate_model,
    save_evaluation_results,
    train_one_epoch,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def train_focal() -> None:
    """Trains the CNN with Focal Loss.
    
    Trains the CompleteCNN from scratch using Focal Loss to handle class imbalance,
    combined with the best optimizer (AdamW).
    """
    logger.info("Starting training for Focal Loss CNN...")
    
    device = torch.device("cpu")
    model = CompleteCNN(num_classes=8).to(device)
    
    train_dl, val_dl, test_dl = get_dataloaders(batch_size=32)
    
    labels = labels_from_loader(train_dl)
    alpha = compute_balanced_alpha(labels)
    loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    epochs = 35
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_dl, loss_fn, optimizer, device)
        val_loss, val_metrics = evaluate_model(model, val_dl, loss_fn, device)
        
        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | Val mAP: %.4f",
            epoch + 1, epochs, train_loss, val_loss, val_metrics["mAP"]
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    test_loss, test_metrics = evaluate_model(model, test_dl, loss_fn, device)
    logger.info("Test Metrics: %s", test_metrics)
    
    metrics = {
        "test_loss": test_loss,
        **test_metrics,
        "epochs_trained": epochs,
        "optimizer": "AdamW",
        "loss_fn": "FocalLoss"
    }
    save_evaluation_results(model, metrics, "evaluation/focal_cnn")
    
    out_dir = Path("evaluation/focal_cnn")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")

if __name__ == "__main__":
    train_focal()
