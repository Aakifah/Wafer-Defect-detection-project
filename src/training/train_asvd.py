"""Script to evaluate the ASVD-compressed version of the Bayesian-optimized CNN."""

import json
import logging
from pathlib import Path

import torch
from dotenv import load_dotenv

from src.ETL_data.loader import get_dataloaders
from src.model.architecture import CompleteCNN
from src.model.compression import apply_asvd_to_linear
from src.model.loss import FocalLoss
from src.training.optimizer_benchmark import compute_balanced_alpha, labels_from_loader
from src.training.utils import evaluate_model, save_evaluation_results

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def evaluate_asvd() -> None:
    """Evaluates the ASVD-compressed model.
    
    Loads the pre-trained Bayesian-optimized model, applies ASVD to reduce the
    size of the linear layers, and evaluates the compressed model's performance
    on the test set, logging the results.
    """
    logger.info("Starting ASVD compression evaluation...")
    
    device = torch.device("cpu")
    model = CompleteCNN(num_classes=8).to(device)
    
    bayesian_model_path = Path("evaluation/bayesian_cnn/model.pt")
    
    if not bayesian_model_path.exists():
        logger.error(f"Pre-trained model not found at {bayesian_model_path}")
        return
        
    model.load_state_dict(torch.load(bayesian_model_path, map_location=device, weights_only=True))
    
    apply_asvd_to_linear(model, preserve_energy=0.90)
    
    train_dl, val_dl, test_dl = get_dataloaders(batch_size=32)
    labels = labels_from_loader(train_dl)
    alpha = compute_balanced_alpha(labels)
    
    gamma = 2.0
    bayesian_metrics_path = Path("evaluation/bayesian_cnn/metrics.json")
    if bayesian_metrics_path.exists():
        with open(bayesian_metrics_path) as f:
            b_metrics = json.load(f)
            gamma = b_metrics.get("best_params", {}).get("gamma", 2.0)
            
    loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
    
    test_loss, test_metrics = evaluate_model(model, test_dl, loss_fn, device)
    
    metrics = {
        "test_loss": test_loss,
        **test_metrics,
        "compression": "ASVD (0.90 energy)"
    }
    
    save_evaluation_results(model, metrics, "evaluation/asvd_cnn")

    torch.save(model, Path("evaluation/asvd_cnn") / "model.pt")

if __name__ == "__main__":
    evaluate_asvd()
