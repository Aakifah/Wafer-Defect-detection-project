"""Script to run Bayesian Optimization on the CNN using Optuna."""

import json
import logging
from pathlib import Path

import optuna
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

def run_bayesian_optimization(n_trials: int = 35) -> None:
    """Runs Bayesian Optimization to find optimal hyperparameters.
    
    Uses Optuna to search for the best learning rate, weight decay, and Focal Loss
    gamma value. After finding the best parameters, it trains a final golden model
    from scratch and saves it for downstream compression and deployment.
    
    Args:
        n_trials: The number of optimization trials to run.
    """
    logger.info("Starting Bayesian Optimization with %d trials...", n_trials)

    device = torch.device("cpu")
    train_dl, val_dl, test_dl = get_dataloaders(batch_size=32)
    labels = labels_from_loader(train_dl)
    alpha = compute_balanced_alpha(labels)

    params_file = Path("evaluation/bayesian_cnn/best_cnn_params.json")
    params_file.parent.mkdir(parents=True, exist_ok=True)

    if params_file.exists():
        logger.info("Best CNN params already exist at %s. Skipping search.", params_file)
        with open(params_file) as f:
            best_params = json.load(f)
    else:
        def objective(trial: optuna.Trial) -> float:
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
            gamma = trial.suggest_float("gamma", 1.0, 5.0)

            opt_file = Path("evaluation/optimizer_benchmark/best_optimizer.json")
            opt_name = "adamw"
            if opt_file.exists():
                with open(opt_file) as f:
                    opt_name = json.load(f).get("optimizer", "adamw").lower()

            model = CompleteCNN(num_classes=8).to(device)
            optimizer: optim.Optimizer
            if opt_name == "adamw":
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

            loss_fn = FocalLoss(alpha=alpha, gamma=gamma)

            epochs = 35
            best_val_mAP = -1.0

            for epoch in range(epochs):
                train_one_epoch(model, train_dl, loss_fn, optimizer, device)
                _, val_metrics = evaluate_model(model, val_dl, loss_fn, device)
                val_mAP = val_metrics.get("mAP", 0.0)
                if val_mAP > best_val_mAP:
                    best_val_mAP = val_mAP

            return best_val_mAP

        study = optuna.create_study(
            study_name="bayesian_cnn_optimization",
            storage="sqlite:///optuna_study.db",
            direction="maximize",
            load_if_exists=True,
        )

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        num_completed = len(completed_trials)

        if num_completed >= n_trials:
            logger.info("Found %d completed trials (>= %d). Loading best parameters from study.", num_completed, n_trials)
            best_params = study.best_params
        else:
            remaining_trials = n_trials - num_completed
            logger.info("Found %d completed trials. Running %d more trials.", num_completed, remaining_trials)
            study.optimize(objective, n_trials=remaining_trials)
            best_params = study.best_params

        logger.info("Best hyperparameters: %s", best_params)
        with open(params_file, "w") as f:
            json.dump(best_params, f, indent=4)
        logger.info("Saved best hyperparameters to %s", params_file)

    logger.info("Training final model from scratch with best hyperparameters...")
    best_lr = best_params["lr"]
    best_wd = best_params["weight_decay"]
    best_gamma = best_params["gamma"]

    opt_file = Path("evaluation/optimizer_benchmark/best_optimizer.json")
    opt_name = "adamw"
    if opt_file.exists():
        with open(opt_file) as f:
            opt_name = json.load(f).get("optimizer", "adamw").lower()

    final_model = CompleteCNN(num_classes=8).to(device)
    optimizer: optim.Optimizer
    if opt_name == "adamw":
        optimizer = optim.AdamW(final_model.parameters(), lr=best_lr, weight_decay=best_wd)
    else:
        optimizer = optim.SGD(final_model.parameters(), lr=best_lr, weight_decay=best_wd, momentum=0.9)

    loss_fn = FocalLoss(alpha=alpha, gamma=best_gamma)

    epochs = 35
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        train_one_epoch(final_model, train_dl, loss_fn, optimizer, device)
        val_loss, val_metrics = evaluate_model(final_model, val_dl, loss_fn, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = final_model.state_dict().copy()

    if best_model_state:
        final_model.load_state_dict(best_model_state)

    test_loss, test_metrics = evaluate_model(final_model, test_dl, loss_fn, device)

    metrics = {
        "test_loss": test_loss,
        **test_metrics,
        "epochs_trained": epochs,
        "best_params": best_params,
    }

    save_evaluation_results(final_model, metrics, "evaluation/bayesian_cnn")

    torch.save(final_model.state_dict(), "evaluation/bayesian_cnn/model.pt")

if __name__ == "__main__":
    run_bayesian_optimization(n_trials=35)
