"""Utility functions for model training and evaluation."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score
from torch.optim import Optimizer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    """Trains the model for one epoch.
    
    Iterates over the provided DataLoader, performing forward and backward passes,
    and updates the model weights using the specified optimizer.
    
    Args:
        model: The neural network model to train.
        loader: DataLoader providing the training data.
        loss_fn: The loss function to optimize.
        optimizer: The optimizer used to update the model parameters.
        device: The device (CPU/GPU) to run the training on.
        
    Returns:
        The average loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        
        batch_size = X.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / total_samples

def evaluate_model(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Evaluates the model on the provided data.
    
    Computes the loss and various classification metrics including Micro F1,
    Macro F1, and mean Average Precision (mAP) for the multi-label task.
    
    Args:
        model: The neural network model to evaluate.
        loader: DataLoader providing the evaluation data.
        loss_fn: The loss function to compute evaluation loss.
        device: The device (CPU/GPU) to run the evaluation on.
        
    Returns:
        A tuple containing the average loss and a dictionary of computed metrics.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            
            batch_size = X.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds_np = np.vstack(all_preds)
    targets_np = np.vstack(all_targets)
    preds_binary = (preds_np >= 0.5).astype(int)

    metrics = {
        "micro_f1": float(f1_score(targets_np, preds_binary, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(targets_np, preds_binary, average="macro", zero_division=0)),
        "mAP": float(average_precision_score(targets_np, preds_np, average="macro"))
    }
    
    return total_loss / total_samples, metrics

def save_evaluation_results(
    model: nn.Module,
    metrics: dict[str, Any],
    save_dir: str,
    model_name: str = "model.pt",
    metrics_name: str = "metrics.json"
) -> None:
    """Saves the trained model and its evaluation metrics to disk.
    
    Creates the target directory if it doesn't exist, saves the PyTorch state_dict,
    and writes the metrics to a JSON file.
    
    Args:
        model: The trained neural network model.
        metrics: Dictionary containing the evaluation metrics.
        save_dir: The directory where artifacts should be saved.
        model_name: The filename for the model weights.
        metrics_name: The filename for the metrics JSON.
    """
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / model_name
    metrics_path = output_dir / metrics_name
    
    torch.save(model.state_dict(), model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info("Saved model and metrics to %s", output_dir)
