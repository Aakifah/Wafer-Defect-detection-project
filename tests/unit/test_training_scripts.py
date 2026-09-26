"""Unit tests for the new training scripts."""

from typing import Any
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model.architecture import CompleteCNN
from src.training.utils import evaluate_model


def test_evaluate_model() -> None:
    model = CompleteCNN(num_classes=8)
    loss_fn = nn.BCEWithLogitsLoss()
    
    X = torch.randn(4, 1, 52, 52)
    y = torch.randint(0, 2, (4, 8)).float()
    loader = DataLoader(TensorDataset(X, y), batch_size=2)
    
    loss, metrics = evaluate_model(model, loader, loss_fn, torch.device("cpu"))
    
    assert loss > 0
    assert "micro_f1" in metrics
    assert "macro_f1" in metrics
    assert "mAP" in metrics

@patch("src.training.train_base.get_dataloaders")
@patch("src.training.train_base.save_evaluation_results")
def test_train_base(mock_save: Any, mock_get_dl: Any) -> None:
    X = torch.randn(2, 1, 52, 52)
    y = torch.randint(0, 2, (2, 8)).float()
    loader = DataLoader(TensorDataset(X, y), batch_size=2)
    mock_get_dl.return_value = (loader, loader, loader)
    
    from src.training.train_base import train_base
    
    with patch("src.training.train_base.train_one_epoch", return_value=0.5):
        mock_eval = (0.5, {"micro_f1": 0.5, "macro_f1": 0.5, "mAP": 0.5})
        with patch("src.training.train_base.evaluate_model", return_value=mock_eval):
            train_base()
            
    assert mock_save.called

@patch("src.training.train_focal.get_dataloaders")
@patch("src.training.train_focal.save_evaluation_results")
def test_train_focal(mock_save: Any, mock_get_dl: Any) -> None:
    X = torch.randn(2, 1, 52, 52)
    y = torch.randint(0, 2, (2, 8)).float()
    loader = DataLoader(TensorDataset(X, y), batch_size=2)
    mock_get_dl.return_value = (loader, loader, loader)
    
    from src.training.train_focal import train_focal
    
    with patch("src.training.train_focal.train_one_epoch", return_value=0.5):
        mock_eval = (0.5, {"micro_f1": 0.5, "macro_f1": 0.5, "mAP": 0.5})
        with patch("src.training.train_focal.evaluate_model", return_value=mock_eval):
            train_focal()
            
    assert mock_save.called
