from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, TensorDataset

from src.model.architecture import CompleteCNN
from src.model.loss import FocalLoss
from src.training.optimizer_benchmark import (
    BenchmarkConfig,
    LoaderPair,
    build_optimizer,
    build_scheduler,
    compute_balanced_alpha,
    default_configs,
    evaluate_loss,
    run_benchmarks,
    train_one_epoch,
)


def make_loaders() -> LoaderPair:
    X = torch.randn(12, 4)
    y = torch.randint(0, 2, (12, 2)).float()
    loader = DataLoader(TensorDataset(X, y), batch_size=4)
    return loader, loader


def make_model() -> nn.Module:
    return nn.Linear(4, 2)


def test_compute_balanced_alpha_weights_rare_class_more() -> None:
    labels = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0]])

    alpha = compute_balanced_alpha(labels)

    assert alpha.shape == (2,)
    assert alpha[1] > alpha[0]


def test_compute_balanced_alpha_rejects_non_matrix_labels() -> None:
    labels = torch.tensor([1.0, 0.0, 1.0])

    with pytest.raises(ValueError, match="shape"):
        compute_balanced_alpha(labels)


def test_build_optimizer_adamw() -> None:
    config = BenchmarkConfig(name="adamw", optimizer="adamw")
    optimizer = build_optimizer(config, make_model())

    assert isinstance(optimizer, AdamW)
    assert optimizer.param_groups[0]["weight_decay"] == config.weight_decay


def test_build_optimizer_sgd_uses_nesterov() -> None:
    config = BenchmarkConfig(name="sgd", optimizer="sgd_nesterov", momentum=0.9)
    optimizer = build_optimizer(config, make_model())

    assert isinstance(optimizer, SGD)
    assert optimizer.param_groups[0]["nesterov"] is True
    assert optimizer.param_groups[0]["momentum"] == config.momentum


def test_build_schedulers() -> None:
    model = make_model()
    step_optimizer = SGD(model.parameters(), lr=0.01)
    cosine_optimizer = SGD(model.parameters(), lr=0.01)

    step = build_scheduler(
        BenchmarkConfig(name="step", optimizer="sgd_nesterov", scheduler="step_lr"),
        step_optimizer,
    )
    cosine = build_scheduler(
        BenchmarkConfig(name="cosine", optimizer="sgd_nesterov", scheduler="cosine"),
        cosine_optimizer,
    )

    assert isinstance(step, StepLR)
    assert isinstance(cosine, CosineAnnealingLR)


def test_build_scheduler_none() -> None:
    optimizer = SGD(make_model().parameters(), lr=0.01)

    scheduler = build_scheduler(BenchmarkConfig(name="none", optimizer="adamw"), optimizer)

    assert scheduler is None


def test_train_and_evaluate_align_with_complete_cnn_and_focal_loss() -> None:
    X = torch.randn(4, 1, 52, 52)
    y = torch.randint(0, 2, (4, 8)).float()
    loader = DataLoader(TensorDataset(X, y), batch_size=2)
    model = CompleteCNN(num_classes=8)
    config = BenchmarkConfig(name="complete", optimizer="adamw", alpha_mode="scalar")
    optimizer = build_optimizer(config, model)
    loss_fn = FocalLoss()

    train_loss = train_one_epoch(model, loader, loss_fn, optimizer)
    val_loss = evaluate_loss(model, loader, loss_fn)

    assert train_loss >= 0
    assert val_loss >= 0


def test_default_configs_cover_required_optimizer_paths() -> None:
    configs = default_configs()

    assert {config.optimizer for config in configs} == {"adamw", "sgd_nesterov"}
    assert {config.scheduler for config in configs} == {"none", "step_lr", "cosine"}


def test_run_benchmarks_writes_results(tmp_path: Path) -> None:
    config = BenchmarkConfig(
        name="tiny_adamw",
        optimizer="adamw",
        alpha_mode="scalar",
        epochs=1,
    )
    output_path = tmp_path / "benchmark.csv"

    results = run_benchmarks(
        configs=[config],
        loaders=make_loaders(),
        model_factory=make_model,
        output_path=output_path,
    )

    assert len(results) == 1
    assert results[0].train_loss >= 0
    assert results[0].val_loss >= 0
    assert output_path.exists()


def test_run_benchmarks_aligns_complete_cnn_with_focal_loss() -> None:
    X = torch.randn(2, 1, 52, 52)
    y = torch.randint(0, 2, (2, 8)).float()
    loader = DataLoader(TensorDataset(X, y), batch_size=1)
    config = BenchmarkConfig(
        name="complete_cnn_contract",
        optimizer="adamw",
        alpha_mode="scalar",
        epochs=1,
    )

    results = run_benchmarks(
        configs=[config],
        loaders=(loader, loader),
        model_factory=lambda: CompleteCNN(num_classes=8),
        output_path=None,
    )

    assert len(results) == 1
    assert results[0].train_loss >= 0
    assert results[0].val_loss >= 0
