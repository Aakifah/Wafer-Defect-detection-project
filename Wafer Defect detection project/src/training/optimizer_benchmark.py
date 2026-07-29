"""Optimizer benchmarking pipeline for CompleteCNN.

Compares AdamW and SGD+Nesterov across multiple learning rates, weight
decay values, and scheduler configurations. Uses Focal Loss with
balanced alpha to handle class imbalance.

Provides utilities for running benchmarks, selecting the best
configuration, and computing test-set evaluation metrics.
"""

import csv
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, StepLR
from torch.utils.data import DataLoader, TensorDataset

from src.model.architecture import CompleteCNN
from src.model.loss import FocalLoss

OptimizerName = Literal["adamw", "sgd_nesterov"]
SchedulerName = Literal["none", "step_lr", "cosine"]
AlphaMode = Literal["scalar", "balanced"]
ModelFactory = Callable[[], nn.Module]
LoaderPair = tuple[DataLoader[tuple[torch.Tensor, ...]], DataLoader[tuple[torch.Tensor, ...]]]


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    optimizer: OptimizerName
    scheduler: SchedulerName = "none"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    step_size: int = 2
    scheduler_gamma: float = 0.5
    cosine_t_max: int | None = None
    focal_gamma: float = 2.0
    alpha_mode: AlphaMode = "balanced"
    scalar_alpha: float = 1.0
    epochs: int = 10


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    optimizer: str
    scheduler: str
    epochs: int
    initial_lr: float
    final_lr: float
    alpha_mode: str
    train_loss: float
    val_loss: float
    best_val_loss: float
    best_epoch: int
    duration_sec: float


def load_npz_dataloaders(
    train_path: str = "data/processed/train_data.npz",
    val_path: str = "data/processed/val_data.npz",
    batch_size: int = 32,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> tuple[DataLoader[tuple[torch.Tensor, ...]], DataLoader[tuple[torch.Tensor, ...]]]:
    train_data = np.load(train_path)
    val_data = np.load(val_path)
    X_train_np, y_train_np = train_data["X"], train_data["y"]
    X_val_np, y_val_np = val_data["X"], val_data["y"]

    if max_train_samples is not None:
        X_train_np, y_train_np = X_train_np[:max_train_samples], y_train_np[:max_train_samples]
    if max_val_samples is not None:
        X_val_np, y_val_np = X_val_np[:max_val_samples], y_val_np[:max_val_samples]

    train_ds = TensorDataset(
        torch.tensor(X_train_np, dtype=torch.float32),
        torch.tensor(y_train_np, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_np, dtype=torch.float32),
        torch.tensor(y_val_np, dtype=torch.float32),
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def labels_from_loader(loader: DataLoader[tuple[torch.Tensor, ...]]) -> torch.Tensor:
    return torch.cat([batch_y for _, batch_y in loader], dim=0)


def compute_balanced_alpha(
    labels: torch.Tensor,
    max_alpha: float = 10.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    if labels.ndim != 2:
        raise ValueError("labels must have shape [num_samples, num_classes]")
    class_freq = labels.float().mean(dim=0)
    alpha = 1.0 / (class_freq + epsilon)
    return torch.clamp(alpha / alpha.mean(), min=epsilon, max=max_alpha).float()


def build_loss(
    config: BenchmarkConfig,
    train_loader: DataLoader[tuple[torch.Tensor, ...]],
) -> nn.Module:
    alpha: float | torch.Tensor
    if config.alpha_mode == "balanced":
        alpha = compute_balanced_alpha(labels_from_loader(train_loader))
    else:
        alpha = config.scalar_alpha
    return FocalLoss(alpha=alpha, gamma=config.focal_gamma)


def build_optimizer(config: BenchmarkConfig, model: nn.Module) -> Optimizer:
    if config.optimizer == "adamw":
        return AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    return SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        nesterov=True,
        weight_decay=config.weight_decay,
    )


def build_scheduler(config: BenchmarkConfig, optimizer: Optimizer) -> LRScheduler | None:
    if config.scheduler == "none":
        return None
    if config.scheduler == "step_lr":
        return StepLR(optimizer, step_size=config.step_size, gamma=config.scheduler_gamma)
    t_max = config.cosine_t_max if config.cosine_t_max is not None else config.epochs
    return CosineAnnealingLR(optimizer, T_max=t_max)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    loss_fn: nn.Module,
    optimizer: Optimizer,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for X, y in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        batch_size = X.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    return total_loss / total_samples


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    loss_fn: nn.Module,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for X, y in loader:
            loss = loss_fn(model(X), y)
            batch_size = X.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    return total_loss / total_samples


def compute_test_metrics(
    model: nn.Module,
    test_loader: DataLoader[tuple[torch.Tensor, ...]],
) -> dict[str, float]:
    """Compute micro F1, macro F1, and mAP on the test set."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in test_loader:
            logits = model(X)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds_np = np.vstack(all_preds)
    targets_np = np.vstack(all_targets)
    preds_binary = (preds_np >= 0.5).astype(int)

    return {
        "test_micro_f1": float(
            f1_score(targets_np, preds_binary, average="micro", zero_division=0)
        ),
        "test_macro_f1": float(
            f1_score(targets_np, preds_binary, average="macro", zero_division=0)
        ),
        "test_mAP": float(
            average_precision_score(targets_np, preds_np, average="macro")
        ),
    }


def run_single_benchmark(
    config: BenchmarkConfig,
    train_loader: DataLoader[tuple[torch.Tensor, ...]],
    val_loader: DataLoader[tuple[torch.Tensor, ...]],
    model_factory: ModelFactory,
    seed: int = 42,
) -> tuple[BenchmarkResult, nn.Module]:
    torch.manual_seed(seed)
    model = model_factory()
    loss_fn = build_loss(config, train_loader)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    best_val_loss = float("inf")
    best_epoch = 0
    train_loss = float("inf")
    val_loss = float("inf")
    start = time.perf_counter()

    print(f"\nRunning benchmark: {config.name}")
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = evaluate_loss(model, val_loader, loss_fn)

        print(
            f"[{config.name}] "
            f"Epoch {epoch}/{config.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        if scheduler is not None:
            scheduler.step()

    result = BenchmarkResult(
        name=config.name,
        optimizer=config.optimizer,
        scheduler=config.scheduler,
        epochs=config.epochs,
        initial_lr=config.lr,
        final_lr=float(optimizer.param_groups[0]["lr"]),
        alpha_mode=config.alpha_mode,
        train_loss=train_loss,
        val_loss=val_loss,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        duration_sec=time.perf_counter() - start,
    )
    return result, model


def sweep_configs(
    optimizers: list[OptimizerName] | None = None,
    schedulers: list[SchedulerName] | None = None,
    learning_rates: list[float] | None = None,
    weight_decays: list[float] | None = None,
    epochs: int = 7,  
    focal_gamma: float = 2.0,
) -> list[BenchmarkConfig]:
    """Generate a grid of BenchmarkConfig for hyperparameter sweep."""
    if optimizers is None:
        optimizers = ["adamw", "sgd_nesterov"]
    if schedulers is None:
        schedulers = ["cosine"]  
    if learning_rates is None:
        learning_rates = [1e-3, 5e-4] 
    if weight_decays is None:
        weight_decays = [1e-2, 1e-4] 

    configs = []
    for opt in optimizers:
        for sched in schedulers:
            for lr in learning_rates:
                for wd in weight_decays:
                    name = f"{opt}_{sched}_lr{lr:.0e}_wd{wd:.0e}"
                    cfg = BenchmarkConfig(
                        name=name,
                        optimizer=opt,
                        scheduler=sched,
                        lr=lr,
                        weight_decay=wd,
                        focal_gamma=focal_gamma,
                        epochs=epochs,
                        cosine_t_max=epochs,
                    )
                    configs.append(cfg)
    return configs


class SweepResult(TypedDict):
    best_config: BenchmarkConfig
    best_result: BenchmarkResult
    best_model: nn.Module
    all_results: list[tuple[BenchmarkConfig, BenchmarkResult, nn.Module]]
    test_metrics: dict[str, float]


def find_best_config(
    train_loader: DataLoader[tuple[torch.Tensor, ...]],
    val_loader: DataLoader[tuple[torch.Tensor, ...]],
    test_loader: DataLoader[tuple[torch.Tensor, ...]] | None = None,
    model_factory: ModelFactory = lambda: CompleteCNN(num_classes=8),
    sweep_params: dict[str, Any] | None = None,
    seed: int = 42,
) -> SweepResult:
    """Run a hyperparameter sweep and return the best configuration.

    Args:
        train_loader: Training data loader.
        val_loader: Validation data loader for selecting the best config.
        test_loader: Optional test loader for computing final metrics.
        model_factory: Callable that returns a fresh model instance.
        sweep_params: Optional dict with keys 'optimizers', 'schedulers',
            'learning_rates', 'weight_decays', 'epochs', 'focal_gamma'.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:
            - best_config: the winning BenchmarkConfig
            - best_result: its BenchmarkResult
            - best_model: the trained nn.Module
            - all_results: list of all (config, result, model) tuples
            - test_metrics: dict if test_loader was provided
    """
    if sweep_params is None:
        sweep_params = {}

    configs = sweep_configs(
        optimizers=sweep_params.get("optimizers"),
        schedulers=sweep_params.get("schedulers"),
        learning_rates=sweep_params.get("learning_rates"),
        weight_decays=sweep_params.get("weight_decays"),
        epochs=sweep_params.get("epochs", 10),
        focal_gamma=sweep_params.get("focal_gamma", 2.0),
    )

    all_results: list[tuple[BenchmarkConfig, BenchmarkResult, nn.Module]] = []

    for config in configs:
        result, model = run_single_benchmark(
            config, train_loader, val_loader, model_factory, seed
        )
        all_results.append((config, result, model))

    best_entry = min(all_results, key=lambda x: x[1].best_val_loss)
    best_config, best_result, best_model = best_entry

    print(f"\nBest config: {best_config.name}")
    print(f"  best_val_loss: {best_result.best_val_loss:.6f} (epoch {best_result.best_epoch})")

    test_metrics: dict[str, float] = {}
    if test_loader is not None:
        test_metrics = compute_test_metrics(best_model, test_loader)
        print(f"  test_micro_f1: {test_metrics['test_micro_f1']:.4f}")
        print(f"  test_macro_f1: {test_metrics['test_macro_f1']:.4f}")
        print(f"  test_mAP:      {test_metrics['test_mAP']:.4f}")

    return {
        "best_config": best_config,
        "best_result": best_result,
        "best_model": best_model,
        "all_results": all_results,
        "test_metrics": test_metrics,
    }


def write_results(results: list[BenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(BenchmarkResult.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def run_benchmarks(
    configs: list[BenchmarkConfig] | None = None,
    loaders: LoaderPair | None = None,
    model_factory: ModelFactory = lambda: CompleteCNN(num_classes=8),
    output_path: Path | None = Path("evaluation/optimizer_benchmark/optimizer_benchmark.csv"),
) -> list[BenchmarkResult]:
    if loaders is None:
        from src.ETL_data.loader import get_dataloaders
        train_loader, val_loader, _ = get_dataloaders()
    else:
        train_loader, val_loader = loaders
        
    results = [
        run_single_benchmark(config, train_loader, val_loader, model_factory)[0]
        for config in (configs or default_configs())
    ]
    if output_path is not None:
        write_results(results, output_path)
    return results


def default_configs() -> list[BenchmarkConfig]:
    return [
        BenchmarkConfig(name="adamw_baseline", optimizer="adamw"),
        BenchmarkConfig(name="sgd_nesterov_step", optimizer="sgd_nesterov", scheduler="step_lr"),
        BenchmarkConfig(name="sgd_nesterov_cosine", optimizer="sgd_nesterov", scheduler="cosine"),
    ]


if __name__ == "__main__":
    from src.ETL_data.loader import get_dataloaders
    train_loader, val_loader, _ = get_dataloaders()
    
    res = find_best_config(
        train_loader=train_loader,
        val_loader=val_loader,
        sweep_params={
            "epochs": 10,
        }
    )
    
    best_config_dict = asdict(res["best_config"])
    output_json = Path("evaluation/optimizer_benchmark/best_optimizer.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(best_config_dict, f, indent=4)
    print(f"Saved best config to {output_json}")
