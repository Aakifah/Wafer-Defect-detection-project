"""Script to evaluate and visualize per-class mAP comparisons between models."""

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
import torch
from sklearn.metrics import average_precision_score

from src.ETL_data.loader import get_dataloaders
from src.model.sssn.ensemble import SpatialSemanticStackingNetwork

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFECT_CLASSES = [
    "Center", "Donut", "Edge_Loc", "Edge_Ring",
    "Loc", "Near_Full", "Scratch", "Random",
]


def get_pkl_probs(pkl_path: str, X_test: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Runs inference using a scikit-learn pickle model and returns probabilities."""
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)

    probs_raw = model.predict_proba(X_test)
    if isinstance(probs_raw, list):
        probs = np.column_stack([p[:, 1] for p in probs_raw])
    else:
        probs = probs_raw
    return probs


def get_cnn_probs(pt_path: str, X_test: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Runs inference using a TorchScript CNN model and returns sigmoid probabilities."""
    device = torch.device("cpu")
    model = torch.jit.load(pt_path, map_location=device)
    model.eval()

    all_probs = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = torch.tensor(X_test[i : i + batch_size], dtype=torch.float32)
            if batch.dim() == 3:
                batch = batch.unsqueeze(1)
            logits = model(batch)
            probs = torch.sigmoid(logits).numpy()
            all_probs.append(probs)

    return np.vstack(all_probs)


def get_sssn_probs(cnn_path: str, sssn_dir: str, X_test_tensor: torch.Tensor) -> npt.NDArray[Any]:
    """Runs inference on the SSSN ensemble and returns predictions."""
    cnn_model = torch.jit.load(cnn_path)
    level1a = joblib.load(Path(sssn_dir) / "level1a.pkl")
    level1b = joblib.load(Path(sssn_dir) / "level1b.pkl")
    level2 = joblib.load(Path(sssn_dir) / "level2.pkl")

    sssn_model = SpatialSemanticStackingNetwork(cnn_model, level1a, level1b, level2)

    all_preds = []
    batch_size = 32

    for i in range(0, len(X_test_tensor), batch_size):
        batch = X_test_tensor[i : i + batch_size]
        preds = sssn_model.predict(None, batch).values
        all_preds.append(preds)

    return np.vstack(all_preds)


def plot_comparison(
    y_test: npt.NDArray[Any],
    probs_dict: dict[str, npt.NDArray[Any]],
    title: str,
    filename: str,
    colors: list[str],
) -> None:
    """Plots a dot-line per-class mAP comparison."""
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "axes.edgecolor": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "grid.alpha": 0.5,
    })

    for (label, probs), color in zip(probs_dict.items(), colors):
        class_maps = average_precision_score(y_test, probs, average=None)
        plt.plot(
            DEFECT_CLASSES, class_maps, marker="o", markersize=10, linewidth=2.5,
            label=label, color=color,
        )

    plt.title(title, pad=20, fontweight="bold", fontsize=16, color="#1A1A1A")
    plt.ylabel("mAP Score", fontweight="bold")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=15, fontweight="bold")
    plt.legend(loc="lower left", fontsize=12)
    plt.tight_layout()

    img_dir = Path("images")
    img_dir.mkdir(parents=True, exist_ok=True)

    svg_path = img_dir / f"{filename}.svg"
    png_path = img_dir / f"{filename}.png"
    plt.savefig(svg_path, format="svg", transparent=True, bbox_inches="tight")
    plt.savefig(png_path, format="png", transparent=True, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info("Saved visualization to %s and %s", svg_path, png_path)


def main() -> None:
    logger.info("Loading test data...")
    _, _, test_dl = get_dataloaders(batch_size=32)

    X_list, y_list = [], []
    for X, y in test_dl:
        X_list.append(X.numpy())
        y_list.append(y.numpy())
    X_test_cnn = np.concatenate(X_list)
    y_test = np.concatenate(y_list)
    X_test_rf = X_test_cnn.reshape(X_test_cnn.shape[0], -1)
    X_test_tensor = torch.tensor(X_test_cnn, dtype=torch.float32)

    probs_dict = {}

    cnn_pt_path = "evaluation/quantized_cnn/model.pt"
    rf_super_pkl = "evaluation/rf_super/model.pkl"
    sssn_dir = "evaluation/sssn_competitor"

    if os.path.exists(cnn_pt_path):
        probs_dict["Quantized CNN"] = get_cnn_probs(cnn_pt_path, X_test_cnn)

    if os.path.exists(rf_super_pkl):
        probs_dict["Super RF"] = get_pkl_probs(rf_super_pkl, X_test_rf)

    if os.path.exists(sssn_dir) and os.path.exists(cnn_pt_path):
        logger.info("Running SSSN Inference...")
        probs_dict["SSSN"] = get_sssn_probs(cnn_pt_path, sssn_dir, X_test_tensor)

    if all(m in probs_dict for m in ["Quantized CNN", "Super RF", "SSSN"]):
        plot_comparison(
            y_test,
            {
                "Quantized CNN": probs_dict["Quantized CNN"],
                "Super RF": probs_dict["Super RF"],
                "SSSN": probs_dict["SSSN"],
            },
            "Quantized CNN vs Super RF vs SSSN (Per-Class mAP)",
            "plot_f_all_comparison",
            ["#00B4D8", "#FF007F", "#8A2BE2"],
        )

    if all(m in probs_dict for m in ["Quantized CNN", "SSSN"]):
        plot_comparison(
            y_test,
            {
                "Quantized CNN": probs_dict["Quantized CNN"],
                "SSSN": probs_dict["SSSN"],
            },
            "Quantized CNN vs SSSN (Per-Class mAP)",
            "plot_g_cnn_vs_sssn",
            ["#00B4D8", "#8A2BE2"],
        )

    if "SSSN" in probs_dict:
        plot_comparison(
            y_test,
            {"SSSN": probs_dict["SSSN"]},
            "SSSN Per-Class mAP",
            "plot_h_sssn_only",
            ["#8A2BE2"],
        )


if __name__ == "__main__":
    main()
