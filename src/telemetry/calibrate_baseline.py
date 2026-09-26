"""Telemetry Calibration Utility to generate dtmc_baseline.pkl files."""

import argparse
import logging
import os
import pickle

import numpy as np
import torch

from src.ETL_data.loader import get_dataloaders
from src.telemetry.model.dtmc_drift import DTMCDataDriftDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def calibrate_model_folder(model_dir_name: str) -> None:
    """Calibrates the telemetry baseline transition matrix for a given model directory.

    Args:
        model_dir_name: The directory name containing the PyTorch model in evaluation/.
    """
    target_dir = os.path.join("evaluation", model_dir_name)
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Directory {target_dir} does not exist!")

    logger.info(f"Commencing Telemetry Calibration for Variant: {model_dir_name}")
    model_path = os.path.join(target_dir, "model.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model at {model_path}!")

    model = torch.jit.load(model_path)
    model.eval()
    logger.info(f"Loaded PyTorch Model from {model_path}")
    
    _, val_dl, _ = get_dataloaders(batch_size=32)
    validation_predictions = []
    
    for images, _ in val_dl:
        with torch.no_grad():
            logits = model(images)
            
        logits_np = logits.cpu().numpy()
        if logits_np.ndim == 1:
            logits_np = np.expand_dims(logits_np, axis=0)
        preds = np.argmax(logits_np, axis=1).tolist()
        
        validation_predictions.extend(preds)
        
    detector = DTMCDataDriftDetector(num_states=8, threshold=0.50)
    detector.fit_baseline(validation_predictions)
    
    logger.info("LOCKED BASELINE TELEMETRY MATRIX (Probability Grid)")
    logger.info("=" * 55)
    assert detector.baseline_matrix is not None
    logger.info(np.array2string(detector.baseline_matrix, precision=3, suppress_small=True))
    logger.info("=" * 55)
    
    os.makedirs("src/benchmark/matrices", exist_ok=True)

    output_path = os.path.join("src/benchmark/matrices", "dtmc_baseline.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(detector.baseline_matrix, f)
        
    logger.info(f"Telemetry calibration complete! Baseline matrix saved to: {output_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Telemetry Baseline Calibration")
    parser.add_argument("--model", type=str, default="quantized_cnn", help="The folder name under evaluation/")
    args = parser.parse_args()
    calibrate_model_folder(args.model)