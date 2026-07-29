"""Calibration script to extract baseline image embeddings from PyTorch for MMD."""

import logging
import os
import pickle

import numpy as np
import torch

from src.ETL_data.loader import get_dataloaders
from src.telemetry.model.mmd_drift import MMDDataDriftDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def calibrate_quantized_baseline() -> None:
    """Extracts baseline image embeddings from PyTorch for MMD calibration.
    
    Generates and saves the baseline embeddings profile for future telemetry drift checks.
    """
    logger.info("[MMD] Starting Quantized Maximum Mean Discrepancy (MMD) Baseline Calibration...")

    MODEL_VARIANT = "quantized_cnn"
    target_dir = os.path.join("evaluation", MODEL_VARIANT)
    model_path = os.path.join(target_dir, "model.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing critical production model at {model_path}!")

    logger.info(f"Loading optimization graph: {model_path}")
    
    model = torch.jit.load(model_path)
    model.eval()
    
    train_dl, _, _ = get_dataloaders(batch_size=32)
    
    collected_embeddings = []
    max_baseline_samples = 500

    logger.info("Extracting embedding matrices via PyTorch...")
    
    for images, _ in train_dl:
        with torch.no_grad():
            features = model.extract_features(images)
            
        features_np = features.cpu().numpy()
        if features_np.ndim == 1:
            features_np = np.expand_dims(features_np, axis=0)
            
        collected_embeddings.append(features_np)
        
        current_count = sum(b.shape[0] for b in collected_embeddings)
        if current_count >= max_baseline_samples:
            break

    baseline_matrix = np.vstack(collected_embeddings)[:max_baseline_samples]
    logger.info(f"Generated Baseline Embeddings Shape: {baseline_matrix.shape}")

    detector = MMDDataDriftDetector()
    detector.fit_baseline(baseline_matrix)
    logger.info(f"[MMD] RBF Bandwidth (Gamma) = {detector.gamma:.6f}")

    matrix_dir = "src/benchmark/matrices"
    os.makedirs(matrix_dir, exist_ok=True)
    
    output_path = os.path.join(matrix_dir, "mmd_baseline.pkl")
    
    payload = {
        "baseline_embeddings": detector.baseline_embeddings,
        "gamma": detector.gamma,
        "threshold": detector.threshold
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)
        
    logger.info(f"[MMD] Calibration Successful! Reference profile: {output_path}")

if __name__ == "__main__":
    calibrate_quantized_baseline()