"""Simulates the live production telemetry loop using unified quantized ONNX inference."""

import json
import logging
import os
import pickle

import numpy as np
import torch

from src.ETL_data.loader import get_dataloaders
from src.telemetry.model.dtmc_drift import DTMCDataDriftDetector
from src.telemetry.model.mmd_drift import MMDDataDriftDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_live_inference_telemetry() -> None:
    """Simulates the live production telemetry loop.
    
    Runs a test batch through the quantized ONNX model, generating both categorical
    predictions and feature embeddings to compute continuous and sequential drift.
    """
    logger.info("Initializing Quantized Live Production Telemetry Loop (DTMC + MMD)...")
    
    MODEL_VARIANT = "quantized_cnn"
    target_dir = os.path.join("evaluation", MODEL_VARIANT)
    matrix_dir = "src/benchmark/matrices"
    
    model_path = os.path.join(target_dir, "model.pt")
    dtmc_baseline_path = os.path.join(matrix_dir, "dtmc_baseline.pkl")
    mmd_baseline_path = os.path.join(matrix_dir, "mmd_baseline.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing critical production model at {model_path}!")

    logger.info(f"Loading PyTorch model: {model_path}")
    model = torch.jit.load(model_path)
    model.eval()

    dtmc_detector = DTMCDataDriftDetector(num_states=8, threshold=0.50)
    if os.path.exists(dtmc_baseline_path):
        with open(dtmc_baseline_path, "rb") as f:
            dtmc_detector.baseline_matrix = pickle.load(f)
        logger.info(f"Successfully loaded Quantized DTMC Baseline Matrix from {dtmc_baseline_path}")
    else:
        raise FileNotFoundError(f"No DTMC baseline matrix found at {dtmc_baseline_path}!")

    mmd_detector = MMDDataDriftDetector(threshold=0.05)
    if os.path.exists(mmd_baseline_path):
        with open(mmd_baseline_path, "rb") as f:
            mmd_payload = pickle.load(f)
        mmd_detector.fit_baseline(mmd_payload["baseline_embeddings"])
        mmd_detector.gamma = mmd_payload["gamma"]
        logger.info(f"Successfully loaded Quantized MMD Baseline Matrix (Gamma: {mmd_detector.gamma:.6f})")
    else:
        raise FileNotFoundError(f"No MMD baseline matrix found at {mmd_baseline_path}!")

    _, _, test_dl = get_dataloaders(batch_size=32)
    
    logger.info("PyTorch is evaluating holdout wafer maps and extracting parallel features...")
    live_inference_predictions = []
    live_embeddings_list = []
    
    for images, _ in test_dl:
        with torch.no_grad():
            features, logits_matrix = model.forward_with_features(images)
        
        features_np = features.cpu().numpy()
        if features_np.ndim == 1:
            features_np = np.expand_dims(features_np, axis=0)
        live_embeddings_list.append(features_np)

        logits_np = logits_matrix.cpu().numpy()
        if logits_np.ndim == 1:
            logits_np = np.expand_dims(logits_np, axis=0)
            
        predictions = np.argmax(logits_np, axis=1).tolist()
        live_inference_predictions.extend(predictions)

    
    live_embeddings = np.vstack(live_embeddings_list)

    logger.info("Evaluating telemetry metrics across concurrent evaluation engines...")
    dtmc_results = dtmc_detector.monitor_live_window(live_inference_predictions)
    mmd_results = mmd_detector.monitor_live_window(live_embeddings)
    
    live_dtmc_matrix = dtmc_detector.calculate_transition_matrix(live_inference_predictions)
    
    logger.info("=" * 60)
    logger.info("TELEMETRY QUALITY DASHBOARD")
    logger.info(f"DTMC Sequence Drift Score : {dtmc_results['drift_score']:.4f} [Flagged: {dtmc_results['drift_detected']}]")
    logger.info(f"MMD Embedding Drift Score  : {mmd_results['drift_score']:.4f} [Flagged: {mmd_results['drift_detected']}]")
    logger.info("=" * 60)
    
    os.makedirs(matrix_dir, exist_ok=True)

    dtmc_live_path = os.path.join(matrix_dir, "dtmc_live.pkl")
    with open(dtmc_live_path, "wb") as f:
        pickle.dump(live_dtmc_matrix, f)

    mmd_live_path = os.path.join(matrix_dir, "mmd_live.pkl")
    with open(mmd_live_path, "wb") as f:
        pickle.dump(live_embeddings, f)

    cache_path = os.path.join(matrix_dir, "telemetry_state_buffer.json")
    
    total_anomaly_count = 1 if (dtmc_results["drift_detected"] or mmd_results["drift_detected"]) else 0
    
    payload = {
        "last_calculated_anomaly_count": total_anomaly_count,
        "dtmc_drift_score": float(dtmc_results["drift_score"]),
        "dtmc_drift_detected": bool(dtmc_results["drift_detected"]),
        "mmd_drift_score": float(mmd_results["drift_score"]),
        "mmd_drift_detected": bool(mmd_results["drift_detected"]),
        "pipeline_state": "staged_for_prometheus"
    }
    
    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=4)
        
    logger.info(f"Saved live matrices to {matrix_dir}/")
    logger.info(f"Local buffer updated with total anomaly count: {total_anomaly_count}")

if __name__ == "__main__":
    run_live_inference_telemetry()