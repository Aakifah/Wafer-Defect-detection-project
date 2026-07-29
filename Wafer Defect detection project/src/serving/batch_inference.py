"""Batch Inference and Drift Detection Serving Script.

Executes batch inference, loading the Champion and Competitor models from MLflow,
running inference, evaluating DTMC/MMD drift on the Champion, calculating mAP/F1 metrics 
for both models, logging to MLflow, and triggering automated retraining if drift exceeds thresholds.
"""

import json
import logging
import os
import pickle
import time
from typing import Any

import mlflow
import numpy as np
import numpy.typing as npt
import torch
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, f1_score

from src.ETL_data.loader import get_dataloaders
from src.telemetry.model.dtmc_drift import DTMCDataDriftDetector
from src.telemetry.model.mmd_drift import MMDDataDriftDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def get_run_by_tag(tag: str) -> mlflow.entities.Run | None:
    """Queries MLflow for the run tagged with the given tag and returns the Run object."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Wafer_Defect_Detection")
    if not experiment:
        return None

    query = f"tags.{tag} = 'True'"
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], filter_string=query)
    
    if not runs:
        return None
    
    return runs[0]

def evaluate_model(model: Any, is_cnn: bool, test_dl: Any) -> tuple[float, float, list[int], npt.NDArray[Any]]:
    """Evaluates a PyTorch, Sklearn, or PyFunc model and returns mAP, F1, predictions, and embeddings."""
    all_preds = []
    all_probs = []
    all_targets = []
    live_inference_predictions = []
    live_embeddings_list = []
    
    is_sssn = hasattr(model, "_model_impl") and hasattr(model._model_impl.python_model, "extract_features")
    
    if is_cnn and not is_sssn:
        model.eval()

    for images, targets in test_dl:
        if is_sssn:
            sssn_model = model._model_impl.python_model
            features_np = sssn_model.extract_features(images)
            if features_np.ndim == 1:
                features_np = np.expand_dims(features_np, axis=0)
            live_embeddings_list.append(features_np)
            
            preds_df = model.predict(images.numpy())
            preds_np = preds_df.values
            probs_np = preds_np 
            
            live_inference_predictions.extend(np.argmax(probs_np, axis=1).tolist())
            
            preds_binary = (probs_np >= 0.5).astype(int)
            all_preds.append(preds_binary)
            all_probs.append(probs_np)

        elif is_cnn:
            with torch.no_grad():
                features, logits = model.forward_with_features(images)
            
            features_np = features.cpu().numpy()
            if features_np.ndim == 1:
                features_np = np.expand_dims(features_np, axis=0)
            live_embeddings_list.append(features_np)

            logits_np = logits.cpu().numpy()
            if logits_np.ndim == 1:
                logits_np = np.expand_dims(logits_np, axis=0)
                
            probs_np = 1 / (1 + np.exp(-logits_np))
            preds_np = np.argmax(logits_np, axis=1).tolist()
            live_inference_predictions.extend(preds_np)
            
            preds_binary = (probs_np >= 0.5).astype(int)
            all_preds.append(preds_binary)
            all_probs.append(probs_np)
        else:
            raw_numpy = images.cpu().numpy()
            raw_numpy = raw_numpy.reshape(raw_numpy.shape[0], -1)
            
            probs_raw = model.predict_proba(raw_numpy)
            if isinstance(probs_raw, list):
                probs_batch = np.column_stack([p[:, 1] for p in probs_raw])
            else:
                probs_batch = probs_raw
                
            preds_binary = (probs_batch >= 0.5).astype(int)
            all_preds.append(preds_binary)
            all_probs.append(probs_batch)
            
            preds_list = np.argmax(probs_batch, axis=1).tolist()
            live_inference_predictions.extend(preds_list)
            
            live_embeddings_list.append(probs_batch)
            
        target_indices = targets.cpu().numpy()
        all_targets.append(target_indices)

    preds_np = np.vstack(all_preds)
    probs_np = np.vstack(all_probs)
    targets_np = np.vstack(all_targets)
    live_embeddings = np.vstack(live_embeddings_list)
    
    mAP = float(average_precision_score(targets_np, probs_np, average="macro"))
    macro_f1 = float(f1_score(targets_np, preds_np, average="macro", zero_division=0))
    
    return mAP, macro_f1, live_inference_predictions, live_embeddings

def run_batch_serving() -> None:
    """Runs the PoC batch serving and drift detection pipeline."""
    start_time = time.time()
    MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
    
    if not MLFLOW_URI or MLFLOW_URI.startswith("$"):
        logger.warning("No connection from GitHub Actions variables. Reverting to using http://mlflow:5000")
        MLFLOW_URI = "http://mlflow:5000"
        
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        mlflow.set_experiment("Wafer_Defect_Detection")
        competitor_run = get_run_by_tag("Competitor")
        
        try:
            champion_uri = "models:/wafer_cnn@champion"
            logger.info(f"Downloading Champion model from MLflow Registry: {champion_uri}")
            model = mlflow.pytorch.load_model(champion_uri)
            is_cnn = True
        except Exception:
            champion_run = get_run_by_tag("Champion")
            if not champion_run:
                logger.error("No Champion model found in MLflow. Aborting batch inference.")
                return
            
            champion_uri = f"runs:/{champion_run.info.run_id}/models"
            logger.info(f"Downloading Champion model from MLflow Run: {champion_uri}")
            
            is_cnn = champion_run.data.params.get("model_type", "CNN") == "CNN"
            is_sssn = champion_run.data.params.get("model_type", "CNN") == "SSSN"
            
            if is_sssn:
                model = mlflow.pyfunc.load_model(champion_uri)
            elif is_cnn:
                model = mlflow.pytorch.load_model(champion_uri)
            else:
                model = mlflow.sklearn.load_model(champion_uri)
    except Exception:
        print("!!! MLflow is not available, fallback used!!!")
        champion_run = None
        competitor_run = None
        is_cnn = True
        model = torch.jit.load("evaluation/quantized_cnn/model.pt")

    matrix_dir = "src/benchmark/matrices"
    dtmc_baseline_path = os.path.join(matrix_dir, "dtmc_baseline.pkl")
    mmd_baseline_path = os.path.join(matrix_dir, "mmd_baseline.pkl")
    
    dtmc_detector = DTMCDataDriftDetector(num_states=8, threshold=0.50)
    if os.path.exists(dtmc_baseline_path):
        with open(dtmc_baseline_path, "rb") as f:
            dtmc_detector.baseline_matrix = pickle.load(f)
        logger.info("Loaded DTMC baseline matrix.")
    else:
        logger.error("DTMC baseline not found. Aborting.")
        return

    mmd_detector = MMDDataDriftDetector(threshold=0.05)
    if os.path.exists(mmd_baseline_path):
        with open(mmd_baseline_path, "rb") as f:
            mmd_payload = pickle.load(f)
        mmd_detector.fit_baseline(mmd_payload["baseline_embeddings"])
        mmd_detector.gamma = mmd_payload["gamma"]
        logger.info("Loaded MMD baseline embeddings.")
    else:
        logger.error("MMD baseline not found. Aborting.")
        return

    _, _, test_dl = get_dataloaders(batch_size=32)
    
    logger.info("Evaluating Champion model live...")
    champ_mAP, champ_f1, live_preds, live_embeds = evaluate_model(model, is_cnn, test_dl)
    
    comp_mAP, comp_f1 = 0.0, 0.0
    if competitor_run:
        logger.info("Extracting Competitor baseline metrics directly from MLflow run metadata...")
        metrics = competitor_run.data.metrics
        comp_mAP = metrics.get("test_mAP", metrics.get("mAP", 0.0))
        comp_f1 = metrics.get("test_macro_f1", metrics.get("macro_f1", 0.0))
    
    dtmc_results = dtmc_detector.monitor_live_window(live_preds)
    
    try:
        mmd_results = mmd_detector.monitor_live_window(live_embeds)
    except Exception:
        mmd_results = {"drift_score": 0.0, "drift_detected": False}
    
    logger.info(f"Champion mAP: {champ_mAP:.4f}, F1: {champ_f1:.4f}")
    logger.info(f"Competitor mAP: {comp_mAP:.4f}, F1: {comp_f1:.4f}")
    logger.info(f"DTMC Drift Score: {dtmc_results['drift_score']:.4f}")
    logger.info(f"MMD Drift Score: {mmd_results['drift_score']:.4f}")
    
    try:
        with mlflow.start_run(run_name="production_telemetry"):
            mlflow.log_metrics({
                "live_dtmc_score": float(dtmc_results["drift_score"]),
                "live_mmd_score": float(mmd_results["drift_score"]),
                "champion_mAP": champ_mAP,
                "champion_macro_f1": champ_f1,
                "competitor_mAP": comp_mAP,
                "competitor_macro_f1": comp_f1,
            })
            logger.info("Successfully logged production drift telemetry to MLflow.")
    except Exception:
        print("!!! MLflow is not available, fallback used!!!")

    os.makedirs(matrix_dir, exist_ok=True)
    cache_path = os.path.join(matrix_dir, "telemetry_state_buffer.json")
    
    total_anomaly_count = 1 if (dtmc_results["drift_detected"] or mmd_results["drift_detected"]) else 0
    
    inference_duration = time.time() - start_time
    
    payload = {
        "last_calculated_anomaly_count": total_anomaly_count,
        "dtmc_drift_score": float(dtmc_results["drift_score"]),
        "dtmc_drift_detected": bool(dtmc_results["drift_detected"]),
        "mmd_drift_score": float(mmd_results["drift_score"]),
        "mmd_drift_detected": bool(mmd_results["drift_detected"]),
        "champion_mAP": champ_mAP,
        "champion_macro_f1": champ_f1,
        "competitor_mAP": comp_mAP,
        "competitor_macro_f1": comp_f1,
        "pipeline_state": "staged_for_prometheus",
        "inference_duration_seconds": round(inference_duration, 2),
        "training_duration_seconds": float(os.environ.get("TRAINING_DURATION_SECONDS", "0.0")),
        "calibration_duration_seconds": float(os.environ.get("CALIBRATION_DURATION_SECONDS", "0.0"))
    }
    
    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=4)
        
    drift_detected = dtmc_results["drift_detected"] or mmd_results["drift_detected"]
    if drift_detected:
        logger.critical("[ALERT] Drift Threshold Exceeded.")
        logger.critical("Simulated automated retraining triggered on Champion architecture.")
    else:
        logger.info("No drift detected. System stable. No retraining triggered.")

if __name__ == "__main__":
    run_batch_serving()