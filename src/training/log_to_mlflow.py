import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import mlflow
import torch
from mlflow.tracking import MlflowClient

from src.model.architecture import CompleteCNN
from src.model.sssn.ensemble import SpatialSemanticStackingNetwork

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DESCRIPTIONS = {
    "base_cnn": "Base CNN trained with BCEWithLogitsLoss and default optimizer",
    "focal_cnn": "CNN trained with Focal Loss gamma=2.0 and class-balanced alpha",
    "optimized_cnn": "CNN trained with best optimizer from benchmark and Focal Loss",
    "bayesian_cnn": "CNN trained with Bayesian-optimized hyperparameters",
    "asvd_cnn": "ASVD-compressed CNN preserving 90% singular value energy",
    "quantized_cnn": "INT8 Post-Training Static Quantized CNN - Production Champion",
    "rf_standard": "Standard Random Forest baseline (100 estimators)",
    "rf_super": "Bayesian-optimized ExtraTrees with CCP pruning",
    "sssn_competitor": "Spatial-Semantic Stacking Network (SSSN) - Competitor"
}

def log_all_stages() -> None:
    MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
    
    if not MLFLOW_URI or MLFLOW_URI.startswith("$"):
        logger.warning("No connection from GitHub Actions variables. Reverting to using http://mlflow:5000")
        MLFLOW_URI = "http://mlflow:5000"
        
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        mlflow.set_experiment("Wafer_Defect_Detection")
    except Exception:
        logger.error("!!! MLflow is not available, fallback used!!!")
        return
    
    if MLFLOW_URI == "http://mlflow:5000":
        logger.info("http://mlflow:5000 is the correct connection endpoint!")

    eval_dir = Path("evaluation")
    if not eval_dir.exists():
        logger.warning("No evaluation directory found. Skipping MLflow logging.")
        return

    client = MlflowClient()

    cnn_order = ["base_cnn", "focal_cnn", "optimized_cnn", "bayesian_cnn", "asvd_cnn", "quantized_cnn"]
    rf_order = ["rf_standard", "rf_super"]
    sssn_order = ["sssn_competitor"]

    for stage_list, model_name in [(cnn_order, "wafer_cnn"), (rf_order, "wafer_rf"), (sssn_order, "wafer_sssn")]:
        for stage in stage_list:
            sub_dir = eval_dir / stage
            if not sub_dir.is_dir():
                continue
                
            metrics_path = sub_dir / "metrics.json"
            model_pt = sub_dir / "model.pt"
            model_pkl = sub_dir / "model.pkl"

            if metrics_path.exists():
                logger.info(f"Logging metrics and artifacts for {stage}...")
                
                try:
                    with open(metrics_path) as f:
                        metrics = json.load(f)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping {stage}: malformed metrics.json — {e}")
                    continue
                
                with mlflow.start_run(run_name=stage) as run:
                    metrics_to_log = {}
                    params_to_log: dict[str, Any] = {"model_type": metrics.get("model_type", "CNN")}
                    
                    if "best_params" in metrics:
                        params_to_log.update(metrics.pop("best_params"))
                        
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            metrics_to_log[key] = value
                        else:
                            params_to_log[key] = str(value)
                            
                    if params_to_log:
                        mlflow.log_params(params_to_log)
                    if metrics_to_log:
                        mlflow.log_metrics(metrics_to_log)

                    version = None
                    
                    if stage == "sssn_competitor":
                        sssn_model = SpatialSemanticStackingNetwork()
                        cnn_pt_path = Path("evaluation/quantized_cnn/model.pt").resolve()

                        mlflow.pyfunc.log_model(
                            stage,
                            python_model=sssn_model,
                            artifacts={
                                "cnn_model": str(cnn_pt_path),
                                "level1a": str((sub_dir / "level1a.pkl").resolve()),
                                "level1b": str((sub_dir / "level1b.pkl").resolve()),
                                "level2": str((sub_dir / "level2.pkl").resolve()),
                            },
                        )
                        result = mlflow.register_model(f"runs:/{run.info.run_id}/{stage}", model_name)
                        version = result.version

                    elif model_pkl.exists():
                        try:
                            with open(model_pkl, "rb") as f:
                                model_loaded = pickle.load(f)
                            mlflow.sklearn.log_model(model_loaded, stage)
                            result = mlflow.register_model(f"runs:/{run.info.run_id}/{stage}", model_name)
                            version = result.version
                        except Exception as e:
                            logger.error(f"Failed to log sklearn model {model_pkl}: {e}")
                            mlflow.log_artifact(str(model_pkl), artifact_path="models_raw")

                    if model_pt.exists():
                        try:
                            if stage == "quantized_cnn":
                                model_loaded = torch.jit.load(model_pt)
                            elif stage == "asvd_cnn":
                                model_loaded = torch.load(model_pt, weights_only=False)
                            else:
                                model_loaded = CompleteCNN(num_classes=8)
                                model_loaded.load_state_dict(torch.load(model_pt))
                            
                            mlflow.pytorch.log_model(model_loaded, stage)
                            result = mlflow.register_model(f"runs:/{run.info.run_id}/{stage}", model_name)
                            version = result.version
                        except Exception as e:
                            logger.error(f"Failed to log pytorch model {model_pt}: {e}")
                            mlflow.log_artifact(str(model_pt), artifact_path="models_raw")

                    if version:
                        if str(version) == "1":
                            framework = "pytorch" if model_name == "wafer_cnn" else "sklearn"
                            client.update_registered_model(
                                name=model_name,
                                description=f"Model family for {model_name}"
                            )
                            client.set_registered_model_tag(model_name, "framework", framework)
                            client.set_registered_model_tag(model_name, "task", "multi_label_classification")
                        
                        client.set_model_version_tag(model_name, version, "pipeline_stage", stage)
                        if "compression" in metrics:
                            client.set_model_version_tag(model_name, version, "compression", str(metrics["compression"]))
                        if "loss_fn" in metrics:
                            client.set_model_version_tag(model_name, version, "loss_fn", str(metrics["loss_fn"]))
                            
                        client.update_model_version(name=model_name, version=version, description=DESCRIPTIONS.get(stage, stage))
                        
                        if stage == "quantized_cnn":
                            client.set_registered_model_alias(model_name, "champion", version)
                        elif stage == "sssn_competitor":
                            client.set_registered_model_alias(model_name, "competitor", version)

if __name__ == "__main__":
    log_all_stages()