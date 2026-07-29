"""Unified training pipeline for the Wafer Defect Detection project."""

import logging
from pathlib import Path

from src.training.train_rf_super import train_rf_super
from src.training.train_sssn import train_sssn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BEST_OPTIMIZER = "evaluation/optimizer_benchmark/best_optimizer.json"
BEST_CNN_PARAMS = "evaluation/bayesian_cnn/best_cnn_params.json"
BEST_RF_PARAMS = "evaluation/rf_super/best_rf_params.json"

def train_full_pipeline() -> None:
    """Trains the Super RF and SSSN using pre-computed hyperparameters.

    Loads pre-computed hyperparameter sets to avoid re-running expensive optimization tasks,
    then trains the Super Random Forest and the competitor SSSN.
    """
    logger.info("Initializing Full Unified Training Pipeline...")

    for path in [BEST_OPTIMIZER, BEST_CNN_PARAMS, BEST_RF_PARAMS]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing pre-computed params: {path}. Pipeline requires all optimization configs to exist.")

    logger.info("=== Stage 1: Super Random Forest ===")
    train_rf_super()

    logger.info("=== Stage 2: Spatial-Semantic Stacking Network (SSSN) ===")
    train_sssn()

    logger.info("Full unified training pipeline completed successfully!")

if __name__ == "__main__":
    train_full_pipeline()