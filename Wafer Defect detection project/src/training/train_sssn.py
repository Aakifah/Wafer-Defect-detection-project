"""Script to train the Spatial-Semantic Stacking Network (SSSN)."""

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import torch
from sklearn.metrics import average_precision_score, f1_score

from src.ETL_data.loader import get_dataloaders
from src.model.sssn.ensemble import SpatialSemanticStackingNetwork
from src.model.sssn.level1_semantic import SemanticLevel1B
from src.model.sssn.level1_spatial import SpatialLevel1A
from src.model.sssn.level2_decider import DeciderLevel2
from src.model.sssn.mlsmote import apply_mlsmote

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

QUANTIZED_MODEL_PATH = "evaluation/quantized_cnn/model.pt"
EVAL_DIR = Path("evaluation/sssn_competitor")


def extract_features(
    model: torch.jit.ScriptModule,
    dataloader: torch.utils.data.DataLoader[Any],
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Extract 512D features, 8D probabilities, and labels from a dataloader.

    Args:
        model: TorchScript model with extract_features and forward methods.
        dataloader: DataLoader yielding (X, y) batches.

    Returns:
        Tuple of (features, probs, labels) as numpy arrays.
    """
    all_features = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            features = model.extract_features(batch_x)
            logits = model(batch_x)
            probs = torch.sigmoid(logits)

            all_features.append(features.numpy())
            all_probs.append(probs.numpy())
            all_labels.append(batch_y.numpy())

    return np.vstack(all_features), np.vstack(all_probs), np.vstack(all_labels)


def train_sssn() -> None:
    """Train the SSSN using the Quantized CNN as a frozen feature extractor.

    Phase 1: Extract 512D features and 8D probabilities from the frozen CNN.
    Phase 2: Oversample minority embeddings with MLSMOTE.
    Phase 3: Train Level 1A (XGBoost) on augmented embeddings and
             Level 1B (MLP) on probability vectors.
    Phase 4: Build the 24D state vector and train Level 2 (Ridge) decider.
    Phase 5: Validate and save all artifacts.
    """
    logger.info("Initializing SSSN Training...")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(QUANTIZED_MODEL_PATH):
        logger.error("Quantized CNN not found. Please ensure train_quantized.py has run successfully.")
        return

    cnn_model = torch.jit.load(QUANTIZED_MODEL_PATH)
    cnn_model.eval()

    train_dl, val_dl, test_dl = get_dataloaders(batch_size=64)

    logger.info("--- Phase 1: Feature Extraction ---")
    X_train_512d, X_train_8d, y_train = extract_features(cnn_model, train_dl)
    logger.info("Features: %s, Probs: %s", X_train_512d.shape, X_train_8d.shape)

    logger.info("--- Phase 2: MLSMOTE ---")
    X_aug, y_aug = apply_mlsmote(X_train_512d, y_train, target_count=500)
    logger.info("Augmented features: %s", X_aug.shape)

    logger.info("--- Phase 3: Level 1 Meta-Learners ---")
    level1a = SpatialLevel1A()
    logger.info("Training Level 1A (XGBoost)...")
    level1a.fit(X_aug, y_aug)

    level1b = SemanticLevel1B()
    logger.info("Training Level 1B (MLP)...")
    level1b.fit(X_train_8d, y_train)

    logger.info("--- Phase 4: Level 2 Decider ---")
    probs_1a = level1a.predict_proba(X_train_512d)
    probs_1b = level1b.predict_proba(X_train_8d)

    state_vector = np.hstack([X_train_8d, probs_1a, probs_1b])
    logger.info("State vector: %s", state_vector.shape)

    level2 = DeciderLevel2()
    logger.info("Training Level 2 (Ridge)...")
    level2.fit(state_vector, y_train)

    logger.info("--- Phase 5: Validation ---")
    X_val_512d, X_val_8d, y_val = extract_features(cnn_model, val_dl)
    val_probs_1a = level1a.predict_proba(X_val_512d)
    val_probs_1b = level1b.predict_proba(X_val_8d)
    val_state_vector = np.hstack([X_val_8d, val_probs_1a, val_probs_1b])
    val_preds = level2.predict(val_state_vector)

    mAP = average_precision_score(y_val, val_preds, average="macro")
    f1 = f1_score(y_val, val_preds, average="macro")
    logger.info("SSSN Validation - mAP: %.4f | F1: %.4f", mAP, f1)

    _ = SpatialSemanticStackingNetwork(cnn_model, level1a, level1b, level2)

    metrics = {"mAP": float(mAP), "macro_f1": float(f1)}
    with open(EVAL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    joblib.dump(level1a, EVAL_DIR / "level1a.pkl")
    joblib.dump(level1b, EVAL_DIR / "level1b.pkl")
    joblib.dump(level2, EVAL_DIR / "level2.pkl")

    logger.info("SSSN training complete. Artifacts in %s", EVAL_DIR)


if __name__ == "__main__":
    train_sssn()
