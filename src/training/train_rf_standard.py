"""Script to train a standard Random Forest Classifier as a baseline competitor."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score

from src.ETL_data.loader import get_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def train_rf_standard() -> None:
    """Trains a standard Random Forest baseline model and exports to ONNX."""
    logger.info("Starting standard Random Forest baseline training...")
    
    train_dl, val_dl, test_dl = get_dataloaders(batch_size=32)

    def extract_xy(dl: Any) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        X_list: list[npt.NDArray[Any]] = []
        y_list: list[npt.NDArray[Any]] = []
        for X, y in dl:
            X_list.append(X.numpy())
            y_list.append(y.numpy())
        return np.concatenate(X_list), np.concatenate(y_list)

    X_train, y_train = extract_xy(train_dl)
    X_val, y_val = extract_xy(val_dl)
    X_test, y_test = extract_xy(test_dl)
    
    y_train = y_train.astype(np.int64)
    y_val = y_val.astype(np.int64)
    y_test = y_test.astype(np.int64)

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    logger.info(f"Training RandomForest on X_train shape {X_train.shape}...")
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    logger.info("Evaluating standard Random Forest...")
    probs_list = clf.predict_proba(X_test)
    if isinstance(probs_list, list):
        probs = np.column_stack([p[:, 1] for p in probs_list])
    else:
        probs = probs_list

    preds = clf.predict(X_test)
    
    if y_test.ndim > 1 and y_test.shape[1] > 1:
        pass

    macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_test, preds, average="micro", zero_division=0))
    mAP = float(average_precision_score(y_test, probs, average="macro"))
    
    metrics = {
        "test_loss": 0.0,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "mAP": mAP,
        "model_type": "RandomForest_Standard"
    }
    
    logger.info(f"Standard RF Results: mAP={mAP:.4f}, Macro F1={macro_f1:.4f}")

    out_dir = Path("evaluation/rf_standard")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info("Exporting Standard Random Forest to .pkl...")
    model_path = out_dir / "model.pkl"
    with open(model_path, "wb") as f_pkl:
        pickle.dump(clf, f_pkl)
        
    logger.info(f"Successfully exported Standard RF model to {model_path}")

if __name__ == "__main__":
    train_rf_standard()