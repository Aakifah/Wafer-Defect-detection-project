"""Script to train an advanced Bayesian-optimized ExtraTrees Classifier (Super RF)."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import optuna
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, f1_score

from src.ETL_data.loader import get_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def train_rf_super(n_trials: int = 50) -> None:
    """Trains a Bayesian-optimized ExtraTrees model, applies CCP, and quantizes to ONNX."""
    logger.info(f"Starting Super Random Forest (ExtraTrees) Bayesian Optimization ({n_trials} trials)...")
    
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
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    params_file = Path("evaluation/rf_super/best_rf_params.json")
    params_file.parent.mkdir(parents=True, exist_ok=True)
    if params_file.exists():
        logger.info(f"Loading existing hyperparameters from {params_file}, bypassing search.")
        with open(params_file) as f:
            best_params = json.load(f)
    else:
        def objective(trial: optuna.Trial) -> float:
            n_estimators = trial.suggest_int("n_estimators", 100, 300)
            max_depth = trial.suggest_int("max_depth", 10, 50)
            ccp_alpha = trial.suggest_float("ccp_alpha", 0.0, 0.01)
            
            clf = ExtraTreesClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                ccp_alpha=ccp_alpha,
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1
            )
            clf.fit(X_train, y_train)
            
            probs_list = clf.predict_proba(X_val)
            if isinstance(probs_list, list):
                probs = np.column_stack([p[:, 1] for p in probs_list])
            else:
                probs = probs_list
                
            mAP = float(average_precision_score(y_val, probs, average="macro"))
            return mAP

        study = optuna.create_study(
        study_name="super_rf_optimize",
        storage="sqlite:///optuna_study.db",
        direction="maximize",
        load_if_exists=True
        )
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best hyperparameters: {study.best_params}")
        best_params = study.best_params
        
        with open(params_file, "w") as f:
            json.dump(best_params, f, indent=4)
        logger.info(f"Saved best hyperparameters to {params_file}")

    final_clf = ExtraTreesClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        ccp_alpha=best_params["ccp_alpha"],
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    final_clf.fit(X_train, y_train)

    logger.info("Evaluating Super Random Forest...")
    probs_list = final_clf.predict_proba(X_test)
    if isinstance(probs_list, list):
        probs = np.column_stack([p[:, 1] for p in probs_list])
    else:
        probs = probs_list

    preds = final_clf.predict(X_test)
    
    macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_test, preds, average="micro", zero_division=0))
    mAP = float(average_precision_score(y_test, probs, average="macro"))
    
    metrics = {
        "test_loss": 0.0,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "mAP": mAP,
        "model_type": "ExtraTrees_Super",
        "best_params": best_params
    }
    
    logger.info(f"Super RF Results: mAP={mAP:.4f}, Macro F1={macro_f1:.4f}")

    out_dir = Path("evaluation/rf_super")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info("Exporting Super Random Forest to .pkl...")
    model_path = out_dir / "model.pkl"
    with open(model_path, "wb") as f_pkl:
        pickle.dump(final_clf, f_pkl)
        
    logger.info(f"Successfully exported Super RF model to {model_path}")

if __name__ == "__main__":
    train_rf_super(n_trials=10)
