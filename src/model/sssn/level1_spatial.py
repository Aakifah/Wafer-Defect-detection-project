from typing import Any

import numpy.typing as npt
from sklearn.multioutput import ClassifierChain
from xgboost import XGBClassifier


class SpatialLevel1A:
    """Level 1A Meta-Learner: Spatial Structural Extraction.

    Uses an XGBoost Classifier wrapped in a ClassifierChain to learn
    the target spatial rules from the 512D continuous embeddings.
    """

    def __init__(self, random_state: int = 42) -> None:
        base_xgb = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
        )
        self.model = ClassifierChain(base_xgb, order="random", random_state=random_state)

    def fit(self, X: npt.NDArray[Any], y: npt.NDArray[Any]) -> "SpatialLevel1A":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return self.model.predict_proba(X)  # type: ignore[no-any-return]
