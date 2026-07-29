from typing import Any

import numpy.typing as npt
from sklearn.linear_model import RidgeClassifier


class DeciderLevel2:
    """Level 2 Decider: Linear Error Correction.

    Evaluates the combined 24D state vector (Level 0 + Level 1A + Level 1B)
    to produce the final 8D multi-label predictions.
    """
    def __init__(self, random_state: int = 42) -> None:
        self.model = RidgeClassifier(alpha=1.0, random_state=random_state)
        
    def fit(self, X: npt.NDArray[Any], y: npt.NDArray[Any]) -> "DeciderLevel2":
        self.model.fit(X, y)
        return self
        
    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return self.model.predict(X) # type: ignore
