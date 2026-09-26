from typing import Any

import numpy.typing as npt
from sklearn.neural_network import MLPClassifier


class SemanticLevel1B:
    """Level 1B Meta-Learner: Semantic Probability Calibration.

    Uses a Multi-Layer Perceptron (MLP) to learn non-linear decision
    thresholds directly from the raw 8D Level 0 probabilities.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.model = MLPClassifier(
            hidden_layer_sizes=(192, 96),
            activation="relu",
            solver="adam",
            max_iter=1500,
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=20,
        )

    def fit(self, X: npt.NDArray[Any], y: npt.NDArray[Any]) -> "SemanticLevel1B":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return self.model.predict_proba(X)  # type: ignore[no-any-return]
