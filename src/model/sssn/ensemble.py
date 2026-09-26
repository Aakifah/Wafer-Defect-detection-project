import logging
from typing import Any

import joblib
import mlflow.pyfunc
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

logger = logging.getLogger(__name__)

class SpatialSemanticStackingNetwork(mlflow.pyfunc.PythonModel): # type: ignore
    """Unified SSSN MLflow PyFunc Wrapper.
    
    Bundles the Quantized CNN Backbone with Level 1A (XGBoost), 
    Level 1B (MLP), and Level 2 (Ridge) meta-learners.
    """
    def __init__(
        self,
        cnn_model: Any = None,
        level1a: Any = None,
        level1b: Any = None,
        level2: Any = None,
    ) -> None:
        self.cnn_model = cnn_model
        self.level1a = level1a
        self.level1b = level1b
        self.level2 = level2

    def load_context(self, context: Any) -> None:
        """Load artifacts from MLflow model context."""
        if self.cnn_model is None:
            self.cnn_model = torch.jit.load(context.artifacts["cnn_model"])
        if self.level1a is None:
            self.level1a = joblib.load(context.artifacts["level1a"])
        if self.level1b is None:
            self.level1b = joblib.load(context.artifacts["level1b"])
        if self.level2 is None:
            self.level2 = joblib.load(context.artifacts["level2"])

    def _extract_from_cnn(self, X: torch.Tensor) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Internal method to get 512D embeddings and 8D probabilities from the CNN."""
        with torch.no_grad():
            embeddings = self.cnn_model.extract_features(X).numpy()
            probs = torch.sigmoid(self.cnn_model(X)).numpy()
        return embeddings, probs

    def extract_features(self, X: torch.Tensor) -> npt.NDArray[Any]:
        """Expose 512D spatial embeddings directly for MMD Drift Detection."""
        with torch.no_grad():
            embeddings = self.cnn_model.extract_features(X).numpy()
        return embeddings # type: ignore

    def predict(self, context: Any, model_input: Any) -> pd.DataFrame:
        """Full inference pass for DTMC Drift Detection and standard serving.

        Args:
            model_input: Raw 52x52x1 tensors (either as DataFrame or raw tensor).

        Returns:
            DataFrame containing final 8D predictions.
        """
        if isinstance(model_input, pd.DataFrame):
            # Try to handle as list or array inside DataFrame
            X = torch.tensor(model_input.values, dtype=torch.float32)
        else:
            X = model_input
            
        if len(X.shape) == 3:
            X = X.unsqueeze(1)
            
        # Level 0 (CNN)
        embeddings, probs_0 = self._extract_from_cnn(X)
        
        # Level 1A (Spatial XGBoost)
        probs_1a = self.level1a.predict_proba(embeddings)
        
        # Level 1B (Semantic MLP)
        probs_1b = self.level1b.predict_proba(probs_0)
        
        # Concatenate 24D state vector
        state_vector = np.hstack([probs_0, probs_1a, probs_1b])
        
        # Level 2 (Linear Decider)
        final_preds = self.level2.predict(state_vector)
        
        return pd.DataFrame(final_preds)

    def predict_internal(self, X: Any) -> npt.NDArray[Any]:
        """Helper for directly returning numpy array predictions."""
        return self.predict(None, X).values  # type: ignore[no-any-return]
