"""Maximum Mean Discrepancy (MMD) Data Drift Mathematical Engine."""

from typing import Any, cast

import numpy as np
import numpy.typing as npt


class MMDDataDriftDetector:
    def __init__(self, gamma: float | None = None, threshold: float = 0.05) -> None:
        """
        Initializes the MMD continuous data drift detector using NumPy.
        
        Args:
            gamma: RBF kernel bandwidth. If None, calculated via median heuristic.
            threshold: Distance threshold above which drift is flagged.
        """
        self.gamma: float | None = gamma
        self.threshold: float = threshold
        self.baseline_embeddings: npt.NDArray[np.float64] | None = None

    def _compute_rbf_kernel(self, X: npt.NDArray[np.float64], Y: npt.NDArray[np.float64], gamma: float) -> npt.NDArray[np.float64]:
        """Computes the RBF (Gaussian) kernel matrix between X and Y using matrix expansion."""
        X_norm = np.sum(X**2, axis=1, keepdims=True)
        Y_norm = np.sum(Y**2, axis=1, keepdims=True)
        dists = X_norm + Y_norm.T - 2 * np.dot(X, Y.T)
        result = np.exp(-gamma * np.maximum(dists, 0.0))
        return cast(npt.NDArray[np.float64], result)

    def fit_baseline(self, baseline_embeddings: npt.NDArray[np.float64]) -> None:
        """Stores reference baseline embeddings and computes heuristic bandwidth if needed."""
        if baseline_embeddings.ndim != 2:
            raise ValueError("Baseline embeddings must be a 2D array of shape (samples, features).")
            
        self.baseline_embeddings = baseline_embeddings.astype(np.float64)
        
        if self.gamma is None:
            X = self.baseline_embeddings
            X_norm = np.sum(X**2, axis=1, keepdims=True)
            dists = X_norm + X_norm.T - 2 * np.dot(X, X.T)
            median_dist = np.median(dists)
            self.gamma = float(1.0 / median_dist if median_dist > 0 else 1.0)

    def monitor_live_window(self, live_embeddings: npt.NDArray[np.float64]) -> dict[str, Any]:
        """Computes the MMD squared distance between live production data and the baseline."""
        if self.baseline_embeddings is None or self.gamma is None:
            raise ValueError("Baseline embeddings have not been fitted yet.")
            
        if live_embeddings.ndim != 2:
            raise ValueError("Live embeddings must be a 2D array.")

        X = self.baseline_embeddings
        Y = live_embeddings.astype(np.float64)
        
        K_xx = self._compute_rbf_kernel(X, X, self.gamma)
        K_yy = self._compute_rbf_kernel(Y, Y, self.gamma)
        K_xy = self._compute_rbf_kernel(X, Y, self.gamma)
        
        mmd_squared = float(K_xx.mean() + K_yy.mean() - 2 * K_xy.mean())
        drift_score = max(mmd_squared, 0.0)
        
        return {
            "drift_score": drift_score,
            "drift_detected": drift_score > self.threshold,
            "gamma": self.gamma,
            "threshold": self.threshold
        }