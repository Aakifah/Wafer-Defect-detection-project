from typing import Any

import numpy as np
import numpy.typing as npt


class DTMCDataDriftDetector:
    def __init__(self, num_states: int = 8, threshold: float = 0.50) -> None:
        """Initializes the Discrete-Time Markov Chain data drift detector.

        Args:
            num_states: The number of distinct categorical states.
            threshold: The Frobenius norm threshold above which drift is detected.
        """
        self.num_states: int = num_states
        self.threshold: float = threshold
        self.baseline_matrix: npt.NDArray[np.float64] | None = None

    def _validate_sequence(self, states: list[int]) -> None:
        """Ensures all class tokens fall inside valid boundary spaces.

        Args:
            states: A sequence of categorical states.

        Raises:
            ValueError: If any state is outside the allowed bounds.
        """
        out_of_bounds = [s for s in states if s < 0 or s >= self.num_states]
        if out_of_bounds:
            raise ValueError(
                f"Encountered out-of-bounds defect class tokens: {out_of_bounds}. "
                f"Allowed: 0-{self.num_states-1}"
            )

    def calculate_transition_matrix(self, states: list[int]) -> npt.NDArray[np.float64]:
        """Builds a row-normalized transition probability matrix from a sequence of tokens.

        Args:
            states: A sequence of categorical states.

        Returns:
            A row-normalized transition probability matrix.
        """
        self._validate_sequence(states)
        matrix = np.zeros((self.num_states, self.num_states), dtype=np.float64)
        
        if len(states) < 2:
            return matrix
            
        for i in range(len(states) - 1):
            curr_state = states[i]
            next_state = states[i + 1]
            matrix[curr_state, next_state] += 1.0
            
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
        return matrix

    def fit_baseline(self, baseline_states: list[int]) -> None:
        """Establishes and locks the permanent reference standard baseline matrix.

        Args:
            baseline_states: A sequence of states to compute the baseline transition matrix.
        """
        self.baseline_matrix = self.calculate_transition_matrix(baseline_states)

    def monitor_live_window(self, live_states: list[int]) -> dict[str, Any]:
        """Compares live factory data strings against the baseline using the Frobenius Norm.

        Args:
            live_states: A sequence of states from the live environment.

        Returns:
            A dictionary containing the drift score and a boolean flag for detected drift.
        """
        if self.baseline_matrix is None:
            raise ValueError("Baseline matrix has not been fitted yet.")
            
        live_matrix = self.calculate_transition_matrix(live_states)
        
        distance: float = float(np.linalg.norm(self.baseline_matrix - live_matrix))
        drift_detected: bool = distance > self.threshold
        
        return {
            "drift_score": distance,
            "drift_detected": drift_detected
        }