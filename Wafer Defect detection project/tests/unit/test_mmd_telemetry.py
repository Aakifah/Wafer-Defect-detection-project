"""Unit test suite for the standalone NumPy MMD Data Drift Engine."""

import numpy as np

from src.telemetry.model.mmd_drift import MMDDataDriftDetector


def test_mmd_engine_catches_distribution_shift() -> None:
    np.random.seed(42)
    
    baseline_data = np.random.normal(loc=0.0, scale=1.0, size=(100, 16))
    detector = MMDDataDriftDetector(threshold=0.02)
    detector.fit_baseline(baseline_data)
    
    stable_live_data = np.random.normal(loc=0.0, scale=1.0, size=(50, 16))
    stable_results = detector.monitor_live_window(stable_live_data)
    assert not stable_results["drift_detected"]
    
    shifted_live_data = np.random.normal(loc=0.7, scale=1.0, size=(50, 16))
    shifted_results = detector.monitor_live_window(shifted_live_data)
    assert shifted_results["drift_detected"]