import numpy as np

from src.telemetry.model.dtmc_drift import DTMCDataDriftDetector


def test_production_telemetry_pipeline() -> None:
    """
    Validates that the DTMC drift engine correctly identifies stationary
    and anomalous prediction streams under production thresholds.
    """
    print("=== Phase 1: Ingesting Data & Initializing Engine ===")
    np.random.seed(42)
    
    historical_predictions = np.random.randint(0, 8, size=2000).tolist()
    
    detector = DTMCDataDriftDetector(num_states=8, threshold=0.50)
    detector.fit_baseline(historical_predictions)
    
    print("\n=== Phase 2: Testing Stationary 'No Drift' Live Stream ===")
    live_normal_predictions = np.random.randint(0, 8, size=1000).tolist()
    
    results_normal = detector.monitor_live_window(live_normal_predictions)
    drift_detected = results_normal["drift_detected"]
    score = results_normal["drift_score"]
    
    assert not drift_detected, f"False Positive: Flagged drift on clean data! Score: {score}"
    print(f"Stationary check passed safe. Score: {score:.4f}")
    
    print("\n=== Phase 3: Testing Anomalous 'Drift Detected' Live Stream ===")
    drifted_predictions = []
    current_state = 3
    for _ in range(1000):
        if np.random.rand() < 0.70:
            current_state = 4 if current_state == 3 else 3
        else:
            current_state = np.random.randint(0, 8)
        drifted_predictions.append(current_state)
        
    results_drifted = detector.monitor_live_window(drifted_predictions)
    drift_detected = results_drifted["drift_detected"]
    score = results_drifted["drift_score"]
    
    assert drift_detected, f"False Negative: Missed operational drift! Score: {score}"
    print(f"Anomalous check passed. Drift captured successfully. Score: {score:.4f}")