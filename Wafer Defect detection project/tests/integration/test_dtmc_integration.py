"""Operational pipeline verification script for the DTMC Data Drift Engine."""

import numpy as np

from src.telemetry.model.dtmc_drift import DTMCDataDriftDetector


def test_pipeline_validation() -> None:
    """
    Integration test checking baseline calibration, live window drift capturing,
    and out-of-bounds error guardrails.
    """
    print("STARTING OPERATIONAL TELEMETRY ENGINE VALIDATION SUITE")
    
    detector = DTMCDataDriftDetector(num_states=8, threshold=0.50)
    
    np.random.seed(42)
    
  
  
    print("STEP 1: Simulating historical clean validation data stream...")
    historical_validation_stream = np.random.randint(0, 8, size=2000).tolist()
    
    detector.fit_baseline(historical_validation_stream)
    print("Baseline transition matrix locked into RAM.\n")
    
  

    print("STEP 2: Simulating live, stable production data window...")
    live_stable_stream = np.random.randint(0, 8, size=1000).tolist()
    
    stable_results = detector.monitor_live_window(live_stable_stream)
    drift_detected = stable_results["drift_detected"]
    
    assert not drift_detected, "Fail: Engine gave false alarm!"
    print(f" Stable Window Check Passed. Drift Detected: '{drift_detected}'")
    print(f" Calculated Distance Score: {stable_results['drift_score']:.4f} (Threshold: {detector.threshold})\n")
    
    
  
    print("STEP 3: Simulating physical tool fault (Injecting severe drift)...")
  
    drifted_stream = []
    current_state = 3
    for _ in range(1000):
        if np.random.rand() < 0.70:
            current_state = 4 if current_state == 3 else 3
        else:
            current_state = np.random.randint(0, 8)
        drifted_stream.append(current_state)
        
    drift_results = detector.monitor_live_window(drifted_stream)
    drift_detected_live = drift_results["drift_detected"]
    
    assert drift_detected_live, "Fail: Engine missed actual structural drift!"
    print(f" Drift Window Check Passed. Drift Detected: '{drift_detected_live}'")
    print(f" Calculated Distance Score: {drift_results['drift_score']:.4f} (Threshold: {detector.threshold})\n")
    


   
    print("STEP 4: Testing ingestion guardrails against corrupted values...")
    corrupted_stream = [9]
    
    try:
        detector.monitor_live_window(corrupted_stream)
        print(" FAILED: Ingestion engine allowed out-of-bounds data to pass through!")
        assert False, "Ingestion engine allowed an out-of-bounds state to bypass validation!"
    except ValueError as e:
        print(" Ingestion Guardrail Passed. Successfully caught corrupt data and raised error:")
        print(f"   -> Error message: {e}\n")
        
print("ALL MONITORING PIPELINE CHECKS PASSED PERFECTLY :)")
 