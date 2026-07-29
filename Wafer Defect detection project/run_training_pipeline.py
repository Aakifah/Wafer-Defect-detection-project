import logging
import os
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_script(script_path: str) -> None:
    """Runs a Python script as a subprocess and streams its output."""
    logger.info(f"========== Starting: {script_path} ==========")
    start_time = time.time()
    
    try:
        subprocess.run(
            [sys.executable, script_path],
            check=True,
            text=True,
            env=None
        )
        duration = time.time() - start_time
        logger.info(f"========== Finished: {script_path} in {duration:.2f} seconds ==========\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"========== FAILED: {script_path} ==========")
        logger.error(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}.")
        logger.error("Aborting pipeline execution to prevent cascading failures.")
        sys.exit(1)
    except FileNotFoundError:
        logger.error(f"Script not found: {script_path}")
        sys.exit(1)

def main() -> None:
    logger.info("Initializing Wafer Defect Detection Training Pipeline...")
    
    pipeline_scripts = [
        "src/training/optimizer_benchmark.py",
        "src/training/train_rf_standard.py",
        "src/training/train_rf_super.py",
        "src/training/train_bayesian.py",
        "src/training/train_base.py",
        "src/training/train_focal.py",
        "src/training/train_asvd.py",
        "src/training/train_quantized.py",
        "src/training/train_sssn.py",
        "src/training/log_to_mlflow.py",
        "src/telemetry/calibrate_baseline.py",
        "src/telemetry/calibrate_mmd.py"
    ]
    
    expected_outputs = {
        "src/training/optimizer_benchmark.py": "best_optimizer.json",
        "src/training/train_rf_standard.py": "evaluation/rf_standard/model.pkl",
        "src/training/train_rf_super.py": "evaluation/rf_super/model.pkl",
        "src/training/train_sssn.py": "evaluation/sssn_competitor/metrics.json",
    }
    
    total_start = time.time()
    
    for script in pipeline_scripts:
        skip = False
        if script in expected_outputs:
            output_file = expected_outputs[script]
            if os.path.exists(output_file):
                logger.info(f"Skipping {script} as output {output_file} already exists.")
                skip = True
                
        if not skip:
            run_script(script)
        
    total_duration = time.time() - total_start
    hours, rem = divmod(total_duration, 3600)
    minutes, seconds = divmod(rem, 60)
    
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
