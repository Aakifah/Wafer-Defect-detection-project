import logging
import os
import subprocess
import time

import boto3
import mlflow
import requests
from botocore.config import Config
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

PUSHGATEWAY_URL = os.environ.get("PUSHGATEWAY_URL", "")
if PUSHGATEWAY_URL.startswith("$"):
    PUSHGATEWAY_URL = ""

PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "")
if PROMETHEUS_URL.startswith("$"):
    PROMETHEUS_URL = ""
    
GARAGE_ENDPOINT = os.environ.get("AWS_ENDPOINT_URL", "")
if GARAGE_ENDPOINT.startswith("$"):
    GARAGE_ENDPOINT = ""

GARAGE_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "")
if GARAGE_ACCESS_KEY.startswith("$"):
    GARAGE_ACCESS_KEY = ""

GARAGE_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
if GARAGE_SECRET_KEY.startswith("$"):
    GARAGE_SECRET_KEY = ""

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def verify_pushgateway() -> None:
    try:
        registry = CollectorRegistry()
        probe = Gauge("wafer_observability_probe", "Observability health probe", registry=registry)
        probe.set(1.0)
        push_to_gateway(PUSHGATEWAY_URL, job="wafer_observability_probe", registry=registry)
        
        resp = requests.get(f"{PUSHGATEWAY_URL}/metrics", timeout=10)
        resp.raise_for_status()
        if "wafer_observability_probe" not in resp.text:
            raise RuntimeError("Probe metric not found in Pushgateway /metrics")
        logger.info("[PUSHGATEWAY] Reachable and metric accepted.")
    except Exception as e:
        print("!!! Pushgateway is not available, fallback used!!!")
        raise e

def verify_prometheus() -> None:
    max_retries = 3
    sleep_interval = 2
    
    try:
        for attempt in range(max_retries):
            logger.info(f"[PROMETHEUS] Polling for metric... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(sleep_interval)
            resp = requests.get(
                f"{PROMETHEUS_URL}/api/v1/query",
                params={"query": "wafer_observability_probe"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data["status"] == "success" and len(data.get("data", {}).get("result", [])) > 0:
                logger.info("[PROMETHEUS] Scraped probe metric successfully.")
                return
                
        raise RuntimeError("Probe metric not found in Prometheus; Pushgateway scrape failed.")
    except Exception as e:
        print("!!! Prometheus is not available, fallback used!!!")
        raise e

def verify_garage() -> None:
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=GARAGE_ENDPOINT,
            aws_access_key_id=GARAGE_ACCESS_KEY,
            aws_secret_access_key=GARAGE_SECRET_KEY,
            region_name="garage",
            config=Config(signature_version="s3v4")
        )
        response = s3.list_buckets()
        buckets = [bucket['Name'] for bucket in response['Buckets']]
        logger.info(f"[GARAGE S3] Reachable. Buckets found: {buckets}")
    except Exception as e:
        print("!!! MinIO S3 is not available, fallback used!!!")
        raise e

def verify_mlflow() -> None:
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if mlflow_uri.startswith("$"):
        mlflow_uri = ""
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("Wafer_Defect_Detection")
        with mlflow.start_run(run_name="connectivity_test"):
            mlflow.log_metric("health_probe", 1.0)
        logger.info(f"[MLFLOW] Reachable. Logged probe to {mlflow_uri}")
    except Exception as e:
        print("!!! MLflow is not available, fallback used!!!")
        raise e

def verify_hardware() -> None:
    logger.info("=== Hardware Specifications ===")
    try:
        logger.info("--- CPU ---")
        subprocess.run(["lscpu"], check=False)
    except Exception as e:
        logger.error(f"Failed to check CPU: {e}")
        
    try:
        logger.info("--- Memory ---")
        subprocess.run(["free", "-h"], check=False)
    except Exception as e:
        logger.error(f"Failed to check memory: {e}")
        
    try:
        logger.info("--- Disk ---")
        subprocess.run(["df", "-h"], check=False)
    except Exception as e:
        logger.error(f"Failed to check disk: {e}")
        
def main() -> None:
    logger.info("Starting comprehensive services health check...")
    verify_hardware()
    results = {}
    
    for name, fn in [
        ("MLflow", verify_mlflow),
        ("Pushgateway", verify_pushgateway),
        ("Prometheus", verify_prometheus),
        ("MinIO S3", verify_garage)
    ]:
        try:
            fn()
            results[name] = "PASS"
        except Exception as e:
            logger.error("[%s] FAIL: %s", name, e)
            results[name] = f"FAIL: {e}"

    failures = [k for k, v in results.items() if v != "PASS"]
    if failures:
        logger.error("Health check FAILED for: %s", ", ".join(failures))
    else:
        logger.info("All services checks passed.")

if __name__ == "__main__":
    main()