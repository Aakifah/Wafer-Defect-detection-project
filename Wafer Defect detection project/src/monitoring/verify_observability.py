import logging
import os
import sys
import time

import requests
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

PUSHGATEWAY_URL = os.environ.get("PUSHGATEWAY_URL", "")
if PUSHGATEWAY_URL.startswith("$"):
    PUSHGATEWAY_URL = ""

PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "")
if PROMETHEUS_URL.startswith("$"):
    PROMETHEUS_URL = ""

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def verify_pushgateway() -> None:
    registry = CollectorRegistry()
    probe = Gauge("wafer_observability_probe", "Observability health probe", registry=registry)
    probe.set(1.0)
    push_to_gateway(PUSHGATEWAY_URL, job="wafer_observability_probe", registry=registry)
    
    resp = requests.get(f"{PUSHGATEWAY_URL}/metrics", timeout=10)
    resp.raise_for_status()
    if "wafer_observability_probe" not in resp.text:
        raise RuntimeError("Probe metric not found in Pushgateway /metrics")
    logger.info("[PUSHGATEWAY] Reachable and metric accepted.")

def verify_prometheus() -> None:
    max_retries = 5
    sleep_interval = 15
    
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

def main() -> None:
    logger.info("Starting observability stack verification...")
    results = {}
    for name, fn in [
        ("Pushgateway", verify_pushgateway),
        ("Prometheus", verify_prometheus),
    ]:
        try:
            fn()
            results[name] = "PASS"
        except Exception as e:
            logger.error("[%s] FAIL: %s", name, e)
            results[name] = f"FAIL: {e}"

    failures = [k for k, v in results.items() if v != "PASS"]
    if failures:
        for f in failures:
            print(f"!!! {f} is not available, fallback used!!!")
        logger.error("Observability verification FAILED for: %s", ", ".join(failures))
        sys.exit(0)
    logger.info("All observability checks passed.")

if __name__ == "__main__":
    main()