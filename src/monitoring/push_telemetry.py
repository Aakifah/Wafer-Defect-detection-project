"""Script to push computed telemetry metrics to Prometheus Pushgateway."""

import json
import logging
import os
from pathlib import Path

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BUFFER_FILE = "src/benchmark/matrices/telemetry_state_buffer.json"
PUSHGATEWAY_URL = os.environ.get("PUSHGATEWAY_URL", "http://pushgateway:9091")

def push_metrics() -> None:
    """Reads the telemetry buffer and pushes the metrics to the Prometheus Pushgateway."""
    if not os.path.exists(BUFFER_FILE):
        logger.warning(f"Telemetry buffer JSON not found at {BUFFER_FILE}. Skipping push.")
        return

    try:
        with open(BUFFER_FILE) as f:
            data = json.load(f)

        registry = CollectorRegistry()
        
        anomaly_gauge = Gauge(
            "wafer_telemetry_anomaly_count", 
            "Total active anomalies flagged across all drift detection engines", 
            registry=registry
        )
        dtmc_gauge = Gauge(
            "wafer_dtmc_drift_score", 
            "Current data drift score from Discrete-Time Markov Chain sequential engine", 
            registry=registry
        )
        mmd_gauge = Gauge(
            "wafer_mmd_drift_score", 
            "Current data drift score from Maximum Mean Discrepancy continuous embedding engine", 
            registry=registry
        )
        
        champ_map_gauge = Gauge(
            "wafer_champion_mAP",
            "Champion model mAP score",
            registry=registry
        )
        champ_f1_gauge = Gauge(
            "wafer_champion_f1",
            "Champion model Macro F1 score",
            registry=registry
        )
        comp_map_gauge = Gauge(
            "wafer_competitor_mAP",
            "Competitor model mAP score",
            registry=registry
        )
        comp_f1_gauge = Gauge(
            "wafer_competitor_f1",
            "Competitor model Macro F1 score",
            registry=registry
        )
        
        inference_duration = Gauge(
            "wafer_inference_duration_seconds",
            "Inference duration in seconds",
            registry=registry
        )
        training_duration = Gauge(
            "wafer_training_duration_seconds",
            "Total training duration in seconds",
            registry=registry
        )
        calibration_duration = Gauge(
            "wafer_calibration_duration_seconds",
            "Total calibration duration in seconds",
            registry=registry
        )

        anomaly_gauge.set(data.get("last_calculated_anomaly_count", 0))
        dtmc_gauge.set(data.get("dtmc_drift_score", 0.0))
        mmd_gauge.set(data.get("mmd_drift_score", 0.0))
        champ_map_gauge.set(data.get("champion_mAP", 0.0))
        champ_f1_gauge.set(data.get("champion_macro_f1", 0.0))
        comp_map_gauge.set(data.get("competitor_mAP", 0.0))
        comp_f1_gauge.set(data.get("competitor_macro_f1", 0.0))
        
        inference_duration.set(data.get("inference_duration_seconds", 0.0))
        training_duration.set(data.get("training_duration_seconds", 0.0))
        calibration_duration.set(data.get("calibration_duration_seconds", 0.0))

        class_metrics_path = Path("evaluation/quantized_cnn/class_metrics.json")
        if class_metrics_path.exists():
            with open(class_metrics_path) as f:
                class_data = json.load(f)

            class_f1_gauge = Gauge(
                "wafer_class_f1",
                "Per-class F1 score",
                ["class_name"],
                registry=registry
            )
            class_ap_gauge = Gauge(
                "wafer_class_ap",
                "Per-class Average Precision",
                ["class_name"],
                registry=registry
            )

            for cls_name, f1_score in class_data.get("class_wise_f1", {}).items():
                class_f1_gauge.labels(class_name=cls_name).set(f1_score)

            for cls_name, ap_score in class_data.get("class_wise_ap", {}).items():
                label = cls_name.replace("_AP", "")
                class_ap_gauge.labels(class_name=label).set(ap_score)

            logger.info(f"[PUSH] Pushed per-class metrics from {class_metrics_path}")

        push_to_gateway(PUSHGATEWAY_URL, job="wafer_telemetry", registry=registry)
        
        logger.info(
            f"[PUSH] Successfully pushed metrics to Pushgateway -> Anomaly: {data.get('last_calculated_anomaly_count')}, "
            f"DTMC: {data.get('dtmc_drift_score'):.4f}, MMD: {data.get('mmd_drift_score'):.4f}"
        )
    except Exception as e:
        print("!!! Pushgateway is not available, fallback used!!!")
        logger.error(f"Failed to read telemetry buffer or push to Pushgateway: {e}")

if __name__ == "__main__":
    push_metrics()