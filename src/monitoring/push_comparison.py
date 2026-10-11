"""Push per-stage comparison metrics to Prometheus Pushgateway."""

import json
import logging
import os
from pathlib import Path

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PUSHGATEWAY_URL = os.environ.get("PUSHGATEWAY_URL", "http://pushgateway:9091")


def push_comparison(eval_dir: str = "evaluation") -> None:
    """Push mAP/F1/loss gauges for every stage with metrics.json.

    Args:
        eval_dir: Directory containing per-stage evaluation subdirectories.
    """
    registry = CollectorRegistry()
    g_map = Gauge("wafer_stage_mAP", "mAP by pipeline stage", ["stage"], registry=registry)
    g_micro = Gauge("wafer_stage_micro_f1", "Micro F1 by pipeline stage", ["stage"], registry=registry)
    g_macro = Gauge("wafer_stage_macro_f1", "Macro F1 by pipeline stage", ["stage"], registry=registry)
    g_loss = Gauge("wafer_stage_test_loss", "Test loss by pipeline stage", ["stage"], registry=registry)
    pushed = 0
    for metrics_path in sorted(Path(eval_dir).glob("*/metrics.json")):
        stage = metrics_path.parent.name
        try:
            metrics = json.loads(metrics_path.read_text())
            g_map.labels(stage=stage).set(float(metrics["mAP"]))
            g_micro.labels(stage=stage).set(float(metrics["micro_f1"]))
            g_macro.labels(stage=stage).set(float(metrics["macro_f1"]))
            g_loss.labels(stage=stage).set(float(metrics.get("test_loss", 0.0)))
            pushed += 1
        except (KeyError, ValueError) as e:
            logger.warning("Skipping %s: %s", stage, e)
    push_to_gateway(PUSHGATEWAY_URL, job="wafer_stages", registry=registry)
    logger.info("Pushed %d stage metric sets to %s", pushed, PUSHGATEWAY_URL)


if __name__ == "__main__":
    push_comparison()
