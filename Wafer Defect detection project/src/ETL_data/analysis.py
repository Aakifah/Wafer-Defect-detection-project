"""Comprehensive data analysis.

Analyzes raw data for nulls, duplicates, class distribution, spatial statistics,
and data integrity issues. Outputs a structured report to console and JSON file.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

REPORT_PATH = Path("data/analysis_report.json")


def load_raw_data() -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Load processed wafer map data from S3."""
    import io

    import boto3

    endpoint_url = os.environ.get("AWS_ENDPOINT_URL", "")
    if endpoint_url.startswith("$"):
        endpoint_url = ""
        
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    if access_key.startswith("$"):
        access_key = ""
        
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if secret_key.startswith("$"):
        secret_key = ""
    client = None
    if not all([endpoint_url, access_key, secret_key]):
        print("!!! MinIO S3 is not available, fallback used!!!")
    else:
        try:
            client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name="garage",
            )
        except Exception:
            print("!!! MinIO S3 is not available, fallback used!!!")
    
    Xs = []
    ys = []
    for split in ["train", "val", "test"]:
        key = f"processed/{split}_data.npz"
        local_path = f"data/processed/{split}_data.npz"
        try:
            if client is None:
                raise ValueError("No S3 client")
            logger.info("Loading processed data from S3: s3://datasets/%s", key)
            response = client.get_object(Bucket="datasets", Key=key)
            archive = np.load(io.BytesIO(response["Body"].read()))
        except Exception:
            print("!!! MinIO S3 is not available, fallback used!!!")
            archive = np.load(local_path)
            
        Xs.append(archive["X"])
        ys.append(archive["y"])
        
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y


def check_nulls_and_nans(X: npt.NDArray[Any], y: npt.NDArray[Any]) -> dict[str, Any]:
    """Check for null, NaN, and Inf values."""
    X_float = X.astype(float)
    y_float = y.astype(float)
    return {
        "X_nan_count": int(np.sum(np.isnan(X_float))),
        "X_inf_count": int(np.sum(np.isinf(X_float))),
        "y_nan_count": int(np.sum(np.isnan(y_float))),
        "y_inf_count": int(np.sum(np.isinf(y_float))),
        "has_issues": bool(
            np.sum(np.isnan(X_float))
            or np.sum(np.isinf(X_float))
            or np.sum(np.isnan(y_float))
            or np.sum(np.isinf(y_float))
        ),
    }


def analyze_value_ranges(X: npt.NDArray[Any]) -> dict[str, Any]:
    """Analyze pixel value distribution."""
    unique_values, counts = np.unique(X, return_counts=True)
    return {
        "dtype": str(X.dtype),
        "min": int(X.min()),
        "max": int(X.max()),
        "mean": float(X.mean()),
        "std": float(X.std()),
        "median": float(np.median(X)),
        "unique_values": {int(v): int(c) for v, c in zip(unique_values, counts)},
    }


def analyze_duplicates(
    X: npt.NDArray[Any],
    y: npt.NDArray[Any],
) -> dict[str, Any]:
    """Analyze duplicate entries: exact (X,y) and same-X-different-label."""
    X_flat = X.reshape(len(X), -1)

    unique_X, inverse_X, counts_X = np.unique(
        X_flat, axis=0, return_inverse=True, return_counts=True
    )
    x_dupe_patterns = int(np.sum(counts_X > 1))
    x_dupe_extra = int(np.sum(counts_X - 1))

    combined = np.hstack([X_flat, y])
    unique_comb, inverse_comb, counts_comb = np.unique(
        combined, axis=0, return_inverse=True, return_counts=True
    )
    exact_dupe_patterns = int(np.sum(counts_comb > 1))
    exact_dupe_extra = int(np.sum(counts_comb - 1))

    same_x_same_label = 0
    same_x_diff_label = 0
    diff_label_examples: list[dict[str, Any]] = []

    for idx in np.where(counts_X > 1)[0]:
        positions = np.where(inverse_X == idx)[0]
        labels = y[positions]
        if np.all(labels == labels[0]):
            same_x_same_label += 1
        else:
            same_x_diff_label += 1
            if len(diff_label_examples) < 5:
                diff_label_examples.append({
                    "indices": positions.tolist(),
                    "labels": labels.tolist(),
                })

    zero_label_count = int(np.sum(np.all(y == 0, axis=1)))

    return {
        "total_samples": len(X),
        "unique_X_maps": len(unique_X),
        "X_duplicate_patterns": x_dupe_patterns,
        "X_extra_copies": x_dupe_extra,
        "exact_Xy_duplicate_patterns": exact_dupe_patterns,
        "exact_Xy_extra_copies": exact_dupe_extra,
        "same_X_same_label_true_redundancy": same_x_same_label,
        "same_X_different_label_data_conflict": same_x_diff_label,
        "all_zeros_label_count_normal_wafers": zero_label_count,
        "data_conflict_examples": diff_label_examples,
    }


def analyze_label_distribution(y: npt.NDArray[Any]) -> dict[str, Any]:
    """Analyze multi-label distribution: per-class and per-pattern."""
    num_classes = y.shape[1]
    per_class = {}
    for c in range(num_classes):
        positive = int(np.sum(y[:, c]))
        per_class[f"class_{c}"] = {
            "positive": positive,
            "negative": len(y) - positive,
            "positive_pct": round(positive / len(y) * 100, 2),
        }

    unique_patterns, pattern_counts = np.unique(y, axis=0, return_counts=True)
    sorted_idx = np.argsort(pattern_counts)[::-1]
    patterns = {}
    for rank, idx in enumerate(sorted_idx):
        patterns[f"pattern_{rank}"] = {
            "labels": unique_patterns[idx].tolist(),
            "count": int(pattern_counts[idx]),
            "pct": round(pattern_counts[idx] / len(y) * 100, 2),
        }

    labels_per_sample = y.sum(axis=1)
    return {
        "num_classes": num_classes,
        "total_samples": len(y),
        "unique_label_patterns": len(unique_patterns),
        "per_class": per_class,
        "label_patterns": patterns,
        "labels_per_sample": {
            "min": int(labels_per_sample.min()),
            "max": int(labels_per_sample.max()),
            "mean": round(float(labels_per_sample.mean()), 2),
            "median": float(np.median(labels_per_sample)),
        },
    }


def analyze_spatial_statistics(X: npt.NDArray[Any]) -> dict[str, Any]:
    """Analyze spatial/pixel-level statistics across the dataset."""
    X_3d = X.reshape(len(X), -1)
    per_pixel_mean = X_3d.mean(axis=0)
    per_pixel_std = X_3d.std(axis=0)

    return {
        "shape": list(X.shape),
        "per_pixel_mean": {
            "min": round(float(per_pixel_mean.min()), 4),
            "max": round(float(per_pixel_mean.max()), 4),
            "overall": round(float(per_pixel_mean.mean()), 4),
        },
        "per_pixel_std": {
            "min": round(float(per_pixel_std.min()), 4),
            "max": round(float(per_pixel_std.max()), 4),
            "overall": round(float(per_pixel_std.mean()), 4),
        },
        "empty_maps_count": int(np.sum(np.all(X == 0, axis=(1, 2)))),
        "fully_filled_maps_count": int(np.sum(np.all(X > 0, axis=(1, 2)))),
    }


def generate_report(X: npt.NDArray[Any], y: npt.NDArray[Any]) -> dict[str, Any]:
    """Generate comprehensive analysis report."""
    logger.info("=== Starting Comprehensive Data Analysis ===")
    logger.info("Dataset shape: X=%s, y=%s", X.shape, y.shape)

    report: dict[str, Any] = {
        "dataset_info": {
            "num_samples": len(X),
            "map_shape": list(X.shape[1:]),
            "num_classes": y.shape[1],
        },
        "null_nan_check": check_nulls_and_nans(X, y),
        "value_ranges": analyze_value_ranges(X),
        "duplicate_analysis": analyze_duplicates(X, y),
        "label_distribution": analyze_label_distribution(y),
        "spatial_statistics": analyze_spatial_statistics(X),
    }

    return report


def print_summary(report: dict[str, Any]) -> None:
    """Print a human-readable summary of the analysis report."""
    print("\n" + "=" * 70)
    print("WAFER DEFECT DETECTION — COMPREHENSIVE DATA ANALYSIS REPORT")
    print("=" * 70)

    info = report["dataset_info"]
    print(f"\nDataset: {info['num_samples']} samples, {info['map_shape']} maps, "
          f"{info['num_classes']} classes")

    nulls = report["null_nan_check"]
    print("\n--- Data Integrity ---")
    print(f"  NaN in X: {nulls['X_nan_count']} | Inf in X: {nulls['X_inf_count']}")
    print(f"  NaN in y: {nulls['y_nan_count']} | Inf in y: {nulls['y_inf_count']}")
    print(f"  Issues found: {'YES' if nulls['has_issues'] else 'NO'}")

    vr = report["value_ranges"]
    print("\n--- Pixel Value Ranges ---")
    print(f"  Dtype: {vr['dtype']} | Range: [{vr['min']}, {vr['max']}]")
    print(f"  Mean: {vr['mean']:.4f} | Std: {vr['std']:.4f} | Median: {vr['median']:.4f}")
    print(f"  Value distribution: {vr['unique_values']}")

    dup = report["duplicate_analysis"]
    print("\n--- Duplicate Analysis ---")
    print(f"  Unique X maps: {dup['unique_X_maps']} / {dup['total_samples']}")
    print(f"  X duplicate patterns: {dup['X_duplicate_patterns']} "
          f"({dup['X_extra_copies']} extra copies)")
    print(f"  Exact (X,y) duplicate patterns: {dup['exact_Xy_duplicate_patterns']} "
          f"({dup['exact_Xy_extra_copies']} extra copies)")
    print(f"  Same X, same label (true redundancy): {dup['same_X_same_label_true_redundancy']}")
    print(
        f"  Same X, different label (DATA CONFLICT): "
        f"{dup['same_X_different_label_data_conflict']}"
    )
    print(f"  All-zeros labels (normal wafers): {dup['all_zeros_label_count_normal_wafers']}")
    if dup["data_conflict_examples"]:
        print("  WARNING: Data conflict examples:")
        for ex in dup["data_conflict_examples"]:
            print(f"    Indices {ex['indices']}: labels = {ex['labels']}")

    ld = report["label_distribution"]
    print("\n--- Label Distribution ---")
    print(f"  Unique label patterns: {ld['unique_label_patterns']}")
    print(f"  Labels per sample: min={ld['labels_per_sample']['min']}, "
          f"max={ld['labels_per_sample']['max']}, "
          f"mean={ld['labels_per_sample']['mean']}")
    print("\n  Per-class frequency:")
    for cls_name, cls_info in ld["per_class"].items():
        bar = "#" * int(cls_info["positive_pct"] / 2)
        print(f"    {cls_name}: {cls_info['positive']:>6} positive "
              f"({cls_info['positive_pct']:>5.2f}%) {bar}")

    print("\n  Top 10 label patterns:")
    for i in range(min(10, len(ld["label_patterns"]))):
        p = ld["label_patterns"][f"pattern_{i}"]
        print(f"    {p['labels']}: {p['count']:>5} samples ({p['pct']:.1f}%)")

    ss = report["spatial_statistics"]
    print("\n--- Spatial Statistics ---")
    print(f"  Per-pixel mean: min={ss['per_pixel_mean']['min']:.4f}, "
          f"max={ss['per_pixel_mean']['max']:.4f}, "
          f"overall={ss['per_pixel_mean']['overall']:.4f}")
    print(f"  Per-pixel std:  min={ss['per_pixel_std']['min']:.4f}, "
          f"max={ss['per_pixel_std']['max']:.4f}, "
          f"overall={ss['per_pixel_std']['overall']:.4f}")
    print(f"  Empty maps (all zeros): {ss['empty_maps_count']}")
    print(f"  Fully filled maps (all >0): {ss['fully_filled_maps_count']}")

    print("\n" + "=" * 70)


def main() -> dict[str, Any]:
    """Run the full analysis pipeline."""
    X, y = load_raw_data()
    report = generate_report(X, y)
    print_summary(report)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to %s", REPORT_PATH)

    return report


if __name__ == "__main__":
    main()
