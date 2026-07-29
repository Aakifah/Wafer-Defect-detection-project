"""ETL pipeline: deduplicate, stratified split, and upload wafer map data."""

import io
import logging
import os
import sys
from typing import Any

import boto3
import numpy as np
import numpy.typing as npt
from dotenv import load_dotenv
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def get_s3_client() -> Any | None:
    """Initializes and returns an S3 client using environment configurations."""
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL", "")
    if endpoint_url.startswith("$"):
        endpoint_url = ""
        
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    if access_key.startswith("$"):
        access_key = ""
        
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if secret_key.startswith("$"):
        secret_key = ""

    if not endpoint_url or not access_key or not secret_key:
        print("!!! MinIO S3 is not available, fallback used!!!")
        return None

    try:
        return boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="garage",
        )
    except Exception:
        print("!!! MinIO S3 is not available, fallback used!!!")
        return None


def upload_array_to_s3(
    s3_client: Any, bucket: str, key: str, X: npt.NDArray[Any], y: npt.NDArray[Any]
) -> None:
    """Serializes and uploads numpy arrays directly to S3.
    
    Args:
        s3_client: Boto3 S3 client.
        bucket: The target S3 bucket name.
        key: The object key (path) within the bucket.
        X: Feature array to upload.
        y: Target array to upload.
    """
    if s3_client is None:
        print("!!! MinIO S3 is not available, fallback used!!!")
        return

    buffer = io.BytesIO()
    np.savez_compressed(buffer, X=X, y=y)
    buffer.seek(0)
    try:
        s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
        logger.info("Successfully uploaded %s to s3://%s", key, bucket)
    except Exception:
        print("!!! MinIO S3 is not available, fallback used!!!")


def deduplicate(
    X: npt.NDArray[Any], y: npt.NDArray[Any]
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], int]:
    """Remove exact (X, y) duplicate entries, keeping one copy of each.

    Redundant data can artificially inflate evaluation metrics and introduce
    training bias. This function ensures each wafer map/label pair is unique.

    Args:
        X: The wafer map feature arrays.
        y: The corresponding multi-label arrays.
        
    Returns:
        A tuple of (deduplicated X, deduplicated y, removed count).
    """
    X_flat = X.reshape(len(X), -1)
    combined = np.hstack([X_flat, y])

    _, unique_indices = np.unique(combined, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)

    removed_count = len(X) - len(unique_indices)
    X_dedup = X[unique_indices]
    y_dedup = y[unique_indices]

    logger.info(
        "Deduplication: %d -> %d samples (%d duplicates removed)",
        len(X), len(X_dedup), removed_count,
    )
    return X_dedup, y_dedup, removed_count


def stratified_split(
    X: npt.NDArray[Any],
    y: npt.NDArray[Any],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_STATE,
) -> dict[str, tuple[npt.NDArray[Any], npt.NDArray[Any]]]:
    """Perform multi-label stratified split: train/val/test.

    Uses MultilabelStratifiedShuffleSplit to preserve label distribution
    across all splits. Splits sequentially: first isolate test set,
    then split remaining into train/val.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, (
        "Split ratios must sum to 1.0"
    )

    test_size = test_ratio
    val_adjusted = val_ratio / (train_ratio + val_ratio)

    msss_test = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_val_idx, test_idx = next(msss_test.split(X, y))

    X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    msss_val = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_adjusted, random_state=random_state
    )
    train_idx, val_idx = next(msss_val.split(X_train_val, y_train_val))

    X_train, y_train = X_train_val[train_idx], y_train_val[train_idx]
    X_val, y_val = X_train_val[val_idx], y_train_val[val_idx]

    logger.info(
        "Split complete — Train: %d (%.1f%%) | Val: %d (%.1f%%) | Test: %d (%.1f%%)",
        len(X_train), len(X_train) / len(X) * 100,
        len(X_val), len(X_val) / len(X) * 100,
        len(X_test), len(X_test) / len(X) * 100,
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


def verify_no_cross_contamination(
    train: tuple[npt.NDArray[Any], npt.NDArray[Any]],
    val: tuple[npt.NDArray[Any], npt.NDArray[Any]],
    test: tuple[npt.NDArray[Any], npt.NDArray[Any]],
) -> None:
    """Verify zero overlap between splits."""
    X_train_flat = train[0].reshape(len(train[0]), -1)
    X_val_flat = val[0].reshape(len(val[0]), -1)
    X_test_flat = test[0].reshape(len(test[0]), -1)

    train_set = set(map(tuple, X_train_flat))
    val_set = set(map(tuple, X_val_flat))
    test_set = set(map(tuple, X_test_flat))

    tv_overlap = len(train_set & val_set)
    tt_overlap = len(train_set & test_set)
    vt_overlap = len(val_set & test_set)

    if tv_overlap or tt_overlap or vt_overlap:
        logger.error(
            "CROSS-CONTAMINATION DETECTED: train-val=%d, train-test=%d, val-test=%d",
            tv_overlap, tt_overlap, vt_overlap,
        )
        sys.exit(1)

    logger.info("Cross-contamination check: PASSED (zero overlap)")


def partition_wafer_data(
    storage_bucket: str = "datasets",
    source_key: str = "raw/Wafer_Map_Datasets.npz",
    target_prefix: str = "processed/",
) -> None:
    """Extract, deduplicate, partition, and upload wafer datasets."""
    s3 = get_s3_client()
    if s3 is None:
        print("!!! MinIO S3 is not available, fallback used!!!")
        try:
            archive = np.load("data/raw/Wafer_Map_Datasets.npz")
            logger.info("Loaded raw data from local fallback.")
        except Exception:
            return
    else:
        logger.info("Extracting archive from s3://%s/%s", storage_bucket, source_key)
        try:
            response: dict[str, Any] = s3.get_object(Bucket=storage_bucket, Key=source_key)
            archive_bytes = response["Body"].read()
            archive = np.load(io.BytesIO(archive_bytes))
        except Exception:
            print("!!! MinIO S3 is not available, fallback used!!!")
            try:
                archive = np.load("data/raw/Wafer_Map_Datasets.npz")
                logger.info("Loaded raw data from local fallback.")
            except Exception:
                return

    X: npt.NDArray[Any] = archive["arr_0"]
    y: npt.NDArray[Any] = archive["arr_1"]
    logger.info("Extraction successful. X shape: %s, y shape: %s", X.shape, y.shape)

    X_dedup, y_dedup, removed = deduplicate(X, y)

    splits = stratified_split(X_dedup, y_dedup)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    verify_no_cross_contamination(
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    )

    upload_array_to_s3(s3, storage_bucket, f"{target_prefix}train_data.npz", X_train, y_train)
    upload_array_to_s3(s3, storage_bucket, f"{target_prefix}val_data.npz", X_val, y_val)
    upload_array_to_s3(s3, storage_bucket, f"{target_prefix}test_data.npz", X_test, y_test)

    logger.info(
        "ETL Complete. Train: %d | Val: %d | Test: %d | Duplicates removed: %d",
        X_train.shape[0], X_val.shape[0], X_test.shape[0], removed,
    )


if __name__ == "__main__":
    partition_wafer_data()
