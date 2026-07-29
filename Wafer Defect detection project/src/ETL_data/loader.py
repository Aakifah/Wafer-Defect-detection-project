"""S3-backed PyTorch Dataset and DataLoader factory for wafer map data.

Credentials are read from environment variables:
  - AWS_ENDPOINT_URL
  - AWS_ACCESS_KEY_ID
  - AWS_SECRET_ACCESS_KEY

Works with .env file locally (via load_dotenv) and GitLab CI/CD variables
remotely (injected as environment variables by the runner).
"""

import io
import logging
import os
import sys
from typing import Any

import boto3
import numpy as np
import numpy.typing as npt
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset

load_dotenv()

logger = logging.getLogger(__name__)

S3_BUCKET = "datasets"
S3_PREFIX = "processed/"
SPLITS = ("train", "val", "test")


def get_s3_client() -> Any | None:
    """Create an S3 client from environment variables.
    
    Returns:
        A configured boto3 S3 client targeting the MinIO endpoint, or None if configured improperly.
    """
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL", "")
    if endpoint_url.startswith("$"):
        endpoint_url = ""
        
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    if access_key.startswith("$"):
        access_key = ""
        
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if secret_key.startswith("$"):
        secret_key = ""

    if not all([endpoint_url, access_key, secret_key]):
        print("!!! MinIO S3 is not available, fallback used!!!")
        logger.warning("Missing AWS S3 environment variables. S3 client will be None.")
        return None

    try:
        return boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="garage",
        )
    except Exception as e:
        print("!!! MinIO S3 is not available, fallback used!!!")
        logger.warning("Failed to initialize S3 client: %s", str(e))
        return None


def _download_split(s3_client: Any, split: str) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Download a single split from S3 and return (X, y) numpy arrays."""
    key = f"{S3_PREFIX}{split}_data.npz"
    local_path = f"data/processed/{split}_data.npz"

    try:
        if s3_client is not None:
            logger.info("Downloading s3://%s/%s", S3_BUCKET, key)
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            data = np.load(io.BytesIO(response["Body"].read()))
        else:
            raise ValueError("S3 client is None")
    except Exception as e:
        print("!!! MinIO S3 is not available, fallback used!!!")
        logger.warning("Failed to download from S3: %s. Falling back to local data at %s", str(e), local_path)
        if os.path.exists(local_path):
            data = np.load(local_path)
        else:
            logger.error("Local file %s not found either.", local_path)
            sys.exit(1)

    X: npt.NDArray[Any] = data["X"]
    y: npt.NDArray[Any] = data["y"]
    logger.info("Loaded %s: X=%s, y=%s", split, X.shape, y.shape)
    return X, y


class WaferMapDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset that serves wafer map tensors from S3-loaded data."""

    def __init__(self, split: str, s3_client: Any | None = None) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got '{split}'")

        client = s3_client if s3_client is not None else get_s3_client()
        X_np, y_np = _download_split(client, split)

        self._X = torch.tensor(X_np, dtype=torch.float32)
        self._y = torch.tensor(y_np, dtype=torch.float32)

    def __len__(self) -> int:
        return self._X.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._X[idx], self._y[idx]

    @property
    def num_classes(self) -> int:
        return self._y.shape[1]

    @property
    def map_shape(self) -> tuple[int, int]:
        h, w = self._X.shape[1], self._X.shape[2]
        return (h, w)


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 0,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
    s3_client: Any | None = None,
) -> tuple[
    DataLoader[tuple[torch.Tensor, ...]],
    DataLoader[tuple[torch.Tensor, ...]],
    DataLoader[tuple[torch.Tensor, ...]],
]:
    """Create train/val/test DataLoaders from S3.

    Args:
        batch_size: Batch size for all loaders.
        num_workers: DataLoader workers (0 for Windows/CPU).
        max_train_samples: Cap training samples (for smoke tests).
        max_val_samples: Cap validation samples.
        max_test_samples: Cap test samples.
        s3_client: Optional pre-created S3 client.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    client = s3_client if s3_client is not None else get_s3_client()

    train_ds = WaferMapDataset("train", client)
    val_ds = WaferMapDataset("val", client)
    test_ds = WaferMapDataset("test", client)

    if max_train_samples is not None:
        train_ds._X = train_ds._X[:max_train_samples]
        train_ds._y = train_ds._y[:max_train_samples]
    if max_val_samples is not None:
        val_ds._X = val_ds._X[:max_val_samples]
        val_ds._y = val_ds._y[:max_val_samples]
    if max_test_samples is not None:
        test_ds._X = test_ds._X[:max_test_samples]
        test_ds._y = test_ds._y[:max_test_samples]

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    logger.info(
        "DataLoaders created — train=%d, val=%d, test=%d (batch_size=%d)",
        len(train_ds),
        len(val_ds),
        len(test_ds),
        batch_size,
    )

    return train_dl, val_dl, test_dl
