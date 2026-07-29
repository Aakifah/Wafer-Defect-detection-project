"""Script to generate metadata for the Feast feature store."""

import os
from datetime import UTC, datetime
from typing import Any

import boto3
import numpy as np
import numpy.typing as npt
import pandas as pd
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

endpoint = os.environ.get("AWS_ENDPOINT_URL")
access_key = os.environ.get("AWS_ACCESS_KEY_ID")
secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")


s3 = None
try:
    if not endpoint or not access_key or not secret_key:
        raise ValueError("Missing S3 credentials")
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,  
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="garage",
        config=Config(signature_version="s3v4")
    )
except Exception:
    print("!!! Garage is not available, fallback used!!!")

def count_duplicates(X: npt.NDArray[Any]) -> int:
    """Counts the number of duplicate samples in the feature array.
    
    Args:
        X: The NumPy feature array.
        
    Returns:
        The total count of removed/duplicate arrays.
    """
    flat = X.reshape(len(X), -1)
    unique = np.unique(flat, axis=0)
    return int(len(X) - len(unique))

def download_npz(bucket: str, key: str, local_path: str) -> Any:
    """Downloads an NPZ file from an S3 bucket and loads it.
    
    Args:
        bucket: Name of the S3 bucket.
        key: The object key within the S3 bucket.
        local_path: The local destination path to save the file.
        
    Returns:
        The loaded NumPy NpzFile object.
    """
    fallback_path = "data/" + key
    try:
        if s3 is None:
            raise ValueError("No S3 client")
        s3.download_file(bucket, key, local_path)
        return np.load(local_path)
    except Exception:
        print("!!! MinIO S3 is not available, fallback used!!!")
        return np.load(fallback_path)


train = download_npz("datasets", "processed/train_data.npz", "train_temp.npz")
test = download_npz("datasets", "processed/test_data.npz", "test_temp.npz")
val = download_npz("datasets", "processed/val_data.npz", "val_temp.npz")

X_train, y_train = train["X"], train["y"]
X_test, y_test = test["X"], test["y"]
X_val, y_val = val["X"], val["y"]

num_classes = y_train.shape[1]
avg_labels = float(y_train.sum(axis=1).mean())
max_labels = int(y_train.sum(axis=1).max())

metadata = {
    "dataset_id": [1],
    "event_timestamp":[datetime.now(UTC)],
    "num_train_samples": [len(X_train)],
    "num_test_samples": [len(X_test)],
    "num_val_samples":[len(X_val)],
    "num_classes": [num_classes],
    "image_height":[X_train.shape[1]],
    "image_width": [X_train.shape[2]],
    "pixel_min": [float(X_train.min())],
    "pixel_max": [float(X_train.max())],
    "pixel_mean":[float(X_train.mean())],
    "pixel_std": [float(X_train.std())],
    "avg_labels_per_sample": [avg_labels],
    "max_labels_per_sample":[max_labels],
    "num_duplicates_train": [count_duplicates(X_train)],
}

df = pd.DataFrame(metadata)
print("Metadata extracted:")
print(df)

print("Serializing and Uploading to MinIO S3...")

df.to_parquet("dataset_metadata.parquet", index=False)
try:
    if s3 is None:
        raise ValueError("No S3 client")
    s3.upload_file(
        "dataset_metadata.parquet",
        "datasets",
        "metadata/dataset_metadata.parquet"
    )
except Exception:
    print("!!! Garage is not available, fallback used!!!")

print("Done!")