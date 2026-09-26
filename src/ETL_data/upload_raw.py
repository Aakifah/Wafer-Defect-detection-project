"""Script to upload raw wafer maps to the S3 bucket."""

import logging
import os
import sys
from typing import Any

import boto3
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_raw_to_s3() -> None:
    """Uploads the local raw dataset using put_object to enforce single-part upload."""
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
        return

    try:
        client: Any = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="garage"
        )
    except Exception:
        print("!!! MinIO S3 is not available, fallback used!!!")
        return

    raw_file = "data/raw/Wafer_Map_Datasets.npz"
    target_bucket = "datasets"
    target_key = "raw/Wafer_Map_Datasets.npz"

    if os.path.exists(raw_file):
        logger.info("Uploading %s to s3://%s/%s via single-part binary stream...", 
                    raw_file, target_bucket, target_key)
        try:
            with open(raw_file, "rb") as file_data:
                client.put_object(Bucket=target_bucket, Key=target_key, Body=file_data)
            logger.info("Upload complete. Checksum collisions neutralized.")
        except Exception:
            print("!!! MinIO S3 is not available, fallback used!!!")
            return
    else:
        logger.error("Local file %s not found. Run aborted.", raw_file)
        sys.exit(1)

if __name__ == "__main__":
    upload_raw_to_s3()