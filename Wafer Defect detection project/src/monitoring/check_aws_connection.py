"""Diagnostic script to verify local MinIO/S3 connection health."""

import http.client
import logging
import os
import sys
from typing import Any
from urllib.parse import urlparse

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PORT: int = 9000
DEFAULT_HOST: str = "localhost"
DEFAULT_ENDPOINT: str = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
AWS_REGION: str = "us-east-1"
HTTP_TIMEOUT_SEC: int = 5


def check_http_health(endpoint: str) -> dict[str, Any] | None:
    logger.info("Testing raw HTTP GET to %s...", endpoint)
    try:
        parsed = urlparse(endpoint)
        host = parsed.hostname or DEFAULT_HOST
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = parsed.path or "/"

        conn = http.client.HTTPConnection(host, port, timeout=HTTP_TIMEOUT_SEC)
        conn.request("GET", path)
        response = conn.getresponse()

        logger.info("HTTP %s %s", response.status, response.reason)
        body = response.read(200).decode("utf-8", errors="ignore")
        logger.info("Response preview: %s", body[:100])
        conn.close()
        return {"status": response.status, "reason": response.reason, "body_preview": body[:100]}
    except (OSError, http.client.HTTPException) as e:
        logger.error("HTTP test failed: %s: %s", type(e).__name__, e)
        return None


def check_s3_compatibility(endpoint: str, access_key: str, secret_key: str) -> bool:
    logger.info("Testing S3 list_buckets() via %s...", endpoint)
    try:
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=AWS_REGION,
            config=boto3.session.Config(signature_version="s3v4"),
        )
        response = client.list_buckets()
        buckets_found = len(response.get("Buckets", []))
        logger.info("S3 list_buckets succeeded: %s buckets found", buckets_found)
        return True
    except (BotoCoreError, ClientError) as e:
        logger.error("S3 test failed: %s: %s", type(e).__name__, e)
        err_str = str(e).lower()
        if "connection" in err_str and ("closed" in err_str or "refused" in err_str):
            logger.info(
                "This is a NETWORK error. Make sure MinIO is running: docker compose up -d minio"
            )
        elif "access" in err_str or "denied" in err_str:
            logger.info(
                "This is an AUTH error. Check AWS_ACCESS_KEY_ID/SECRET in .env"
            )
        elif "signature" in err_str or "v4" in err_str:
            logger.info(
                "This is a SIGNING error. MinIO requires signature_version='s3v4'"
            )
        elif "nosuchbucket" in err_str:
            logger.info(
                "Bucket doesn't exist yet. That's expected for first run!"
            )
        return False


def check_env_vars() -> dict[str, bool]:
    logger.info("Checking environment variables...")
    required = ["AWS_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    result: dict[str, bool] = {}
    for var in required:
        val = os.getenv(var)
        if val:
            logger.info("%s is set (value preview: %s...)", var, val[:20])
            result[var] = True
        else:
            logger.error("%s is MISSING", var)
            result[var] = False
    return result


def main() -> int:
    print("Starting S3/MinIO Connection Diagnostic...\n")

    load_dotenv()

    endpoint = os.getenv("AWS_ENDPOINT_URL", DEFAULT_ENDPOINT)
    access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")

    print(f"Target endpoint: {endpoint}\n")

    http_ok = check_http_health(endpoint)
    print()

    env_ok = check_env_vars()
    print()

    if http_ok and all(env_ok.values()):
        s3_ok = check_s3_compatibility(endpoint, access_key, secret_key)
    else:
        logger.warning("Skipping S3 test due to earlier failures")
        s3_ok = False

    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Raw HTTP Response      : {'PASS' if http_ok else 'FAIL'}")
    print(f"Environment Variables  : {'PASS' if all(env_ok.values()) else 'FAIL'}")
    print(f"S3 Protocol Test       : {'PASS' if s3_ok else 'FAIL'}")
    print("=" * 60)

    if not http_ok:
        print("\nCRITICAL: No HTTP response from MinIO.")
        print("Possible causes:")
        print("  1. MinIO container is not running. Start it with: docker compose up -d minio")
        print("  2. MinIO API port changed. Check AWS_ENDPOINT_URL in .env")
        return 1
    elif not all(env_ok.values()):
        print("\nCRITICAL: Missing environment variables.")
        print("Create a .env file with the following variables:")
        print(f"  AWS_ENDPOINT_URL={DEFAULT_ENDPOINT}")
        print("  AWS_ACCESS_KEY_ID=minioadmin")
        print("  AWS_SECRET_ACCESS_KEY=minioadmin")
        return 1
    elif not s3_ok:
        print("\nS3 protocol test failed but HTTP works.")
        print("This is often a configuration issue:")
        print("  - MinIO requires signature_version='s3v4' in boto3 config")
        print(f"  - MinIO may need explicit region_name='{AWS_REGION}'")
        print("  - Check MinIO logs: docker logs wafer-minio")
        return 1
    else:
        print("\nAll checks passed! Your S3/MinIO connection is healthy.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
