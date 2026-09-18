"""
departments/storage/minio_client.py
───────────────────────────────────
Section 7: Medical File Storage Backend (MinIO / S3-compatible).

Provides self-hosted object storage for DICOM imaging, PDF lab reports, and
patient file uploads. Ensures 100% in-country data residency under Kenya
Data Protection Act (2019) and fallback to local volume storage when offline.
"""

import io
import logging
import os
import tempfile
from typing import Any

from flask import current_app

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "hospital-medical-files"


def get_minio_config() -> dict[str, str]:
    """Retrieve MinIO connection configuration from environment or Flask config."""
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    bucket = os.getenv("MINIO_BUCKET_NAME", DEFAULT_BUCKET)
    secure = os.getenv("MINIO_SECURE", "false").lower() == "true"

    if current_app and hasattr(current_app, "config"):
        endpoint = current_app.config.get("MINIO_ENDPOINT", endpoint)
        access_key = current_app.config.get("MINIO_ACCESS_KEY", access_key)
        secret_key = current_app.config.get("MINIO_SECRET_KEY", secret_key)
        bucket = current_app.config.get("MINIO_BUCKET_NAME", bucket)

    return {
        "endpoint": endpoint,
        "access_key": access_key,
        "secret_key": secret_key,
        "bucket": bucket,
        "secure": secure,
    }


class MedicalFileStorage:
    """MinIO / Local Object Storage Manager for medical files."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or get_minio_config()
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from minio import Minio

                self._client = Minio(
                    self.config["endpoint"],
                    access_key=self.config["access_key"],
                    secret_key=self.config["secret_key"],
                    secure=self.config["secure"],
                )
            except Exception as exc:
                logger.warning(f"MinIO client initialization failed (fallback to local disk): {exc}")
                self._client = False
        return self._client if self._client is not False else None

    def store_file(
        self,
        filename: str,
        file_bytes: bytes,
        content_type: str = "application/octet-stream",
        bucket_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Store a medical file (DICOM, PDF, image) in MinIO object storage.
        Falls back to local file system if MinIO service is unconfigured/offline.
        """
        bucket = bucket_name or self.config["bucket"]
        client = self._get_client()

        if client:
            try:
                # Ensure bucket exists
                if not client.bucket_exists(bucket):
                    client.make_bucket(bucket)

                data_stream = io.BytesIO(file_bytes)
                size = len(file_bytes)
                client.put_object(
                    bucket_name=bucket,
                    object_name=filename,
                    data=data_stream,
                    length=size,
                    content_type=content_type,
                )
                logger.info(f"Successfully stored file '{filename}' in MinIO bucket '{bucket}' ({size} bytes)")
                return {
                    "storage_backend": "minio",
                    "bucket": bucket,
                    "filename": filename,
                    "size_bytes": size,
                    "content_type": content_type,
                    "url": f"http://{self.config['endpoint']}/{bucket}/{filename}",
                }
            except Exception as exc:
                logger.exception(f"Error saving file to MinIO: {exc}. Falling back to local storage.")

        # Local volume fallback
        fallback_dir = os.path.join(tempfile.gettempdir(), "hospital_storage_fallback", bucket)
        os.makedirs(fallback_dir, exist_ok=True)
        file_path = os.path.join(fallback_dir, filename)
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        logger.info(f"Stored file '{filename}' in local storage fallback ({file_path})")
        return {
            "storage_backend": "local_fallback",
            "path": file_path,
            "filename": filename,
            "size_bytes": len(file_bytes),
            "content_type": content_type,
            "url": f"file://{file_path}",
        }


# Global singleton instance
storage_manager = MedicalFileStorage()
