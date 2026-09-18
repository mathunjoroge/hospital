"""
tests/test_medical_file_storage.py
───────────────────────────────────
Unit tests for MedicalFileStorage (MinIO object storage & local fallback - Section 7).
"""

from departments.storage.minio_client import MedicalFileStorage


def test_medical_file_storage_fallback():
    storage = MedicalFileStorage()
    sample_pdf = b"%PDF-1.4 sample medical report bytes"

    res = storage.store_file(
        filename="test_report_001.pdf",
        file_bytes=sample_pdf,
        content_type="application/pdf",
        bucket_name="test-lab-reports",
    )
    assert res is not None
    assert res["size_bytes"] == len(sample_pdf)
    assert res["filename"] == "test_report_001.pdf"
    assert res["storage_backend"] in ("minio", "local_fallback")
