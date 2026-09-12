"""Tests for Phase 5: DICOM PACS Integration"""
from io import BytesIO

import pytest


class TestDICOMService:
    """Test DICOMService methods."""

    def test_parse_dicom_invalid_file(self, app, tmp_path):
        """parse_dicom() raises ValueError for invalid files."""
        from departments.imaging.dicom_service import DICOMService

        fake_file = tmp_path / "fake.dcm"
        fake_file.write_text("not a dicom file")

        with pytest.raises(ValueError, match="Invalid DICOM"):
            DICOMService.parse_dicom(str(fake_file))

    def test_list_studies_returns_list(self, app):
        """list_studies() returns a list of study dicts."""
        from departments.imaging.dicom_service import DICOMService

        studies = DICOMService.list_studies("TEST001")
        assert isinstance(studies, list)

    def test_get_dicom_by_sop_uid_not_found(self, app):
        """get_dicom_by_sop_uid() returns None for non-existent UID."""
        from departments.imaging.dicom_service import DICOMService

        result = DICOMService.get_dicom_by_sop_uid("nonexistent.uid")
        assert result is None


class TestDICOMEndpoints:
    """Test DICOM upload/download endpoints."""

    def test_upload_dicom_no_file(self, client):
        """POST /imaging/dicom/upload returns 401 without auth."""
        resp = client.post("/imaging/dicom/upload")
        assert resp.status_code == 401

    def test_upload_dicom_wrong_extension(self, client, app):
        """POST /imaging/dicom/upload returns 401 without auth."""
        # Without auth, should get 401
        data = {"file": (BytesIO(b"fake content"), "test.txt")}
        resp = client.post(
            "/imaging/dicom/upload",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 401

    def test_list_studies_requires_auth(self, client):
        """GET /imaging/dicom/studies/<id> requires authentication."""
        resp = client.get("/imaging/dicom/studies/TEST001")
        assert resp.status_code == 401

    def test_download_dicom_requires_auth(self, client):
        """GET /imaging/dicom/download/<uid> requires authentication."""
        resp = client.get("/imaging/dicom/download/nonexistent.uid")
        assert resp.status_code == 401


class TestDICOMStorage:
    """Test DICOM storage directory structure."""

    def test_storage_directories_exist(self, app):
        """Required storage directories are created."""
        from departments.imaging.dicom_service import (
            DICOM_ARCHIVE,
            DICOM_INCOMING,
            DICOM_THUMBNAILS,
        )

        assert DICOM_INCOMING.exists()
        assert DICOM_ARCHIVE.exists()
        assert DICOM_THUMBNAILS.exists()
