"""Tests for Phase 5 DICOM UI Routes."""


class TestDicomUIRoutes:
    """Test DICOM UI routes."""

    def test_upload_requires_auth(self, client):
        """GET /imaging/dicom/ui/upload requires authentication."""
        resp = client.get("/imaging/dicom/ui/upload", follow_redirects=False)
        assert resp.status_code == 302  # Redirect to login

    def test_upload_with_admin(self, client, admin_user):
        """GET /imaging/dicom/ui/upload returns 200 for admin."""
        resp = client.get("/imaging/dicom/ui/upload", follow_redirects=True)
        assert resp.status_code == 200
        assert b"Upload DICOM" in resp.data or b"patient_id" in resp.data

    def test_studies_search_requires_auth(self, client):
        """GET /imaging/dicom/ui/studies requires authentication."""
        resp = client.get("/imaging/dicom/ui/studies", follow_redirects=False)
        assert resp.status_code == 302

    def test_studies_search_with_admin(self, client, admin_user):
        """GET /imaging/dicom/ui/studies returns 200 for admin."""
        resp = client.get("/imaging/dicom/ui/studies", follow_redirects=True)
        assert resp.status_code == 200
        assert b"Browse DICOM Studies" in resp.data or b"patient_id" in resp.data

    def test_studies_detail_requires_auth(self, client):
        """GET /imaging/dicom/ui/studies/<patient_id> requires authentication."""
        resp = client.get("/imaging/dicom/ui/studies/TEST001", follow_redirects=False)
        assert resp.status_code == 302

    def test_studies_detail_with_admin(self, client, admin_user):
        """GET /imaging/dicom/ui/studies/<patient_id> returns 200 for admin."""
        resp = client.get("/imaging/dicom/ui/studies/TEST001", follow_redirects=True)
        assert resp.status_code == 200
        assert b"DICOM Studies" in resp.data or b"Study UID" in resp.data
