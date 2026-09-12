"""Tests for Phase 7 TB/DOTS UI Routes."""


class TestTbDotsUIRoutes:
    """Test TB/DOTS UI routes."""

    def test_dashboard_requires_auth(self, client):
        """GET /tb_dots/ui/dashboard requires authentication."""
        resp = client.get("/tb_dots/ui/dashboard", follow_redirects=False)
        assert resp.status_code == 302  # Redirect to login

    def test_dashboard_with_admin(self, client, admin_user):
        """GET /tb_dots/ui/dashboard returns 200 for admin."""
        resp = client.get("/tb_dots/ui/dashboard", follow_redirects=True)
        assert resp.status_code == 200
        assert b"TB/DOTS Program Dashboard" in resp.data or b"Total Enrolled" in resp.data

    def test_enroll_requires_auth(self, client):
        """GET /tb_dots/ui/enroll requires authentication."""
        resp = client.get("/tb_dots/ui/enroll", follow_redirects=False)
        assert resp.status_code == 302

    def test_enroll_with_admin(self, client, admin_user):
        """GET /tb_dots/ui/enroll returns 200 for admin."""
        resp = client.get("/tb_dots/ui/enroll", follow_redirects=True)
        assert resp.status_code == 200
        assert b"TB/DOTS Patient Enrollment" in resp.data or b"patient_id" in resp.data

    def test_patient_detail_requires_auth(self, client):
        """GET /tb_dots/ui/patient/<id> requires authentication."""
        resp = client.get("/tb_dots/ui/patient/1", follow_redirects=False)
        assert resp.status_code == 302

    def test_patient_detail_with_admin(self, client, admin_user):
        """GET /tb_dots/ui/patient/<id> returns 200 for admin."""
        resp = client.get("/tb_dots/ui/patient/1", follow_redirects=True)
        assert resp.status_code == 200
        assert b"TB Patient" in resp.data or b"Patient Information" in resp.data
