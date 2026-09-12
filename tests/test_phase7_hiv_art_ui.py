"""Tests for Phase 7 HIV/ART UI Routes."""


class TestHivArtUIRoutes:
    """Test HIV/ART UI routes."""

    def test_dashboard_requires_auth(self, client):
        """GET /hiv_art/ui/dashboard requires authentication."""
        resp = client.get("/hiv_art/ui/dashboard", follow_redirects=False)
        assert resp.status_code == 302  # Redirect to login

    def test_dashboard_with_admin(self, client, admin_user):
        """GET /hiv_art/ui/dashboard returns 200 for admin."""
        resp = client.get("/hiv_art/ui/dashboard", follow_redirects=True)
        assert resp.status_code == 200
        assert b"HIV/ART Program Dashboard" in resp.data or b"Total Enrolled" in resp.data

    def test_enroll_requires_auth(self, client):
        """GET /hiv_art/ui/enroll requires authentication."""
        resp = client.get("/hiv_art/ui/enroll", follow_redirects=False)
        assert resp.status_code == 302

    def test_enroll_with_admin(self, client, admin_user):
        """GET /hiv_art/ui/enroll returns 200 for admin."""
        resp = client.get("/hiv_art/ui/enroll", follow_redirects=True)
        assert resp.status_code == 200
        assert b"HIV/ART Patient Enrollment" in resp.data or b"patient_id" in resp.data

    def test_patient_detail_requires_auth(self, client):
        """GET /hiv_art/ui/patient/<id> requires authentication."""
        resp = client.get("/hiv_art/ui/patient/1", follow_redirects=False)
        assert resp.status_code == 302

    def test_patient_detail_with_admin(self, client, admin_user):
        """GET /hiv_art/ui/patient/<id> returns 200 for admin."""
        resp = client.get("/hiv_art/ui/patient/1", follow_redirects=True)
        assert resp.status_code == 200
        assert b"ART Patient" in resp.data or b"Patient Information" in resp.data
