"""Tests for Phase 7 Malaria UI Routes."""


class TestMalariaUIRoutes:
    """Test Malaria UI routes."""

    def test_dashboard_requires_auth(self, client):
        """GET /malaria/ui/dashboard requires authentication."""
        resp = client.get("/malaria/ui/dashboard", follow_redirects=False)
        assert resp.status_code == 302  # Redirect to login

    def test_dashboard_with_admin(self, client, admin_user):
        """GET /malaria/ui/dashboard returns 200 for admin."""
        resp = client.get("/malaria/ui/dashboard", follow_redirects=True)
        assert resp.status_code == 200
        assert b"Malaria Case Management" in resp.data or b"Total Cases" in resp.data

    def test_new_case_requires_auth(self, client):
        """GET /malaria/ui/new-case requires authentication."""
        resp = client.get("/malaria/ui/new-case", follow_redirects=False)
        assert resp.status_code == 302

    def test_new_case_with_admin(self, client, admin_user):
        """GET /malaria/ui/new-case returns 200 for admin."""
        resp = client.get("/malaria/ui/new-case", follow_redirects=True)
        assert resp.status_code == 200
        assert b"New Malaria Case Registration" in resp.data or b"patient_id" in resp.data

    def test_case_detail_requires_auth(self, client):
        """GET /malaria/ui/case/<id> requires authentication."""
        resp = client.get("/malaria/ui/case/1", follow_redirects=False)
        assert resp.status_code == 302

    def test_case_detail_with_admin(self, client, admin_user):
        """GET /malaria/ui/case/<id> returns 200 for admin."""
        resp = client.get("/malaria/ui/case/1", follow_redirects=True)
        assert resp.status_code == 200
        assert b"Malaria Case" in resp.data or b"Case Information" in resp.data
