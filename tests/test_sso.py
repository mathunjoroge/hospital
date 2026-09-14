"""
tests/test_sso.py
──────────────────
Unit and integration tests for Enterprise LDAP / Active Directory & OIDC Single Sign-On (SSO).
"""
from departments.sso.sso_engine import SSOEngine


def test_sso_status_endpoint(client):
    """Test /auth/sso/status endpoint returns SSO health & configuration."""
    response = client.get("/auth/sso/status")
    assert response.status_code == 200
    data = response.get_json()
    assert "sso_enabled" in data
    assert "provider" in data
    assert "supported_methods" in data


def test_oidc_login_redirect(client):
    """Test /auth/sso/login redirects to enterprise IdP authorization URL."""
    response = client.get("/auth/sso/login", follow_redirects=False)
    assert response.status_code == 302
    assert "login.microsoftonline.com" in response.headers["Location"]


def test_oidc_callback_provisions_user(client, app):
    """Test /auth/sso/callback exchanges code and provisions user."""
    headers = {"Accept": "application/json"}
    response = client.get("/auth/sso/callback?code=mock_auth_code_123", headers=headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["username"] == "dr.johnson"
    assert data["role"] == "doctor"


def test_ldap_bind_authentication(client, app):
    """Test /auth/sso/ldap authenticates user against Active Directory."""
    payload = {
        "username": "nurse.mary",
        "password": "ValidADPassword123!",
    }
    response = client.post("/auth/sso/ldap", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["username"] == "nurse.mary"
    assert data["role"] == "nursing"


def test_ldap_bind_invalid_credentials(client):
    """Test /auth/sso/ldap rejects invalid credentials."""
    payload = {
        "username": "nurse.mary",
        "password": "WrongPassword!",
    }
    response = client.post("/auth/sso/ldap", json=payload)
    assert response.status_code == 401
    data = response.get_json()
    assert "Invalid Active Directory credentials" in data["error"]


def test_ad_group_mapping_logic():
    """Test Active Directory group string to HIMS role mapping."""
    engine = SSOEngine()

    admin_groups = ["CN=HIMS-Admins,OU=Groups,DC=hospital,DC=org"]
    assert engine.map_ad_groups_to_role(admin_groups) == "admin"

    doctor_groups = ["CN=HIMS-Doctors,OU=Groups,DC=hospital,DC=org"]
    assert engine.map_ad_groups_to_role(doctor_groups) == "doctor"

    nurse_groups = ["CN=HIMS-Nurses,OU=Groups,DC=hospital,DC=org"]
    assert engine.map_ad_groups_to_role(nurse_groups) == "nursing"

    pharm_groups = ["CN=HIMS-Pharmacists,OU=Groups,DC=hospital,DC=org"]
    assert engine.map_ad_groups_to_role(pharm_groups) == "pharmacy"
