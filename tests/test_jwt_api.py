"""
tests/test_jwt_api.py
──────────────────────
Test suite for JWT authentication & dual auth (JWT + Session) on API endpoints.
"""

import pytest
from datetime import date
from werkzeug.security import generate_password_hash
from extensions import db
from departments.models.user import User
from departments.models.records import Patient


@pytest.fixture
def jwt_client(client):
    """Seed test user and patient in the test database for JWT testing."""
    admin = User.query.filter_by(username="api_jwt_user").first()
    if not admin:
        admin = User(
            username="api_jwt_user",
            password=generate_password_hash("AdminPassword123!", method="pbkdf2:sha256"),
            role="admin",
        )
        db.session.add(admin)

    patient = Patient.query.filter_by(patient_id="P-JWT-001").first()
    if not patient:
        patient = Patient(
            patient_id="P-JWT-001",
            name="JWT Test Patient",
            sex="M",
            date_of_birth=date(1990, 1, 1),
            contact="0700000000",
            national_id="88776655",
            place_of_residence="Nairobi",
            marital_status="Single",
            next_of_kin="Kin",
            relationship_with_next_of_kin="Brother",
            next_of_kin_contact="0711111111",
            emergency_contact="0711111111",
        )
        db.session.add(patient)

    db.session.commit()
    return client


def test_jwt_token_issuance_success(jwt_client):
    """Test getting a JWT access token with valid credentials."""
    res = jwt_client.post(
        "/api/auth/token",
        json={"username": "api_jwt_user", "password": "AdminPassword123!"},
    )
    assert res.status_code == 200
    data = res.get_json()
    assert "access_token" in data
    assert data["token_type"] == "Bearer"
    assert data["user"]["username"] == "api_jwt_user"


def test_jwt_token_issuance_invalid_credentials(jwt_client):
    """Test getting a JWT token with invalid credentials fails with 401."""
    res = jwt_client.post(
        "/api/auth/token",
        json={"username": "api_jwt_user", "password": "WrongPassword!"},
    )
    assert res.status_code == 401
    data = res.get_json()
    assert "error" in data


def test_jwt_authenticated_whoami(jwt_client):
    """Test accessing /api/auth/me using a Bearer token."""
    token_res = jwt_client.post(
        "/api/auth/token",
        json={"username": "api_jwt_user", "password": "AdminPassword123!"},
    )
    token = token_res.get_json()["access_token"]

    res = jwt_client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["username"] == "api_jwt_user"


def test_fhir_endpoint_with_jwt(jwt_client):
    """Test FHIR R4 Patient endpoint with Bearer token authentication."""
    token_res = jwt_client.post(
        "/api/auth/token",
        json={"username": "api_jwt_user", "password": "AdminPassword123!"},
    )
    token = token_res.get_json()["access_token"]

    res = jwt_client.get(
        "/api/fhir/R4/Patient/P-JWT-001",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data.get("resourceType") == "Patient"
    assert data.get("id") == "P-JWT-001"


def test_fhir_endpoint_unauthenticated(jwt_client):
    """Test FHIR R4 Patient endpoint without authentication returns 401."""
    res = jwt_client.get("/api/fhir/R4/Patient/P-JWT-001")
    assert res.status_code == 401
