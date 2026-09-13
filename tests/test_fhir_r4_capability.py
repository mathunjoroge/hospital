"""
tests/test_fhir_r4_capability.py
──────────────────────────────────
Unit and integration tests for FHIR R4 CapabilityStatement, SMART Discovery,
Patient Search, and Batch Bundle processing (Gap #7).
"""

from datetime import date
import pytest
from flask_jwt_extended import create_access_token
from werkzeug.security import generate_password_hash

from app import app
from extensions import db
from departments.models.user import User
from departments.models.records import Patient
from departments.models.nursing import Vitals
from departments.models.encounter import Encounter


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            yield client


@pytest.fixture
def auth_headers(client):
    """Simulate authenticated session/JWT headers."""
    with app.app_context():
        user = User.query.filter_by(username="admin_fhir_test").first()
        if not user:
            user = User(
                username="admin_fhir_test",
                password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
                role="admin",
            )
            db.session.add(user)
            db.session.commit()
        token = create_access_token(identity=str(user.id))
    return {"Authorization": f"Bearer {token}"}


def _make_patient(patient_id: str, name: str = "Test Patient", sex: str = "Female") -> Patient:
    """Insert a minimal patient into DB context."""
    p = Patient.query.filter_by(patient_id=patient_id).first()
    if not p:
        p = Patient(
            patient_id=patient_id,
            name=name,
            sex=sex,
            date_of_birth=date(1990, 1, 1),
            emergency_contact="0700000000",
        )
        db.session.add(p)
        db.session.commit()
    return p


def test_fhir_metadata_capability_statement(client):
    """Test GET /api/fhir/R4/metadata returns a valid CapabilityStatement."""
    res = client.get("/api/fhir/R4/metadata")
    assert res.status_code == 200
    data = res.get_json()
    assert data["resourceType"] == "CapabilityStatement"
    assert data["fhirVersion"] == "4.0.1"
    assert data["status"] == "active"
    
    resource_types = [r["type"] for r in data["rest"][0]["resource"]]
    expected = ["Patient", "Observation", "Condition", "DiagnosticReport", "MedicationRequest", "Encounter", "ImagingStudy"]
    for exp in expected:
        assert exp in resource_types


def test_smart_on_fhir_discovery(client):
    """Test SMART-on-FHIR OAuth2 discovery configuration."""
    # Test top-level route
    res1 = client.get("/.well-known/smart-configuration")
    assert res1.status_code == 200
    data1 = res1.get_json()
    assert "authorization_endpoint" in data1
    assert "token_endpoint" in data1
    assert "patient/*.read" in data1["scopes_supported"]

    # Test blueprint route
    res2 = client.get("/api/fhir/R4/.well-known/smart-configuration")
    assert res2.status_code == 200
    data2 = res2.get_json()
    assert data2["issuer"].endswith("/api/fhir/R4")


def test_fhir_patient_search(client, auth_headers):
    """Test searching FHIR R4 Patients by name, identifier, or gender."""
    with app.app_context():
        _make_patient("PAT-FHIR-1", name="Alice FHIR", sex="Female")
        _make_patient("PAT-FHIR-2", name="Bob FHIR", sex="Male")

    # Search by name
    res = client.get("/api/fhir/R4/Patient?name=Alice", headers=auth_headers)
    assert res.status_code == 200
    data = res.get_json()
    assert data["resourceType"] == "Bundle"
    assert data["type"] == "searchset"
    assert data["total"] == 1
    assert data["entry"][0]["resource"]["id"] == "PAT-FHIR-1"

    # Search by patient ID
    res2 = client.get("/api/fhir/R4/Patient?patient=PAT-FHIR-2", headers=auth_headers)
    assert res2.status_code == 200
    data2 = res2.get_json()
    assert data2["total"] == 1
    assert data2["entry"][0]["resource"]["id"] == "PAT-FHIR-2"


def test_fhir_batch_bundle_processing(client, auth_headers):
    """Test processing a FHIR Batch Bundle containing multi-resource export queries."""
    with app.app_context():
        _make_patient("PAT-BATCH-1", name="Charlie Batch", sex="Male")
        v = Vitals(patient_id="PAT-BATCH-1", nurse_id=1, temperature=37.2, pulse=78)
        enc = Encounter(encounter_id="ENC-BATCH-1", patient_id="PAT-BATCH-1", status="ACTIVE")
        db.session.add_all([v, enc])
        db.session.commit()

    batch_bundle = {
        "resourceType": "Bundle",
        "type": "batch",
        "entry": [
            {
                "request": {
                    "method": "GET",
                    "url": "Patient/PAT-BATCH-1"
                }
            },
            {
                "request": {
                    "method": "GET",
                    "url": "Observation?patient=PAT-BATCH-1"
                }
            },
            {
                "request": {
                    "method": "GET",
                    "url": "Encounter?patient=PAT-BATCH-1"
                }
            }
        ]
    }

    res = client.post("/api/fhir/R4/", json=batch_bundle, headers=auth_headers)
    assert res.status_code == 200
    data = res.get_json()
    assert data["resourceType"] == "Bundle"
    assert data["type"] == "batch-response"
    assert data["total"] == 3

    # Check Patient read entry
    assert data["entry"][0]["response"]["status"] == "200 OK"
    assert data["entry"][0]["resource"]["id"] == "PAT-BATCH-1"

    # Check Observation search entry
    assert data["entry"][1]["response"]["status"] == "200 OK"
    assert data["entry"][1]["resource"]["resourceType"] == "Bundle"

    # Check Encounter search entry
    assert data["entry"][2]["response"]["status"] == "200 OK"
    assert data["entry"][2]["resource"]["entry"][0]["resource"]["id"] == "ENC-BATCH-1"
