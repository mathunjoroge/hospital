"""
tests/test_fhir_hie_subscriptions.py
──────────────────────────────────────
Unit and integration tests for Multi-Facility HIE FHIR $everything exporter and Subscription webhooks.
"""

from unittest.mock import MagicMock, patch

from departments.api.fhir_hie import export_patient_everything_bundle
from departments.api.fhir_subscriptions import (
    create_subscription,
    delete_subscription,
    dispatch_subscription_event,
    generate_hmac_signature,
    list_subscriptions,
)
from departments.models.encounter import Encounter
from departments.models.medicine import PrescribedMedicine, SOAPNote
from departments.models.nursing import Vitals
from departments.models.records import Patient
from extensions import db


def test_hmac_signature_generation():
    """Test HMAC-SHA256 signature generation."""
    secret = "my_secret_token_123"
    payload = b'{"test": "data"}'
    sig = generate_hmac_signature(secret, payload)
    assert sig.startswith("sha256=")
    assert len(sig) == 7 + 64  # sha256= + 64 hex chars


def test_export_patient_everything_bundle(app):
    """Test FHIR R4 Patient/$everything bundle exporter across clinical modules."""
    with app.app_context():
        # Setup patient and clinical records
        from datetime import date

        pat = Patient(
            patient_id="PAT_HIE_01",
            name="Jane HIE Doe",
            sex="Female",
            date_of_birth=date(1990, 1, 1),
        )
        db.session.add(pat)

        enc = Encounter(
            patient_id="PAT_HIE_01", encounter_type="Inpatient", status="active"
        )
        soap = SOAPNote(
            patient_id="PAT_HIE_01",
            situation="Fever and sore throat",
            hpi="Acute onset fever 38.5°C for 2 days",
            assessment="Acute Pharyngitis",
            recommendation="Rest and adequate hydration",
        )
        vitals = Vitals(patient_id="PAT_HIE_01", nurse_id=1, temperature=38.5, pulse=90)
        rx = PrescribedMedicine(
            patient_id="PAT_HIE_01",
            medicine_id=1,
            dosage="500mg",
            strength="500mg",
            frequency="BD",
            num_days=5,
            prescription_id="rx-hie-01",
        )

        db.session.add_all([enc, soap, vitals, rx])
        db.session.commit()

        bundle = export_patient_everything_bundle("PAT_HIE_01")
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "collection"
        assert bundle["total"] >= 4

        resource_types = [e["resource"]["resourceType"] for e in bundle["entry"]]
        assert "Patient" in resource_types
        assert "Encounter" in resource_types
        assert "Condition" in resource_types
        assert "Observation" in resource_types
        assert "MedicationRequest" in resource_types

        # Test non-existent patient
        nf_bundle = export_patient_everything_bundle("PAT_NON_EXISTENT")
        assert nf_bundle["resourceType"] == "OperationOutcome"
        assert nf_bundle["status"] == 404


def test_fhir_subscription_crud(app):
    """Test FHIRSubscription database CRUD operations."""
    with app.app_context():
        sub = create_subscription(
            criteria="Observation",
            endpoint_url="https://hie.national-health.go.ke/webhooks/vitals",
            secret_token="secret_key_777",
            reason="National Vitals Sync",
        )
        assert sub.status == "active"

        active_subs = list_subscriptions()
        assert len(active_subs) >= 1
        assert active_subs[0]["criteria"] == "Observation"

        del_ok = delete_subscription(sub.subscription_id)
        assert del_ok is True
        assert delete_subscription("non_existent_id") is False


@patch("departments.api.fhir_subscriptions.requests.post")
def test_dispatch_subscription_event(mock_post, app):
    """Test webhook event dispatcher with HMAC signature header."""
    with app.app_context():
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Create active subscription
        create_subscription(
            criteria="Encounter",
            endpoint_url="https://hie.partner-hospital.com/events",
            secret_token="hmac_secret_99",
        )

        obs_data = {"resourceType": "Encounter", "id": "enc-101", "status": "finished"}
        count = dispatch_subscription_event("Encounter", obs_data)
        assert count == 1

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://hie.partner-hospital.com/events"
        headers = call_args[1]["headers"]
        assert "X-FHIR-Signature" in headers
        assert headers["X-FHIR-Signature"].startswith("sha256=")


def test_fhir_hie_routes(client, app):
    """Test FHIR $everything and Subscription HTTP API endpoints."""
    with app.app_context():
        from datetime import date

        pat = Patient(
            patient_id="PAT_API_HIE",
            name="John API Doe",
            sex="Male",
            date_of_birth=date(1985, 6, 15),
        )
        db.session.add(pat)
        db.session.commit()

    # Test GET $everything endpoint
    resp_e = client.get("/api/fhir/R4/Patient/PAT_API_HIE/$everything")
    assert resp_e.status_code == 200
    data_e = resp_e.get_json()
    assert data_e["resourceType"] == "Bundle"

    # Test POST Subscription
    resp_sub = client.post(
        "/api/fhir/R4/Subscription",
        json={
            "criteria": "Observation",
            "channel": {"endpoint": "https://hie.test.org/webhook"},
            "secret_token": "token_abc",
        },
    )
    assert resp_sub.status_code == 201
    sub_data = resp_sub.get_json()
    assert sub_data["resourceType"] == "Subscription"
    sub_id = sub_data["id"]

    # Test GET Subscriptions list
    resp_list = client.get("/api/fhir/R4/Subscription")
    assert resp_list.status_code == 200

    # Test DELETE Subscription
    resp_del = client.delete(f"/api/fhir/R4/Subscription/{sub_id}")
    assert resp_del.status_code == 200
