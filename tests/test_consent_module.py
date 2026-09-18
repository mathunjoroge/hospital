"""
tests/test_consent_module.py
─────────────────────────────
Consent capture/query endpoints and the gates that depend on them.

Context: /consent/api/grant and /consent/api/revoke previously returned 200/201
with {"message": "Consent grant stub active."} and wrote nothing, while
/consent/api/patient/<id> returned a "Full query logic in Phase 1.1" placeholder.
Separately, two consent tables existed — `consents` (read by the prescribing
gate, written by nothing) and `patient_consents` (everything else) — so a
consent granted anywhere was invisible to the prescribing gate by construction.
"""

from datetime import date, datetime

import pytest
from werkzeug.security import generate_password_hash

from departments.models.compliance import (
    grant_patient_consent,
    has_ai_consent,
    has_consent,
    list_patient_consents,
    revoke_patient_consent,
)
from departments.models.medicine import AdmittedPatient, Ward
from departments.models.records import Patient
from departments.models.user import User
from extensions import db


@pytest.fixture
def patient(app):
    p = Patient(
        patient_id="P0001",
        name="Consent Tester",
        sex="M",
        date_of_birth=date(1985, 1, 1),
    )
    db.session.add(p)
    db.session.commit()
    return p


@pytest.fixture
def records_client(client, app):
    db.session.add(
        User(
            username="consent_clerk",
            role="records",
            password=generate_password_hash("Clerk!2345"),
        )
    )
    db.session.commit()
    client.post(
        "/login",
        data={"username": "consent_clerk", "password": "Clerk!2345"},
        follow_redirects=True,
    )
    return client


@pytest.fixture
def lab_client(client, app):
    """A role with no consent-capture authority."""
    db.session.add(
        User(
            username="consent_labtech",
            role="laboratory",
            password=generate_password_hash("Lab!2345"),
        )
    )
    db.session.commit()
    client.post(
        "/login",
        data={"username": "consent_labtech", "password": "Lab!2345"},
        follow_redirects=True,
    )
    return client


# ── Endpoints must do the work, not report that they would ────────────────


def test_grant_persists_a_real_record(records_client, patient):
    """Regression: this endpoint used to return 201 and write nothing."""
    response = records_client.post(
        "/consent/api/grant",
        json={
            "patient_id": "P0001",
            "consent_type": "TREATMENT",
            "notes": "signed in clinic",
        },
    )
    assert response.status_code == 201
    assert response.get_json()["consent"]["status"] == "ACTIVE"

    # The write is visible to the authoritative gate, not just the response.
    assert has_consent("P0001", "TREATMENT") is True
    assert len(list_patient_consents("P0001")) == 1


def test_no_endpoint_returns_a_stub_message(records_client, patient):
    """No consent endpoint may answer with placeholder prose."""
    grant = records_client.post(
        "/consent/api/grant", json={"patient_id": "P0001", "consent_type": "TREATMENT"}
    )
    responses = [
        grant,
        records_client.get("/consent/api/patient/P0001"),
        records_client.get(
            "/consent/api/check?patient_id=P0001&consent_type=TREATMENT"
        ),
        records_client.post(
            "/consent/api/revoke",
            json={"patient_id": "P0001", "consent_type": "TREATMENT"},
        ),
    ]
    for response in responses:
        body = response.get_data(as_text=True).lower()
        for marker in ("stub", "phase 1", "implementation pending", "full query logic"):
            assert (
                marker not in body
            ), f"{response.request.path} still returns a placeholder"


def test_list_returns_actual_records(records_client, patient):
    grant_patient_consent("P0001", "TREATMENT")
    grant_patient_consent("P0001", "ai_diagnosis")

    data = records_client.get("/consent/api/patient/P0001").get_json()
    assert data["count"] == 2
    assert {c["consent_type"] for c in data["consents"]} == {
        "TREATMENT",
        "ai_diagnosis",
    }


def test_revoke_persists_and_flips_the_gate(records_client, patient):
    grant_patient_consent("P0001", "ai_diagnosis")
    assert has_ai_consent("P0001") is True

    response = records_client.post(
        "/consent/api/revoke",
        json={
            "patient_id": "P0001",
            "consent_type": "ai_diagnosis",
            "reason": "withdrawn",
        },
    )
    assert response.status_code == 200
    assert response.get_json()["consent"]["status"] == "REVOKED"
    assert has_ai_consent("P0001") is False


def test_revoke_is_non_destructive(records_client, patient):
    """DPA 2019: the consent history must survive withdrawal."""
    grant_patient_consent("P0001", "TREATMENT")
    records_client.post(
        "/consent/api/revoke", json={"patient_id": "P0001", "consent_type": "TREATMENT"}
    )

    records = list_patient_consents("P0001")
    assert len(records) == 1
    assert records[0].revoked_at is not None


def test_revoking_what_was_never_granted_is_404(records_client, patient):
    response = records_client.post(
        "/consent/api/revoke", json={"patient_id": "P0001", "consent_type": "TREATMENT"}
    )
    assert response.status_code == 404


# ── Identifier bridging ───────────────────────────────────────────────────


def test_endpoints_accept_either_patient_identifier(records_client, patient):
    """Callers arrive with Patient.id or Patient.patient_id; both must resolve."""
    records_client.post(
        "/consent/api/grant", json={"patient_id": "P0001", "consent_type": "TREATMENT"}
    )

    by_business_id = records_client.get(
        "/consent/api/check?patient_id=P0001&consent_type=TREATMENT"
    ).get_json()
    by_numeric_id = records_client.get(
        f"/consent/api/check?patient_id={patient.id}&consent_type=TREATMENT"
    ).get_json()

    assert by_business_id["status"] == "ACTIVE"
    assert by_numeric_id["status"] == "ACTIVE"


def test_unknown_patient_is_404_not_a_silent_pass(records_client):
    for path in [
        "/consent/api/patient/P9999",
        "/consent/api/check?patient_id=P9999&consent_type=TREATMENT",
    ]:
        assert records_client.get(path).status_code == 404


def test_check_distinguishes_never_granted_from_revoked(records_client, patient):
    url = "/consent/api/check?patient_id=P0001&consent_type=TREATMENT"
    assert records_client.get(url).get_json()["status"] == "NOT_GRANTED"

    grant_patient_consent("P0001", "TREATMENT")
    assert records_client.get(url).get_json()["status"] == "ACTIVE"

    revoke_patient_consent("P0001", "TREATMENT")
    assert records_client.get(url).get_json()["status"] == "REVOKED"


def test_missing_parameters_are_rejected(records_client, patient):
    assert records_client.get("/consent/api/check?patient_id=P0001").status_code == 400
    assert (
        records_client.post(
            "/consent/api/grant", json={"patient_id": "P0001"}
        ).status_code
        == 400
    )


# ── Authorization ─────────────────────────────────────────────────────────


def test_consent_capture_requires_an_authorized_role(lab_client, patient):
    response = lab_client.post(
        "/consent/api/grant", json={"patient_id": "P0001", "consent_type": "TREATMENT"}
    )
    assert response.status_code in (302, 403)
    assert has_consent("P0001", "TREATMENT") is False


def test_consent_endpoints_require_login(client, patient):
    response = client.get("/consent/api/patient/P0001", follow_redirects=False)
    assert response.status_code in (302, 401)


# ── The duplicate model must stay gone ────────────────────────────────────


def test_duplicate_consent_model_is_removed():
    """Regression: two consent tables meant grants were invisible to the gates."""
    with pytest.raises(ImportError):
        from departments.consent.models import Consent  # noqa: F401


def test_only_one_consent_table_is_mapped():
    tables = set(db.metadata.tables)
    assert "patient_consents" in tables
    assert "consents" not in tables


# ── The prescribing gate now actually runs ────────────────────────────────


@pytest.fixture
def admitted_patient(app, patient):
    doctor = User(
        username="consent_doc",
        role="medicine",
        password=generate_password_hash("Doc!2345"),
    )
    ward = Ward(
        name="Test Ward", sex="M", number_of_beds=10, occupied_beds=0, daily_charge=1000
    )
    db.session.add_all([doctor, ward])
    db.session.commit()
    db.session.add(
        AdmittedPatient(
            patient_id="P0001",
            ward_id=ward.id,
            admission_criteria="test",
            admitted_by=doctor.id,
            admitted_on=datetime.utcnow(),
        )
    )
    db.session.commit()
    return doctor


def test_cds_engine_is_actually_invoked(client, app, admitted_patient, monkeypatch):
    """
    Regression: prescribe_drugs did int(patient_id) on a business ID like
    "P0001", which raised, left patient_int_id None, and skipped the clinical
    safety check entirely — silently, for every patient.
    """
    import departments.medicine.prescriptions as prescriptions

    calls = []
    original = prescriptions.ClinicalSafetyEngine.check_prescription

    def spy(self, patient_id, drug_ids, **kwargs):
        calls.append(patient_id)
        return original(self, patient_id=patient_id, drug_ids=drug_ids, **kwargs)

    monkeypatch.setattr(prescriptions.ClinicalSafetyEngine, "check_prescription", spy)

    client.post(
        "/login",
        data={"username": "consent_doc", "password": "Doc!2345"},
        follow_redirects=True,
    )
    response = client.get("/medicine/prescribe_drugs/P0001")

    assert response.status_code == 200
    assert calls, "clinical safety engine was never called"
    assert calls[0] == Patient.query.filter_by(patient_id="P0001").first().id


def test_prescribing_consent_gate_sees_granted_consent(
    client, app, admitted_patient, monkeypatch
):
    """A TREATMENT grant must reach the prescribing gate — it never could before."""
    seen = {}
    import departments.medicine.prescriptions as prescriptions

    def capture(template, **context):
        seen.update(context)
        return ""

    monkeypatch.setattr(prescriptions, "render_template", capture)

    client.post(
        "/login",
        data={"username": "consent_doc", "password": "Doc!2345"},
        follow_redirects=True,
    )

    client.get("/medicine/prescribe_drugs/P0001")
    assert seen.get("consent_status") == "MISSING"

    grant_patient_consent("P0001", "TREATMENT")
    client.get("/medicine/prescribe_drugs/P0001")
    assert seen.get("consent_status") == "ACTIVE"


# ── Laboratory landing page (found while wiring the RBAC fixture above) ────


def test_laboratory_index_renders_for_lab_role(client, app):
    """
    Regression: the laboratory index linked url_for('laboratory.pending_lab_patients'),
    but that endpoint lives on the medicine blueprint. Every lab tech hitting
    their own department's landing page got a BuildError 500.
    """
    db.session.add(
        User(
            username="lab_index_probe",
            role="laboratory",
            password=generate_password_hash("Lab!2345"),
        )
    )
    db.session.commit()
    client.post(
        "/login",
        data={"username": "lab_index_probe", "password": "Lab!2345"},
        follow_redirects=True,
    )
    assert client.get("/laboratory/").status_code == 200


def test_laboratory_templates_reference_no_missing_endpoints(app):
    """
    Guard the class of bug, scoped to laboratory — the department fixed here.

    The same sweep across the whole template tree currently reports 17 further
    broken targets (hr 11, imaging 3, medicine 2, pharmacy 1). Those are
    untouched features rather than typos, so widening this assertion is
    deliberately left until they are triaged; widen the glob below to re-run it
    platform-wide.
    """
    import re
    from pathlib import Path

    endpoints = {rule.endpoint for rule in app.url_map.iter_rules()}
    pattern = re.compile(r"url_for\(\s*['\"]([a-z_]+\.[a-z_]+)['\"]", re.IGNORECASE)

    broken = []
    for template in Path("departments/laboratory").rglob("*.html"):
        for endpoint in pattern.findall(template.read_text(errors="replace")):
            if endpoint not in endpoints:
                broken.append(f"{template}: {endpoint}")

    assert not broken, "templates reference endpoints that do not exist:\n" + "\n".join(
        broken
    )
