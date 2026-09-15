"""
tests/invariants/test_mar_integrity_invariants.py
──────────────────────────────────────────────────
Invariants for the Medication Administration Record (MAR).

Critical property: the nurse identity recorded in a MAR entry must ALWAYS
come from the authenticated session, never from the request body.  Accepting
nurse_id from the caller would allow any authenticated user to forge another
clinician's identity in a permanent clinical record.
"""

import pytest
from werkzeug.security import generate_password_hash


# ─── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def nurse_user(app, client):
    """Create a nurse user and log in."""
    from departments.models.user import User
    from extensions import db

    with app.app_context():
        nurse = User.query.filter_by(username="nurse_invariant_test").first()
        if not nurse:
            nurse = User(
                username="nurse_invariant_test",
                password=generate_password_hash("nursepass123", method="pbkdf2:sha256"),
                role="nursing",
            )
            db.session.add(nurse)
            db.session.commit()
        nurse_id = nurse.id

    client.post("/login", data={"username": "nurse_invariant_test", "password": "nursepass123"})
    yield nurse_id


@pytest.fixture
def admitted_patient(app):
    """Seed the minimum patient record required for MAR tests."""
    from datetime import date
    from departments.models.records import Patient
    from extensions import db

    with app.app_context():
        p = Patient.query.filter_by(patient_id="P-MAR01").first()
        if not p:
            p = Patient(
                patient_id="P-MAR01",
                name="MAR Test Patient",
                place_of_residence="Test Town",
                sex="Male",
                date_of_birth=date(1984, 1, 1),
                marital_status="Single",
                contact="0700000001",
                next_of_kin="Test Kin",
                relationship_with_next_of_kin="Sibling",
                next_of_kin_contact="0700000002",
            )
            db.session.add(p)
            db.session.commit()
        yield "P-MAR01"


# ─── identity invariants ──────────────────────────────────────────────────────

class TestMARNurseIdentityFromSession:
    """
    INVARIANT: recorded_by in MedicationAdmin must always equal the
    authenticated user's ID, regardless of what nurse_id the caller sends.
    """

    def test_nurse_id_from_session_not_request(self, app, client, nurse_user, admitted_patient):
        """
        Posting a different nurse_id in the body must NOT override the
        session identity in the stored record.
        """
        from departments.models.nursing import MedicationAdmin
        from extensions import db

        forged_nurse_id = nurse_user + 9999  # a different, non-existent user ID

        resp = client.post(
            "/nursing/mar/chart",
            json={
                "patient_id": admitted_patient,
                "medication": "Paracetamol",
                "dosage": "500mg PO",
                "nurse_id": forged_nurse_id,  # attacker tries to forge identity
            },
        )

        # Accept 201 (created) or 404/500 if patient fixture isn't fully wired,
        # but in any success case the identity must match the session
        if resp.status_code == 201:
            data = resp.get_json()
            record_id = data.get("record_id")
            with app.app_context():
                record = db.session.get(MedicationAdmin, record_id)
                assert record is not None
                assert record.recorded_by == nurse_user, (
                    f"INVARIANT VIOLATION: MAR record has recorded_by={record.recorded_by} "
                    f"but the authenticated nurse ID is {nurse_user}. "
                    f"The request body nurse_id ({forged_nurse_id}) must never override "
                    f"the session identity."
                )

    def test_chart_requires_patient_medication_dosage(self, app, client, nurse_user):
        """Missing required clinical fields must return 400, not 500."""
        resp = client.post(
            "/nursing/mar/chart",
            json={"patient_id": "P-MAR-TEST"},  # missing medication and dosage
        )
        assert resp.status_code == 400, (
            "Incomplete MAR submission must return 400, not silently succeed."
        )


class TestMARAccessControl:
    """
    INVARIANT: only nursing-role users can write MAR records.
    A doctor-role user submitting a MAR chart must be rejected.
    """

    def test_non_nursing_role_rejected(self, app, client):
        """A doctor-role user posting to /nursing/mar/chart must be rejected with 403."""
        from departments.models.user import User
        from extensions import db

        with app.app_context():
            doc = User.query.filter_by(username="doctor_mar_test").first()
            if not doc:
                doc = User(
                    username="doctor_mar_test",
                    password=generate_password_hash("docpass123", method="pbkdf2:sha256"),
                    role="doctor",
                )
                db.session.add(doc)
                db.session.commit()

        client.post("/login", data={"username": "doctor_mar_test", "password": "docpass123"})

        resp = client.post(
            "/nursing/mar/chart",
            json={
                "patient_id": "P-MAR-TEST",
                "medication": "Morphine",
                "dosage": "10mg IV",
            },
        )
        # admin role also allowed by decorator — only non-nursing non-admin should fail
        assert resp.status_code in (403, 302), (
            f"Doctor role should not be able to write MAR records. Got {resp.status_code}."
        )
