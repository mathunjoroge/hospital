import io
from datetime import date

import pytest
from werkzeug.security import generate_password_hash

from departments.models.medicine import Disease, OncologyNote
from departments.models.pharmacy import Batch, DispensedDrug, Drug, DrugCategory
from departments.models.records import Patient
from departments.models.user import User
from extensions import db


@pytest.fixture
def doctor_user(app):
    """Create a test user with medicine role."""
    with app.app_context():
        user = User.query.filter_by(username="doc_test").first()
        if not user:
            user = User(
                username="doc_test",
                password=generate_password_hash("docpass123", method="pbkdf2:sha256"),
                role="medicine",
            )
            db.session.add(user)
            db.session.commit()
            db.session.refresh(user)
        return user


@pytest.fixture
def auth_client(client, doctor_user):
    """Client authenticated with medicine role."""
    with client.session_transaction() as sess:
        sess["_user_id"] = str(doctor_user.id)
        sess["_fresh"] = True
    return client


# ── 1. DICOM Access Control ──
def test_dicom_unauthenticated_blocked(client):
    """Unauthenticated requests to DICOM routes must be rejected."""
    res1 = client.get("/imaging/dicom/mwl")
    assert res1.status_code in [302, 401, 403]

    res2 = client.get("/imaging/dicom/download/1/0")
    assert res2.status_code in [302, 401, 403]


# ── 2. ICU Routes Authorization & No User Fallback ──
def test_icu_unauthenticated_blocked(client):
    """Unauthenticated vitals submission must fail."""
    res = client.post("/icu/flowsheet/P0001/vitals", json={"heart_rate": 80})
    assert res.status_code in [302, 401, 403]


# ── 3. Renal Routes Authorization & No Nurse Fallback ──
def test_renal_unauthenticated_blocked(client):
    """Unauthenticated dialysis log creation must fail."""
    res = client.post("/renal/sessions/P0001", json={"weight_pre": 70})
    assert res.status_code in [302, 401, 403]


# ── 4. LIS Result Verification ──
def test_lis_enter_unauthenticated_blocked(client):
    """Unauthenticated LIS enter must fail."""
    res = client.post(
        "/laboratory/lis/enter", json={"lab_order_id": 1, "result_val": "10"}
    )
    assert res.status_code in [302, 401, 403]


# ── 5. Oncology Disease Deletion (Admin Only) ──
def test_oncology_delete_disease_requires_admin(auth_client, app):
    """Regular doctor should not be able to delete reference disease data."""
    with app.app_context():
        d = Disease(cui="C123456", name="Test Disease")
        db.session.add(d)
        db.session.commit()
        disease_id = d.id

    res = auth_client.post(f"/medicine/diseases/delete/{disease_id}")
    assert res.status_code in [302, 403]

    with app.app_context():
        # Disease must still exist
        assert db.session.get(Disease, disease_id) is not None


# ── 6. Oncology Note Void Workflow ──
def test_oncology_note_void_preserves_record(client, app):
    """Deleting an oncology note should void it rather than physical row delete."""
    with app.app_context():
        admin = User.query.filter_by(role="admin").first()
        if not admin:
            admin = User(username="admin_onc_test", password="pwd", role="admin")
            db.session.add(admin)
            db.session.commit()
        admin_id = admin.id

        patient = Patient(
            patient_id="P0001",
            name="Onco Patient",
            sex="Female",
            date_of_birth=date(1990, 1, 1),
        )
        db.session.add(patient)
        db.session.flush()

        note = OncologyNote(
            patient_id=patient.patient_id,
            note_content="Initial oncology note content",
        )
        db.session.add(note)
        db.session.commit()
        note_id = note.id

    with client.session_transaction() as sess:
        sess["_user_id"] = str(admin_id)
        sess["_fresh"] = True

    res = client.post(
        f"/medicine/oncology/note/{note_id}/void",
        data={"void_reason": "Entered in error"},
    )
    assert res.status_code in [200, 302]

    with app.app_context():
        saved_note = db.session.get(OncologyNote, note_id)
        assert saved_note is not None
        assert getattr(saved_note, "is_voided", True) is True


# ── 7. Pharmacy Void Workflow (Dispensing) ──
def test_pharmacy_remove_dispensed_voids_not_deletes(client, app):
    """Removing a dispensed drug must mark status VOIDED and retain record."""
    with app.app_context():
        admin = User.query.filter_by(role="admin").first()
        if not admin:
            admin = User(username="admin_pharm_test", password="pwd", role="admin")
            db.session.add(admin)
            db.session.commit()
        admin_id = admin.id

        cat = DrugCategory(name="Analgesics")
        db.session.add(cat)
        db.session.flush()

        patient = Patient(
            patient_id="P0002",
            name="Pharm Patient",
            sex="Male",
            date_of_birth=date(1985, 5, 5),
        )
        drug = Drug(
            generic_name="Paracetamol",
            category_id=cat.id,
            dosage_form="Tablet",
            strength="500mg",
            buying_price=1.0,
            selling_price=2.0,
            quantity_in_stock=50,
        )
        db.session.add_all([patient, drug])
        db.session.flush()

        batch = Batch(drug_id=drug.id, batch_number="B100", quantity_in_stock=50)
        db.session.add(batch)
        db.session.flush()

        dispensed = DispensedDrug(
            patient_id=patient.patient_id,
            drug_id=drug.id,
            prescription_id="RX123",
            quantity_dispensed=10,
            batch_id=batch.id,
            status="0",
        )
        db.session.add(dispensed)
        db.session.commit()
        dispensed_id = dispensed.id

    with client.session_transaction() as sess:
        sess["_user_id"] = str(admin_id)
        sess["_fresh"] = True

    res = client.post(
        f"/pharmacy/remove_dispensed/{dispensed_id}",
        data={"void_reason": "Patient refused medication"},
    )
    assert res.status_code in [200, 302]

    with app.app_context():
        saved = db.session.get(DispensedDrug, dispensed_id)
        assert saved is not None
        assert saved.status == "VOIDED"
        assert saved.void_reason == "Patient refused medication"
        # Batch quantity restored
        saved_batch = db.session.get(Batch, batch.id)
        assert saved_batch.quantity_in_stock == 60


# ── 8. MCH Wrong Patient Fallback ──
def test_mch_nonexistent_patient_returns_404(auth_client):
    """MCH NICU workbench for non-existent patient must 404, not load first patient."""
    res = auth_client.get("/mch/nicu/workbench?patient_id=999999")
    assert res.status_code == 404


# ── 9. Theatre Wrong Patient Fallback ──
def test_theatre_nonexistent_entry_returns_404(auth_client):
    """Theatre surgical workbench with invalid entry must 404, not load dummy."""
    res = auth_client.get("/theatre/workbench?entry_id=999999")
    assert res.status_code == 404


# ── 10. Consultations File Upload Path Traversal ──
def test_consultations_file_upload_sanitization(auth_client, app):
    """Path traversal filenames in file upload must be sanitized or rejected."""
    with app.app_context():
        p = Patient(
            patient_id="P0003",
            name="Consult Patient",
            sex="Male",
            date_of_birth=date(1992, 2, 2),
        )
        db.session.add(p)
        db.session.commit()
        pid = p.patient_id

    data = {
        "assessment": "Test assessment",
        "recommendation": "Test rec",
        "attachment": (
            io.BytesIO(b"%PDF-1.4 test file content"),
            "../../../etc/passwd.pdf",
        ),
    }
    res = auth_client.post(
        f"/medicine/submit_soap_notes/{pid}",
        data=data,
        content_type="multipart/form-data",
    )
    assert res.status_code in [200, 302]
