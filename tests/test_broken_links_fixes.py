"""
tests/test_broken_links_fixes.py
───────────────────────────────────
Regression tests for the platform-wide broken-template-link sweep: every
url_for() target across every .html file must resolve to a real endpoint.
Six were broken; this file locks in the fixes so they can't quietly regress,
and a full-tree guard test makes sure no others slip back in.

  - imaging/index.html linked imaging.dicom_studies / imaging.dicom_upload
    (typos for imaging.dicom_studies_search_ui / imaging.dicom_upload_ui,
    which already existed, fully built, on a differently-named blueprint
    variable in the same file) and imaging.unmatched_imaging_requests
    (no backing feature existed anywhere; replaced with a working link).
  - medicine's add_ward_round() redirected to medicine.view_ward_rounds on
    every branch (success, validation failure, and error) — an endpoint
    that was never defined, so every ward-round note submission saved to
    the database and then 500'd.
  - pharmacy/prescriptions.html's Dispense button was permanently disabled
    via a `{{ x if false else y }}` Jinja trick, silently substituting a
    no-op link for the real dispense_prescription page.
"""
from datetime import date, datetime

import pytest
from werkzeug.security import generate_password_hash

from departments.models.medicine import AdmittedPatient, Ward, WardRound
from departments.models.records import Patient
from departments.models.user import User
from extensions import db


def test_no_template_in_the_whole_app_references_a_missing_endpoint(app):
    """Platform-wide guard: the sweep that found all six of today's bugs."""
    import re
    from pathlib import Path

    endpoints = {rule.endpoint for rule in app.url_map.iter_rules()}
    pattern = re.compile(r"url_for\(\s*['\"]([a-zA-Z_]+\.[a-zA-Z_]+)['\"]")

    broken = []
    for template in Path(".").rglob("*.html"):
        if "node_modules" in str(template):
            continue
        for endpoint in pattern.findall(template.read_text(errors="replace")):
            if endpoint not in endpoints:
                broken.append(f"{template}: {endpoint}")

    assert not broken, "templates reference endpoints that do not exist:\n" + "\n".join(broken)


# ── imaging ────────────────────────────────────────────────────────────────

@pytest.fixture
def imaging_client(client, app):
    db.session.add(User(username="imaging_tech", role="imaging",
                        password=generate_password_hash("Imaging!234")))
    db.session.commit()
    client.post("/login", data={"username": "imaging_tech", "password": "Imaging!234"},
                follow_redirects=True)
    return client


def test_imaging_index_links_to_real_dicom_endpoints(imaging_client):
    response = imaging_client.get("/imaging/")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "/imaging/dicom/ui/studies" in body
    assert "/imaging/dicom/ui/upload" in body
    assert "unmatched_imaging_requests" not in body


def test_imaging_dicom_links_are_actually_reachable(imaging_client):
    """Not enough for the link to resolve — the destination must also load."""
    assert imaging_client.get("/imaging/dicom/ui/studies").status_code == 200
    assert imaging_client.get("/imaging/dicom/ui/upload").status_code == 200


def test_imaging_radiology_actions_card_points_somewhere_real(imaging_client):
    response = imaging_client.get("/imaging/")
    assert b"Radiology Reports" in response.data
    follow = imaging_client.get("/imaging/reports", follow_redirects=True)
    assert follow.status_code == 200


# ── medicine: ward rounds ───────────────────────────────────────────────────

@pytest.fixture
def admitted(app):
    doctor = User(username="ward_doc", role="medicine",
                 password=generate_password_hash("Doc!23456"))
    ward = Ward(name="Ward Fix", sex="M", number_of_beds=10, occupied_beds=0, daily_charge=1000)
    patient = Patient(patient_id="P-WARDFIX", name="Ward Fix Patient", sex="M",
                      date_of_birth=date(1980, 1, 1))
    db.session.add_all([doctor, ward, patient])
    db.session.commit()
    admission = AdmittedPatient(patient_id="P-WARDFIX", ward_id=ward.id,
                                admission_criteria="test", admitted_by=doctor.id,
                                admitted_on=datetime.utcnow())
    db.session.add(admission)
    db.session.commit()
    return doctor, admission


def test_ward_round_submission_no_longer_crashes(client, app, admitted):
    """Regression: this always saved the note, then 500'd on the redirect."""
    doctor, admission = admitted
    client.post("/login", data={"username": "ward_doc", "password": "Doc!23456"},
                follow_redirects=True)
    response = client.post("/medicine/ward-rounds/add", data={
        "admission_id": admission.id, "notes": "Patient stable", "status": "Stable",
    }, follow_redirects=True)
    assert response.status_code == 200
    assert WardRound.query.filter_by(admission_id=admission.id).count() == 1


def test_view_ward_rounds_shows_submitted_notes(client, app, admitted):
    doctor, admission = admitted
    client.post("/login", data={"username": "ward_doc", "password": "Doc!23456"},
                follow_redirects=True)
    client.post("/medicine/ward-rounds/add", data={
        "admission_id": admission.id, "notes": "Improving steadily", "status": "Stable",
    })
    response = client.get(f"/medicine/ward-rounds/{admission.id}")
    assert response.status_code == 200
    assert b"Improving steadily" in response.data


def test_view_ward_rounds_handles_unknown_admission(client, app):
    db.session.add(User(username="ward_doc2", role="medicine",
                        password=generate_password_hash("Doc!23456")))
    db.session.commit()
    client.post("/login", data={"username": "ward_doc2", "password": "Doc!23456"},
                follow_redirects=True)
    response = client.get("/medicine/ward-rounds/999999", follow_redirects=True)
    assert response.status_code == 200


def test_ward_round_validation_failure_also_redirects_cleanly(client, app, admitted):
    """Even the error branch used to hit the same undefined-endpoint crash."""
    doctor, admission = admitted
    client.post("/login", data={"username": "ward_doc", "password": "Doc!23456"},
                follow_redirects=True)
    response = client.post("/medicine/ward-rounds/add", data={
        "admission_id": admission.id, "notes": "", "status": "",
    }, follow_redirects=True)
    assert response.status_code == 200


# ── pharmacy ─────────────────────────────────────────────────────────────

def test_dispense_button_is_no_longer_disabled(client, app):
    """
    Regression: a `{{ x if false else y }}` hack silently no-op'd this button.
    Also caught along the way: the queue itself never showed anything,
    because the template looped over 'prescriptions' while the route always
    passes 'pending_prescriptions' — a different name, silently empty.
    """
    from departments.models.encounter import Encounter
    from departments.models.records import Patient

    db.session.add(User(username="pharm_tech", role="pharmacy",
                        password=generate_password_hash("Pharm!234")))
    db.session.add(Patient(patient_id="P-DISPENSE", name="Dispense Patient", sex="F",
                           date_of_birth=date(1990, 1, 1)))
    db.session.commit()
    db.session.add(Encounter(patient_id="P-DISPENSE", encounter_type="OPD",
                             status="ACTIVE", stage="AWAITING_PHARMACY"))
    db.session.commit()

    client.post("/login", data={"username": "pharm_tech", "password": "Pharm!234"},
                follow_redirects=True)
    response = client.get("/pharmacy/prescriptions")
    body = response.get_data(as_text=True)
    assert "Dispense Patient" in body
    assert "if false else" not in body
    assert "/pharmacy/dispense/" in body


def test_pharmacy_queue_is_not_permanently_empty(client, app):
    """
    Regression: the loop variable mismatch meant this page showed 'No active
    prescriptions' regardless of how many patients were actually queued.
    """
    from departments.models.encounter import Encounter
    from departments.models.records import Patient

    db.session.add(User(username="pharm_tech2", role="pharmacy",
                        password=generate_password_hash("Pharm!234")))
    db.session.add(Patient(patient_id="P-Q2", name="Second Queue Patient", sex="M",
                           date_of_birth=date(1985, 1, 1)))
    db.session.commit()
    db.session.add(Encounter(patient_id="P-Q2", encounter_type="OPD",
                             status="ACTIVE", stage="AWAITING_PHARMACY"))
    db.session.commit()

    client.post("/login", data={"username": "pharm_tech2", "password": "Pharm!234"},
                follow_redirects=True)
    response = client.get("/pharmacy/prescriptions")
    body = response.get_data(as_text=True)
    assert "No active prescriptions currently waiting" not in body
    assert "1 Order" in body
