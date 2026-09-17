from datetime import date
from werkzeug.security import generate_password_hash

from departments.models.encounter import Encounter
from departments.models.medicine import Imaging, LabTest, RequestedImage, RequestedLab
from departments.models.records import Patient, PatientWaitingList
from departments.models.user import User
from departments.shared.encounter_utils import active_encounter
from departments.shared.queue_constants import QueueStatus
from extensions import db


def _setup_patient_with_active_encounter(pid="P_TEST_SCOPING"):
    p = Patient(patient_id=pid, name=f"Test {pid}", sex="M", date_of_birth=date(1995, 5, 15))
    db.session.add(p)
    db.session.add(PatientWaitingList(patient_id=pid, seen=QueueStatus.WAITING_TRIAGE))
    enc = Encounter(patient_id=pid, encounter_type="OPD", status="ACTIVE", stage="IN_CONSULTATION")
    db.session.add(enc)
    db.session.commit()
    return p, enc


def test_active_encounter_excludes_cancelled_encounters(app):
    with app.app_context():
        p, enc = _setup_patient_with_active_encounter("P_CANCELLED_TEST")
        assert active_encounter("P_CANCELLED_TEST").id == enc.id

        # Cancel the encounter
        enc.status = "CANCELLED"
        enc.stage = "CANCELLED"
        db.session.commit()

        assert active_encounter("P_CANCELLED_TEST") is None


def test_request_lab_tests_populates_encounter_id(app, client):
    with app.app_context():
        # Setup admin user
        user = User.query.filter_by(username="admin_scoping").first()
        if not user:
            user = User(id=9988, username="admin_scoping", role="admin", password=generate_password_hash("pass123"))
            db.session.add(user)
            db.session.commit()

        lab = LabTest(test_name="Complete Blood Count", cost=600)
        db.session.add(lab)
        db.session.commit()
        lab_id = lab.id

        p, enc = _setup_patient_with_active_encounter("P_LAB_SCOPING")
        enc_id = enc.id

    # Authenticate client
    client.post("/login", data={"username": "admin_scoping", "password": "pass123"}, follow_redirects=True)

    resp = client.post(
        "/medicine/request_lab_tests/P_LAB_SCOPING",
        data={"lab_tests[]": [str(lab_id)], f"descriptions[{lab_id}]": "Routine check"},
        follow_redirects=True,
    )
    assert resp.status_code == 200

    with app.app_context():
        req_lab = RequestedLab.query.filter_by(patient_id="P_LAB_SCOPING").first()
        assert req_lab is not None
        assert req_lab.encounter_id == enc_id


def test_request_imaging_populates_encounter_id(app, client):
    with app.app_context():
        user = User.query.filter_by(username="admin_scoping").first()
        if not user:
            user = User(id=9988, username="admin_scoping", role="admin", password=generate_password_hash("pass123"))
            db.session.add(user)
            db.session.commit()

        img = Imaging(imaging_type="Chest X-Ray", cost=1500)
        db.session.add(img)
        db.session.commit()
        img_id = img.id

        p, enc = _setup_patient_with_active_encounter("P_IMG_SCOPING")
        enc_id = enc.id

    client.post("/login", data={"username": "admin_scoping", "password": "pass123"}, follow_redirects=True)

    resp = client.post(
        "/medicine/request_imaging/P_IMG_SCOPING",
        data={"imaging_types[]": [str(img_id)], f"descriptions[{img_id}]": "Cough assessment"},
        follow_redirects=True,
    )
    assert resp.status_code == 200

    with app.app_context():
        req_img = RequestedImage.query.filter_by(patient_id="P_IMG_SCOPING").first()
        assert req_img is not None
        assert req_img.encounter_id == enc_id
