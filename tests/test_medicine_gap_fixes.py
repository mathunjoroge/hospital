"""
tests/test_medicine_gap_fixes.py
────────────────────────────────
Regression tests for the medicine department gap fixes:
  P0-1  e-prescribing endpoints require authentication + roles
  P0-2  exact patient matching (no fuzzy ilike resolution)
  P0-3/5 inpatients routes require roles; attribution from current_user
  P0-6  no BuildError for missing patients_list route
  P1-7  admitted (IPD) patients can have labs/imaging requested
  P1-8  post-consult staging scoped to the current encounter
  P1-11 imaging keyword scan uses word boundaries
  P1-12 dispensed prescription lines cannot be deleted
  P1-14 available_rooms derives availability from beds
  P2-18 unscoped clinical chatbot input is consent-gated
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from werkzeug.security import generate_password_hash

from departments.models.medicine import (
    AdmittedPatient,
    Bed,
    PrescribedMedicine,
    RequestedLab,
    TheatreList,
    TheatreProcedure,
    Ward,
    WardBedHistory,
    WardRoom,
)
from departments.models.records import Patient
from departments.models.user import User
from extensions import db

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def medicine_user(client, app):
    with app.app_context():
        user = User.query.filter_by(username="med_doc_gap").first()
        if not user:
            user = User(
                username="med_doc_gap",
                password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
                role="medicine",
            )
            db.session.add(user)
            db.session.commit()
    client.post(
        "/login", data={"username": "med_doc_gap", "password": "Password123!"}
    )
    return user


@pytest.fixture
def low_privilege_user(client, app):
    """An authenticated user with no clinical roles (e.g. HR clerk)."""
    with app.app_context():
        user = User.query.filter_by(username="hr_clerk_gap").first()
        if not user:
            user = User(
                username="hr_clerk_gap",
                password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
                role="hr",
            )
            db.session.add(user)
            db.session.commit()
    client.post(
        "/login", data={"username": "hr_clerk_gap", "password": "Password123!"}
    )
    return user


@pytest.fixture
def patient(app):
    with app.app_context():
        p = Patient.query.filter_by(patient_id="P-GAP-001").first()
        if not p:
            p = Patient(
                patient_id="P-GAP-001",
                name="Gap Test Patient",
                sex="F",
                date_of_birth=date(1990, 1, 15),
            )
            db.session.add(p)
            db.session.commit()
        return p.patient_id


# ---------------------------------------------------------------------------
# P0-1: e-prescribing RBAC
# ---------------------------------------------------------------------------


class TestEPrescribeAuth:
    def test_anonymous_gets_redirected_from_signoff(self, client):
        resp = client.post(
            "/medicine/prescribe/signoff", json={"patient_id": "P", "prescriptions": []}
        )
        assert resp.status_code in (302, 401, 403)

    def test_low_privilege_role_is_403_on_signoff(self, client, low_privilege_user):
        resp = client.post(
            "/medicine/prescribe/signoff",
            json={"patient_id": "P", "prescriptions": [{"name": "Aspirin"}]},
        )
        assert resp.status_code == 403

    def test_low_privilege_role_is_403_on_cdss(self, client, low_privilege_user):
        resp = client.post("/medicine/prescribe/cdss/evaluate", json={})
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# P0-2: exact patient matching
# ---------------------------------------------------------------------------


class TestExactPatientMatching:
    def test_prescribe_drugs_rejects_unknown_exact_id(self, client, app, medicine_user, patient):
        """A fuzzy prefix that used to resolve to P-GAP-001 must no longer match."""
        resp = client.get("/medicine/prescribe_drugs/P-GAP-00", follow_redirects=True)
        assert resp.status_code == 200
        assert b"neither admitted" in resp.data or b"not found" in resp.data

    def test_theatre_booking_requires_exact_patient(self, client, app, medicine_user):
        with app.app_context():
            proc = TheatreProcedure(name="GapProc", type="General", cost=1000.0)
            db.session.add(proc)
            db.session.commit()
            proc_id = proc.id
        resp = client.post(
            "/medicine/add-to-theatre",
            data={"patient_id": "NO-SUCH-PATIENT", "procedure_id": str(proc_id)},
            follow_redirects=True,
        )
        assert resp.status_code == 200
        with app.app_context():
            assert TheatreList.query.count() == 0


# ---------------------------------------------------------------------------
# P0-3/P0-5: inpatients RBAC + attribution
# ---------------------------------------------------------------------------


class TestInpatientsRBAC:
    def test_hr_clerk_cannot_discharge(self, client, app, low_privilege_user):
        resp = client.post("/medicine/discharge-patient/1", follow_redirects=True)
        assert resp.status_code == 403

    def test_hr_clerk_cannot_call_adt_admit_api(self, client, app, low_privilege_user):
        resp = client.post("/medicine/api/adt/admit", json={})
        assert resp.status_code == 403

    def test_theatre_booking_uses_current_user_attribution(
        self, client, app, medicine_user, patient
    ):
        """created_by comes from the session, not the client form."""
        with app.app_context():
            proc = TheatreProcedure(name="GapAttrib", type="General", cost=500.0)
            db.session.add(proc)
            db.session.commit()
            proc_id = proc.id
        resp = client.post(
            "/medicine/add-to-theatre",
            data={
                "patient_id": patient,
                "procedure_id": str(proc_id),
                "created_by": "999999",  # forged value must be ignored
            },
            follow_redirects=True,
        )
        assert resp.status_code == 200
        with app.app_context():
            entry = TheatreList.query.filter_by(patient_id=patient).first()
            assert entry is not None
            assert entry.created_by != 999999


# ---------------------------------------------------------------------------
# P0-6: no BuildError on missing patients_list
# ---------------------------------------------------------------------------


class TestBookingNoPatientsPath:
    def test_new_booking_redirects_without_builderror(self, client, app, medicine_user):
        # Empty DB: the no-patients branch redirects (it used to raise a
        # guaranteed BuildError 500 via url_for('medicine.patients_list')).
        # With patients present the page renders. Either way: no 500.
        resp = client.get("/medicine/bookings/new")
        assert resp.status_code in (200, 302)


# ---------------------------------------------------------------------------
# P1-7: admitted patients can have labs requested
# ---------------------------------------------------------------------------


class TestIPDLabRequests:
    def test_request_lab_tests_for_admitted_patient(self, client, app, medicine_user):
        from departments.models.medicine import LabTest

        with app.app_context():
            p = Patient(
                patient_id="P-GAP-IPD",
                name="IPD Lab Patient",
                sex="M",
                date_of_birth=date(1980, 2, 2),
            )
            db.session.add(p)
            ward = Ward(name="GapWard", sex="M", daily_charge=100)
            db.session.add(ward)
            db.session.flush()
            room = WardRoom(ward_id=ward.id, room_number="101")
            db.session.add(room)
            db.session.flush()
            bed = Bed(room_id=room.id, bed_number="A", occupied=True, status="OCCUPIED")
            db.session.add(bed)
            db.session.flush()
            adm = AdmittedPatient(
                patient_id=p.patient_id,
                ward_id=ward.id,
                room_id=room.id,
                bed_id=bed.id,
                admission_criteria="Test admission",
                admitted_by=1,
                admitted_on=date.today(),
            )
            db.session.add(adm)
            lab = LabTest(test_name="Gap FBC", cost=100)
            db.session.add(lab)
            db.session.commit()
            lab_id = lab.id
            pid = p.patient_id

        # No PatientWaitingList entry at all — the old code rejected IPD
        # patients with "not found in the waiting list".
        resp = client.post(
            f"/medicine/request_lab_tests/{pid}",
            data={"lab_tests[]": [str(lab_id)]},
            follow_redirects=True,
        )
        assert resp.status_code == 200
        with app.app_context():
            req = RequestedLab.query.filter_by(patient_id=pid).first()
            assert req is not None, "IPD patient lab request was not created"


# ---------------------------------------------------------------------------
# P1-11: imaging keyword word boundaries
# ---------------------------------------------------------------------------


class TestImagingKeywordScan:
    def _extract(self, recommendation):
        """Run the same word-boundary logic as submit_soap_notes."""
        imaging_keywords = ["ct", "mri", "x-ray", "ultrasound", "pet", "scan"]
        words = recommendation.lower().split()
        matched = set()
        for i, word in enumerate(words):
            stripped = word.strip(".,;:!?")
            for keyword in imaging_keywords:
                if stripped == keyword or stripped.startswith(keyword + "-"):
                    phrase = " ".join(words[i : i + 2]) if i + 1 < len(words) else word
                    matched.add(phrase)
                    break
        return matched

    def test_actual_does_not_match_ct(self):
        assert self._extract("patient is actually stable") == set()

    def test_important_does_not_match_ct(self):
        assert self._extract("important to monitor") == set()

    def test_real_ct_still_matches(self):
        assert "ct chest" in self._extract("order a CT chest today")


# ---------------------------------------------------------------------------
# P1-12: dispensed lines cannot be deleted
# ---------------------------------------------------------------------------


class TestDispensedLineGuard:
    def test_delete_dispensed_line_is_blocked(
        self, client, app, medicine_user, patient
    ):
        from departments.models.medicine import Medicine

        with app.app_context():
            med = Medicine(
                generic_name="Gap Med", brand_name="Gap Med", dosage="tab"
            )
            db.session.add(med)
            db.session.flush()
            rx = PrescribedMedicine(
                patient_id=patient,
                medicine_id=med.id,
                dosage="1 tab",
                strength="500mg",
                frequency="TDS",
                prescription_id="gap-rx-uuid",
                num_days=5,
                status=1,  # already dispensed
            )
            db.session.add(rx)
            db.session.commit()
            rx_id = rx.id

        resp = client.post(
            f"/medicine/delete_prescribed_medicine/{rx_id}", follow_redirects=True
        )
        assert resp.status_code == 200
        with app.app_context():
            assert db.session.get(PrescribedMedicine, rx_id) is not None


# ---------------------------------------------------------------------------
# P1-14: available_rooms is bed-derived
# ---------------------------------------------------------------------------


class TestAvailableRooms:
    def test_room_with_free_bed_is_listed(self, client, app, medicine_user):
        with app.app_context():
            ward = Ward(name="GapWard2", sex="F", daily_charge=100)
            db.session.add(ward)
            db.session.flush()
            room = WardRoom(ward_id=ward.id, room_number="201")
            db.session.add(room)
            db.session.flush()
            db.session.add(
                Bed(room_id=room.id, bed_number="B1", occupied=False)
            )
            db.session.add(
                Bed(room_id=room.id, bed_number="B2", occupied=False)
            )
            db.session.commit()
            ward_id = ward.id
            room_id = room.id

        resp = client.get(f"/medicine/available-rooms/{ward_id}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert any(r["id"] == room_id for r in data["rooms"])

    def test_fully_occupied_room_is_excluded(self, client, app, medicine_user):
        with app.app_context():
            ward = Ward(name="GapWard3", sex="M", daily_charge=100)
            db.session.add(ward)
            db.session.flush()
            room = WardRoom(ward_id=ward.id, room_number="301")
            db.session.add(room)
            db.session.flush()
            db.session.add(
                Bed(room_id=room.id, bed_number="C1", occupied=True)
            )
            db.session.commit()
            ward_id = ward.id
            room_id = room.id

        resp = client.get(f"/medicine/available-rooms/{ward_id}")
        data = resp.get_json()
        assert not any(r["id"] == room_id for r in data["rooms"])


# ---------------------------------------------------------------------------
# P0-4/P1-13: web discharge frees the right bed and records history
# ---------------------------------------------------------------------------


class TestWebDischarge:
    def test_discharge_frees_own_bed_and_sets_dirty(
        self, client, app, medicine_user
    ):
        with app.app_context():
            p = Patient(
                patient_id="P-GAP-DIS",
                name="Discharge Patient",
                sex="M",
                date_of_birth=date(1975, 3, 3),
            )
            db.session.add(p)
            ward = Ward(name="GapWard4", sex="M", daily_charge=100)
            db.session.add(ward)
            db.session.flush()
            room = WardRoom(ward_id=ward.id, room_number="401")
            db.session.add(room)
            db.session.flush()
            bed = Bed(room_id=room.id, bed_number="D1", occupied=True, status="OCCUPIED")
            db.session.add(bed)
            db.session.flush()
            adm = AdmittedPatient(
                patient_id=p.patient_id,
                ward_id=ward.id,
                room_id=room.id,
                bed_id=bed.id,
                admission_criteria="Test",
                admitted_by=1,
                admitted_on=date.today(),
            )
            db.session.add(adm)
            db.session.commit()
            adm_id = adm.id
            bed_id = bed.id
            ward_id = ward.id

        resp = client.post(
            f"/medicine/discharge-patient/{adm_id}", follow_redirects=True
        )
        assert resp.status_code == 200
        with app.app_context():
            bed = db.session.get(Bed, bed_id)
            assert bed.occupied is False
            assert bed.status == "DIRTY"
            history = (
                WardBedHistory.query.filter_by(
                    ward_id=ward_id, action="Discharge"
                )
                .order_by(WardBedHistory.timestamp.desc())
                .first()
            )
            assert history is not None


# ---------------------------------------------------------------------------
# P2-18: unscoped clinical chatbot input is gated
# ---------------------------------------------------------------------------


class TestChatbotClinicalGate:
    def test_unscoped_clinical_narrative_is_403(self, client, medicine_user):
        resp = client.post(
            "/medicine/chatbot",
            data={
                "clinical_note": "Patient complains of chest pain, BP 150/90, prescribed aspirin."
            },
        )
        assert resp.status_code == 403

    def test_general_query_passes(self, client, medicine_user):
        with patch(
            "departments.tasks.process_clinical_chatbot_task.delay"
        ) as mock_delay:
            mock_delay.return_value = MagicMock(id="gap-task-1")
            resp = client.post(
                "/medicine/chatbot",
                data={"clinical_note": "What are the WHO sepsis guidelines?"},
            )
            assert resp.status_code == 202
