"""
tests/test_records_specialty_booking.py
────────────────────────────────────────
Records → specialty department booking bridge.

Verifies that:
  1. The specialty clinic seeder creates Renal & Oncology clinics (idempotent).
  2. Booking a patient into the Renal / Dialysis Clinic from Records creates
     a SCHEDULED DialysisSession visible in the renal console.
  3. Booking into the Oncology Clinic creates a Scheduled OncologyBooking
     visible on the oncology bookings board.
  4. Regular clinics do not create specialty department records.
  5. Soft-deleted (merged) patients cannot be booked from Records.
  6. Duplicate bookings are still rejected.
"""

from datetime import date, timedelta

import pytest
from werkzeug.security import generate_password_hash

from departments.models.medicine import OncologyBooking
from departments.models.records import Clinic, ClinicBooking, Patient
from departments.models.renal import DialysisSession
from departments.models.user import User
from departments.records.clinic_bridge import (
    seed_specialty_clinics,
    specialty_for_clinic_name,
)
from extensions import db

BOOKING_DATE = date.today() + timedelta(days=7)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def _make_patient(suffix):
    return Patient(
        patient_id=f"PT{suffix}",
        name=f"Patient {suffix}",
        place_of_residence="Nairobi",
        sex="Male",
        date_of_birth=date(1990, 1, 1),
        marital_status="Single",
        blood_group="O+",
        contact=f"070000{suffix}",
        next_of_kin="Next Kin",
        relationship_with_next_of_kin="Parent",
        next_of_kin_contact="0711111111",
        national_id=f"ID{suffix}",
        emergency_contact="0722222222",
    )


@pytest.fixture
def records_client(app, client):
    """Create and log in a Records-role user."""
    user = User(
        username="records_officer",
        role="records",
        password=generate_password_hash("TestPass1!"),
    )
    db.session.add(user)
    db.session.commit()
    client.post(
        "/login", data={"username": "records_officer", "password": "TestPass1!"}
    )
    return client


def _book(client, patient, clinic):
    return client.post(
        "/records/book_clinic",
        data={
            "patient_id": patient.patient_id,
            "clinic_id": clinic.clinic_id,
            "clinic_date": BOOKING_DATE.strftime("%Y-%m-%d"),
        },
    )


# ─────────────────────────────────────────────
# 1. Clinic classification & seeder
# ─────────────────────────────────────────────


class TestClinicClassification:
    def test_specialty_for_clinic_name(self):
        assert specialty_for_clinic_name("Renal / Dialysis Clinic") == "renal"
        assert specialty_for_clinic_name("Dialysis Unit") == "renal"
        assert specialty_for_clinic_name("Oncology Clinic") == "oncology"
        assert specialty_for_clinic_name("Cancer Care Centre") == "oncology"
        assert specialty_for_clinic_name("General Outpatient") is None
        assert specialty_for_clinic_name("") is None
        assert specialty_for_clinic_name(None) is None


class TestSpecialtyClinicSeeder:
    def test_seed_creates_renal_and_oncology_clinics(self, app):
        with app.app_context():
            created = seed_specialty_clinics()
            assert created == 2
            names = {c.name for c in Clinic.query.all()}
            assert "Renal / Dialysis Clinic" in names
            assert "Oncology Clinic" in names

    def test_seed_is_idempotent(self, app):
        with app.app_context():
            seed_specialty_clinics()
            assert seed_specialty_clinics() == 0
            assert Clinic.query.count() == 2

    def test_seed_skips_staff_created_equivalent(self, app):
        """A clinic already named 'Dialysis' (any case) blocks re-seeding."""
        with app.app_context():
            db.session.add(Clinic(name="renal / dialysis clinic", fee=100.0))
            db.session.commit()
            created = seed_specialty_clinics()
            assert created == 1  # only Oncology missing
            assert Clinic.query.count() == 2


# ─────────────────────────────────────────────
# 2. Booking propagation
# ─────────────────────────────────────────────


class TestSpecialtyBookingPropagation:
    def test_renal_clinic_booking_creates_scheduled_dialysis_session(
        self, app, records_client
    ):
        with app.app_context():
            seed_specialty_clinics()
            patient = _make_patient("REN1")
            db.session.add(patient)
            db.session.commit()
            clinic = Clinic.query.filter_by(name="Renal / Dialysis Clinic").first()

            resp = _book(records_client, patient, clinic)

            assert resp.status_code == 200
            assert resp.json["status"] == "success"
            assert "Renal Unit" in resp.json["message"]

            session = DialysisSession.query.one()
            assert session.patient_id == "PTREN1"
            assert session.status == "SCHEDULED"
            assert session.session_date == BOOKING_DATE
            assert session.modality in ("HD", "CRRT")
            assert session.source == "RECORDS"
            assert "Booked from Records" in (session.notes or "")
            assert ClinicBooking.query.count() == 1

    def test_oncology_clinic_booking_creates_scheduled_oncology_booking(
        self, app, records_client
    ):
        with app.app_context():
            seed_specialty_clinics()
            patient = _make_patient("ONC1")
            db.session.add(patient)
            db.session.commit()
            clinic = Clinic.query.filter_by(name="Oncology Clinic").first()

            resp = _book(records_client, patient, clinic)

            assert resp.status_code == 200
            assert "Oncology board" in resp.json["message"]

            booking = OncologyBooking.query.one()
            assert booking.patient_id == "PTONC1"
            assert booking.status == "Scheduled"
            assert booking.purpose == "Consultation"
            assert booking.booking_date == BOOKING_DATE
            assert booking.source == "RECORDS"
            assert "Booked from Records" in (booking.notes or "")

    def test_general_clinic_booking_creates_no_specialty_records(
        self, app, records_client
    ):
        with app.app_context():
            db.session.add(Clinic(name="General Outpatient", fee=500.0))
            db.session.commit()
            patient = _make_patient("GEN1")
            db.session.add(patient)
            db.session.commit()
            clinic = Clinic.query.filter_by(name="General Outpatient").first()

            resp = _book(records_client, patient, clinic)

            assert resp.status_code == 200
            assert DialysisSession.query.count() == 0
            assert OncologyBooking.query.count() == 0
            assert ClinicBooking.query.count() == 1

    def test_renal_propagation_dedupes_per_patient_and_date(
        self, app, records_client
    ):
        with app.app_context():
            seed_specialty_clinics()
            patient = _make_patient("REN2")
            db.session.add(patient)
            db.session.commit()
            clinic = Clinic.query.filter_by(name="Renal / Dialysis Clinic").first()

            assert _book(records_client, patient, clinic).status_code == 200
            assert _book(records_client, patient, clinic).status_code == 400
            assert DialysisSession.query.count() == 1


# ─────────────────────────────────────────────
# 3. Guard rails
# ─────────────────────────────────────────────


class TestBookingGuardRails:
    def test_soft_deleted_patient_cannot_be_booked(self, app, records_client):
        with app.app_context():
            seed_specialty_clinics()
            patient = _make_patient("DEL1")
            db.session.add(patient)
            db.session.commit()
            patient.soft_delete()
            db.session.commit()
            clinic = Clinic.query.filter_by(name="Oncology Clinic").first()

            resp = _book(records_client, patient, clinic)

            assert resp.status_code == 400
            assert "inactive" in resp.json["message"]
            assert ClinicBooking.query.count() == 0
            assert OncologyBooking.query.count() == 0

    def test_unknown_patient_rejected(self, app, records_client):
        with app.app_context():
            seed_specialty_clinics()
            clinic = Clinic.query.filter_by(name="Oncology Clinic").first()

            resp = records_client.post(
                "/records/book_clinic",
                data={
                    "patient_id": "PTNOPE",
                    "clinic_id": clinic.clinic_id,
                    "clinic_date": BOOKING_DATE.strftime("%Y-%m-%d"),
                },
            )

            assert resp.status_code == 400
            assert ClinicBooking.query.count() == 0


# ─────────────────────────────────────────────────────────────
# 4. Oncology board parity (source badge & filter)
# ─────────────────────────────────────────────────────────────


class TestOncologySourceParity:
    """Records bookings must be visible & filterable on the oncology board."""

    @staticmethod
    def _seed_two_bookings():
        """One RECORDS booking + one unit booking with distinct patient names."""
        from departments.models.user import User
        from werkzeug.security import generate_password_hash as gph

        from extensions import db

        records_p = _make_patient("OCR1")
        records_p.name = "Records Referred"
        unit_p = _make_patient("OCU1")
        unit_p.name = "Unit Booked"
        db.session.add_all([records_p, unit_p])
        db.session.commit()

        db.session.add(
            OncologyBooking(
                patient_id="PTOCR1", booking_date=BOOKING_DATE,
                purpose="Consultation", status="Scheduled", source="RECORDS",
                notes="Booked from Records — Oncology Clinic.",
            )
        )
        db.session.add(
            OncologyBooking(
                patient_id="PTOCU1", booking_date=BOOKING_DATE,
                purpose="Chemotherapy", status="Scheduled", source="ONCOLOGY",
            )
        )
        db.session.commit()

    def test_oncology_board_source_filter(self, app, admin_user):
        self._seed_two_bookings()

        resp = app.test_client().get("/medicine/bookings/?source=RECORDS")
        assert resp.status_code == 200
        html = resp.data
        assert b"Records Referred" in html          # records booking shown
        assert b"Unit Booked" not in html           # unit booking filtered out
        assert html.count(b"fas fa-clipboard") == 1  # exactly one Records badge
        assert b'value="RECORDS" selected' in html   # filter stays selected

    def test_oncology_board_shows_all_sources_by_default(self, app, admin_user):
        self._seed_two_bookings()

        resp = app.test_client().get("/medicine/bookings/")
        assert resp.status_code == 200
        html = resp.data
        assert b"Records Referred" in html
        assert b"Unit Booked" in html
        assert html.count(b"fas fa-clipboard") == 1  # only the records card is badged

    def test_oncology_board_invalid_source_ignored(self, app, admin_user):
        self._seed_two_bookings()

        resp = app.test_client().get("/medicine/bookings/?source=bogus")
        assert resp.status_code == 200
        html = resp.data
        assert b"Records Referred" in html
        assert b"Unit Booked" in html
