from datetime import date

import pytest

from departments.models.records import Patient
from extensions import db


def test_patient_registration_happy_path(app):
    with app.app_context():
        patient = Patient(
            patient_id="PTEST100",
            name="Alice Smith",
            place_of_residence="Nairobi",
            sex="Female",
            date_of_birth=date(1995, 5, 15),
            marital_status="Single",
            blood_group="A+",
            contact="0700000000",
            next_of_kin="Bob Smith",
            relationship_with_next_of_kin="Brother",
            next_of_kin_contact="0711111111",
            national_id="12345678",
            emergency_contact="0722222222",
        )
        db.session.add(patient)
        db.session.commit()

        saved = Patient.query.filter_by(patient_id="PTEST100").first()
        assert saved is not None
        assert saved.name == "Alice Smith"
        assert saved.blood_group == "A+"


def test_patient_registration_missing_national_id(app):
    with app.app_context():
        patient = Patient(
            patient_id="PTEST101",
            name="No ID Patient",
            place_of_residence="Nairobi",
            sex="Male",
            date_of_birth=date(1990, 1, 1),
            marital_status="Single",
            blood_group="B+",
            contact="0700000001",
            next_of_kin="Kin",
            relationship_with_next_of_kin="Parent",
            next_of_kin_contact="0711111112",
            national_id=None,
            emergency_contact="0722222223",
        )
        db.session.add(patient)
        db.session.commit()
        saved = Patient.query.filter_by(patient_id="PTEST101").first()
        assert saved is not None
        assert saved.national_id is None


def test_patient_registration_missing_required_name(app):
    with app.app_context():
        patient = Patient(
            patient_id="PTEST102",
            name=None,
            place_of_residence="Nairobi",
            sex="Male",
            date_of_birth=date(1990, 1, 1),
            marital_status="Single",
            contact="0700000002",
            next_of_kin="Kin",
            relationship_with_next_of_kin="Parent",
            next_of_kin_contact="0711111112",
            emergency_contact="0722222223",
        )
        db.session.add(patient)
        with pytest.raises(Exception):  # noqa: B017
            db.session.commit()
        db.session.rollback()
