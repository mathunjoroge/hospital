from datetime import date

from departments.models.admin import Log
from departments.models.records import Patient
from extensions import db


def test_audit_logging_on_patient_crud(app):
    with app.app_context():
        # 1. CREATE Patient
        patient = Patient(
            patient_id="PTEST001",
            name="John Audit",
            place_of_residence="123 Audit St",
            sex="Male",
            date_of_birth=date(1990, 1, 1),
            marital_status="Single",
            blood_group="O+",
            contact="1234567890",
            next_of_kin="Jane Audit",
            relationship_with_next_of_kin="Sister",
            next_of_kin_contact="0987654321",
            national_id="ID999000111",
            emergency_contact="1234567890"
        )
        db.session.add(patient)
        db.session.commit()

        # Verify INSERT log created
        insert_logs = Log.query.filter(Log.message.like("%Audit [INSERT] Patient%")).all()
        assert len(insert_logs) >= 1
        assert "PTEST001" in insert_logs[0].message or "Patient" in insert_logs[0].message

        # 2. UPDATE Patient
        patient.name = "John Audit Updated"
        db.session.commit()

        update_logs = Log.query.filter(Log.message.like("%Audit [UPDATE] Patient%")).all()
        assert len(update_logs) >= 1

        # 3. DELETE Patient
        db.session.delete(patient)
        db.session.commit()

        delete_logs = Log.query.filter(Log.message.like("%Audit [DELETE] Patient%")).all()
        assert len(delete_logs) >= 1
