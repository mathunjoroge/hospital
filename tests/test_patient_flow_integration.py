from werkzeug.security import generate_password_hash

from departments.appointments.models import Appointment
from departments.models.records import Patient, PatientWaitingList
from departments.models.user import User
from departments.shared.queue_constants import QueueStatus
from extensions import db


def test_unified_patient_flow(app, client):
    """
    Test complete unified flow from patient registration to live queue dashboard,
    nitals, doctor consultation, and discharge.
    """
    with app.app_context():
        # Create test admin/doctor user
        user = User(
            id=1,
            username="testadmin",
            password=generate_password_hash("password123"),
            role="admin",
        )
        db.session.add(user)
        db.session.commit()

        # Log in
        client.post(
            "/login",
            data={"username": "testadmin", "password": "password123"},
            follow_redirects=True,
        )

        # 1. Register a new patient
        resp = client.post(
            "/records/new_patient",
            data={
                "name": "Jane Doe",
                "place_of_residence": "Nairobi",
                "sex": "Female",
                "date_of_birth": "1995-05-15",
                "marital_status": "Single",
                "blood_group": "O+",
                "contact": "0712345678",
                "next_of_kin": "John Doe",
                "relationship_with_next_of_kin": "Brother",
                "next_of_kin_contact": "0787654321",
                "national_id": "12345678",
                "emergency_contact": "0712345678",
            },
            follow_redirects=True,
        )
        assert resp.status_code == 200

        # Retrieve registered patient
        patient = Patient.query.filter_by(name="Jane Doe").first()
        assert patient is not None
        patient_id = patient.patient_id

        # Phase 4: PatientWaitingList is retired. Check Encounter instead.
        from departments.models.encounter import Encounter
        enc = Encounter.query.filter_by(patient_id=patient_id).first()
        assert enc is not None, "Encounter should be created on registration"
        assert enc.stage == "REGISTERED", "New encounter should start at REGISTERED stage"
            
        # Legacy table should no longer be written to
        waiting_entry = PatientWaitingList.query.filter_by(patient_id=patient_id).first()
        assert waiting_entry is None, "PatientWaitingList should not be created in Phase 4"

        # Check Appointment record (Bridging working!)
        appt = Appointment.query.filter_by(patient_id=patient_id).first()
        assert appt is not None
        assert appt.status == "CHECKED_IN"

        # 2. Check Live Queue API returns patient name instead of generic ID
        queue_resp = client.get("/appointments/api/queue/all", follow_redirects=True)
        assert queue_resp.status_code == 200
        data = queue_resp.get_json()
        assert data["waiting_count"] == 1
        assert data["queue"][0]["patient_id"] == patient_id
        assert data["queue"][0]["patient_name"] == "Jane Doe"

        # 3. Record Vitals in Nursing
        vitals_resp = client.post(
            f"/nursing/vitals/{patient_id}",
            data={
                "temperature": "37.2",
                "pulse": "72",
                "blood_pressure_systolic": "120",
                "blood_pressure_diastolic": "80",
                "respiratory_rate": "16",
                "oxygen_saturation": "98",
                "blood_glucose": "5.5",
                "weight": "65",
                "height": "170",
            },
            follow_redirects=True,
        )
        assert vitals_resp.status_code == 200
        # Phase 4: Check Encounter instead of retired PatientWaitingList
        enc = Encounter.query.filter_by(patient_id=patient_id).first()
        assert enc is not None
        assert enc.status == "ACTIVE"

        # 4. Doctor opens SOAP Notes
        soap_get_resp = client.get(f"/medicine/soap_notes/{patient_id}", follow_redirects=True)
        assert soap_get_resp.status_code == 200
        # Phase 4: Check Encounter instead of retired PatientWaitingList
        enc = Encounter.query.filter_by(patient_id=patient_id).first()
        assert enc is not None
        assert enc.status == "ACTIVE"
        appt = Appointment.query.filter_by(patient_id=patient_id).first()
        assert appt.status == "IN_PROGRESS"

        # 5. Doctor submits SOAP Notes
        soap_post_resp = client.post(
            f"/medicine/submit_soap_notes/{patient_id}",
            data={
                "situation": "Patient complains of mild headache",
                "hpi": "Started 2 days ago",
                "assessment": "Tension headache",
                "recommendation": "Rest and paracetamol",
            },
            follow_redirects=True,
        )
        assert soap_post_resp.status_code == 200
        # Phase 4: Check Encounter instead of retired PatientWaitingList
        enc = Encounter.query.filter_by(patient_id=patient_id).first()
        assert enc is not None
        assert enc.status == "DISCHARGED"
        appt = Appointment.query.filter_by(patient_id=patient_id).first()
        assert appt.status == "COMPLETED"

        # 6. Verify Queue is now clear
        queue_resp2 = client.get("/appointments/api/queue/all", follow_redirects=True)
        data2 = queue_resp2.get_json()
        assert data2["waiting_count"] == 0
