import unittest
from datetime import date, datetime, timezone

from werkzeug.security import generate_password_hash

from app import app, db
from departments.mch.models import AncVisit, ImmunizationRecord
from departments.models.encounter import Encounter
from departments.models.medicine import SOAPNote
from departments.models.nursing import Vitals
from departments.models.records import Patient
from departments.models.user import User


class TestFHIRAndDHIS2Exporter(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        app.config["RATELIMIT_ENABLED"] = False
        try:
            from app import limiter

            limiter.enabled = False
        except ImportError:
            pass
        self.client = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()
        db.create_all()

        # Seed admin user
        admin = User.query.filter_by(username="test_admin_fhir").first()
        if not admin:
            admin = User(
                username="test_admin_fhir",
                password=generate_password_hash("password123", method="pbkdf2:sha256"),
                role="admin",
            )
            db.session.add(admin)
        else:
            admin.password = generate_password_hash(
                "password123", method="pbkdf2:sha256"
            )
            admin.failed_login_attempts = 0
            admin.locked_until = None
        db.session.commit()

        # Seed sample patient
        patient = Patient.query.filter_by(patient_id="P-FHIR-001").first()
        if not patient:
            patient = Patient(
                patient_id="P-FHIR-001",
                name="Jane Wanjiku Doe",
                sex="F",
                date_of_birth=date(1995, 5, 20),
                contact="0712345678",
                national_id="12345678",
                place_of_residence="Nairobi",
                marital_status="Single",
                next_of_kin="John Doe",
                relationship_with_next_of_kin="Spouse",
                next_of_kin_contact="0722000000",
                emergency_contact="0722000000",
            )
            db.session.add(patient)
            db.session.commit()

        # Seed vitals record
        vitals = Vitals.query.filter_by(patient_id="P-FHIR-001").first()
        if not vitals:
            vitals = Vitals(
                patient_id="P-FHIR-001",
                nurse_id=admin.id,
                temperature=37.2,
                pulse=78,
                blood_pressure_systolic=120,
                blood_pressure_diastolic=80,
                oxygen_saturation=98,
                weight=65.0,
                height=168.0,
                timestamp=datetime.now(timezone.utc),
            )
            db.session.add(vitals)
            db.session.commit()

        # Seed SOAP note
        soap = SOAPNote.query.filter_by(patient_id="P-FHIR-001").first()
        if not soap:
            soap = SOAPNote(
                patient_id="P-FHIR-001",
                situation="Headache and fever for 3 days",
                hpi="Patient presents with high fever, chills, and headache",
                assessment="Acute Malaria",
                recommendation="Prescribe Coartem 80/480mg",
                symptoms="Fever, headache, chills",
                created_at=datetime.now(timezone.utc),
            )
            db.session.add(soap)
            db.session.commit()

        # Seed Encounter
        enc = Encounter.query.filter_by(patient_id="P-FHIR-001").first()
        if not enc:
            enc = Encounter(
                patient_id="P-FHIR-001",
                encounter_type="OPD",
                status="ACTIVE",
                chief_complaint="Fever and chills",
            )
            db.session.add(enc)
            db.session.commit()

        # Seed ANC Visit (MCH)
        anc = AncVisit.query.filter_by(patient_id=1).first()
        if not anc:
            anc = AncVisit(
                patient_id=1,
                visit_number=1,
                gestation_weeks=12,
            )
            db.session.add(anc)
            db.session.commit()

        # Seed Immunization Record
        imm = ImmunizationRecord.query.filter_by(child_patient_id=1).first()
        if not imm:
            imm = ImmunizationRecord(
                child_patient_id=1,
                vaccine_name="BCG",
                dose_number=1,
                batch_number="BATCH-2026-01",
            )
            db.session.add(imm)
            db.session.commit()

        self.patient_id = "P-FHIR-001"

    def tearDown(self):
        self.client.get("/logout")
        self.app_context.pop()

    def test_fhir_patient_resource(self):
        """Test GET /api/fhir/R4/Patient/<patient_id>."""
        self.client.post(
            "/login", data={"username": "test_admin_fhir", "password": "password123"}
        )
        res = self.client.get(f"/api/fhir/R4/Patient/{self.patient_id}")

        self.assertEqual(res.status_code, 200)
        data = res.json
        self.assertEqual(data["resourceType"], "Patient")
        self.assertEqual(data["id"], self.patient_id)
        self.assertEqual(data["gender"], "female")
        self.assertEqual(data["name"][0]["text"], "Jane Wanjiku Doe")

    def test_fhir_observation_search(self):
        """Test GET /api/fhir/R4/Observation?patient=<patient_id>."""
        self.client.post(
            "/login", data={"username": "test_admin_fhir", "password": "password123"}
        )
        res = self.client.get(f"/api/fhir/R4/Observation?patient={self.patient_id}")

        self.assertEqual(res.status_code, 200)
        data = res.json
        self.assertEqual(data["resourceType"], "Bundle")
        self.assertGreater(data["total"], 0)
        self.assertEqual(data["entry"][0]["resource"]["resourceType"], "Observation")

    def test_fhir_condition_search(self):
        """Test GET /api/fhir/R4/Condition?patient=<patient_id>."""
        self.client.post(
            "/login", data={"username": "test_admin_fhir", "password": "password123"}
        )
        res = self.client.get(f"/api/fhir/R4/Condition?patient={self.patient_id}")

        self.assertEqual(res.status_code, 200)
        data = res.json
        self.assertEqual(data["resourceType"], "Bundle")
        self.assertGreater(data["total"], 0)
        self.assertEqual(data["entry"][0]["resource"]["code"]["text"], "Acute Malaria")

    def test_fhir_encounter_search(self):
        """Test GET /api/fhir/R4/Encounter?patient=<patient_id>."""
        self.client.post(
            "/login", data={"username": "test_admin_fhir", "password": "password123"}
        )
        res = self.client.get(f"/api/fhir/R4/Encounter?patient={self.patient_id}")

        self.assertEqual(res.status_code, 200)
        data = res.json
        self.assertEqual(data["resourceType"], "Bundle")
        self.assertGreater(data["total"], 0)
        self.assertEqual(data["entry"][0]["resource"]["resourceType"], "Encounter")
        self.assertEqual(data["entry"][0]["resource"]["status"], "in-progress")

    def test_dhis2_json_export(self):
        """Test GET /api/khis/export/dhis2_json."""
        self.client.post(
            "/login", data={"username": "test_admin_fhir", "password": "password123"}
        )
        res = self.client.get("/api/khis/export/dhis2_json")

        self.assertEqual(res.status_code, 200)
        data = res.json
        self.assertEqual(data["dataSet"], "MOH_MONTHLY_SUMMARY_V2")
        self.assertIn("dataValues", data)
        self.assertTrue(len(data["dataValues"]) > 0)
        element_names = [dv["dataElement"] for dv in data["dataValues"]]
        self.assertIn("MOH731_ANC_VISITS_TOTAL", element_names)
        self.assertIn("MOH710_IMMUNIZATIONS_ADMINISTERED", element_names)

    def test_dhis2_csv_export(self):
        """Test GET /api/khis/export/csv."""
        self.client.post(
            "/login", data={"username": "test_admin_fhir", "password": "password123"}
        )
        res = self.client.get("/api/khis/export/csv")

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, "text/csv")
        self.assertIn(b"dataElement,period,orgUnit", res.data)
        self.assertIn(b"MOH731_ANC_VISITS_TOTAL", res.data)
        self.assertIn(b"MOH710_IMMUNIZATIONS_ADMINISTERED", res.data)

    def test_khis_exporter_ui_route(self):
        """Test GET /records/khis_exporter UI dashboard endpoint."""
        self.client.post(
            "/login", data={"username": "test_admin_fhir", "password": "password123"}
        )
        res = self.client.get("/records/khis_exporter")

        self.assertEqual(res.status_code, 200)
        self.assertIn(b"National KHIS & HL7 FHIR Exporter", res.data)
        self.assertIn(b"Download KHIS CSV", res.data)
        self.assertIn(b"View DHIS2 JSON", res.data)


if __name__ == "__main__":
    unittest.main()

