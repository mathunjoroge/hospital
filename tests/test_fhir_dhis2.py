import unittest
from datetime import date, datetime, timezone

from werkzeug.security import generate_password_hash

from app import app, db
from departments.malaria.models import MalariaCase
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

        # Seed Malaria Case (confirmed, RDT, adult/over-5 patient)
        malaria_case = MalariaCase.query.filter_by(case_number="MAL-FHIR-001").first()
        if not malaria_case:
            malaria_case = MalariaCase(
                patient_id="P-FHIR-001",
                case_number="MAL-FHIR-001",
                malaria_species="falciparum",
                diagnosis_method="RDT",
                severity="uncomplicated",
                pregnancy_status="not_pregnant",
                diagnosis_date=datetime.now(timezone.utc),
            )
            db.session.add(malaria_case)
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
        self.assertIn("MOH705_MALARIA_CONFIRMED_TOTAL", element_names)
        self.assertIn("khisReady", data)

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
        self.assertIn(b"MOH705_MALARIA_CONFIRMED_TOTAL", res.data)

    def test_dhis2_malaria_case_disaggregation(self):
        """Confirmed malaria case should be counted, correctly disaggregated
        by age band and diagnosis method, and reflected in the summary."""
        self.client.post(
            "/login", data={"username": "test_admin_fhir", "password": "password123"}
        )
        res = self.client.get("/api/khis/reports/monthly")

        self.assertEqual(res.status_code, 200)
        data = res.json
        elements = {e["dataElement"]: e["value"] for e in data["data_elements"]}

        # Seeded patient (Jane Wanjiku Doe, DOB 1995-05-20) is over 5, and
        # the seeded case used diagnosis_method="RDT".
        self.assertGreaterEqual(elements["MOH705_MALARIA_CONFIRMED_RDT_OVER5"], 1)
        self.assertEqual(elements["MOH705_MALARIA_CONFIRMED_RDT_UNDER5"], 0)
        self.assertGreaterEqual(elements["MOH705_MALARIA_CONFIRMED_TOTAL"], 1)
        self.assertGreaterEqual(data["summary"]["malaria_confirmed_total"], 1)
        self.assertIn("khis_upload_readiness", data)
        self.assertIn("unmapped_data_elements", data["khis_upload_readiness"])

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
        self.assertIn(b"MOH 645 &amp; MOH 743", res.data)

    def test_moh645_743_weight_band_classification(self):
        """Test weight band classification and age fallback."""
        from departments.malaria.moh_645_743 import classify_al_weight_band, estimate_weight_from_age, classify_drug_category

        self.assertEqual(classify_al_weight_band(10.0), "AL6")
        self.assertEqual(classify_al_weight_band(18.0), "AL12")
        self.assertEqual(classify_al_weight_band(30.0), "AL18")
        self.assertEqual(classify_al_weight_band(45.0), "AL24")

        self.assertEqual(estimate_weight_from_age(2), 10.0)
        self.assertEqual(estimate_weight_from_age(5), 20.0)
        self.assertEqual(estimate_weight_from_age(10), 30.0)
        self.assertEqual(estimate_weight_from_age(20), 45.0)

        self.assertEqual(classify_drug_category("Artemether 20mg / Lumefantrine 120mg"), "al")
        self.assertEqual(classify_drug_category("Artesunate 60mg Injectable"), "artesunate")
        self.assertEqual(classify_drug_category("Quinine Sulphate 300mg"), "quinine")
        self.assertEqual(classify_drug_category("Sulfadoxine Pyrimethamine 500/25mg"), "sp")

    def test_moh645_743_dhis2_export_elements(self):
        """Test that MOH-645/743 data elements are present in DHIS2 export outputs."""
        self.client.post(
            "/login", data={"username": "test_admin_fhir", "password": "password123"}
        )
        res = self.client.get("/api/khis/export/dhis2_json")
        self.assertEqual(res.status_code, 200)
        data = res.json

        element_names = [dv["dataElement"] for dv in data["dataValues"]]
        self.assertIn("MOH645_AL6_DISPENSED", element_names)
        self.assertIn("MOH645_AL12_DISPENSED", element_names)
        self.assertIn("MOH645_AL18_DISPENSED", element_names)
        self.assertIn("MOH645_AL24_DISPENSED", element_names)
        self.assertIn("MOH645_ARTESUNATE_INJ_DISPENSED", element_names)
        self.assertIn("MOH645_QUININE_DISPENSED", element_names)
        self.assertIn("MOH645_SP_DISPENSED", element_names)
        self.assertIn("MOH645_PATIENTS_TREATED_BY_WBAND_35PLUS", element_names)

    def test_moh647_tracer_hpt_aggregation(self):
        """Test MOH 647 Tracer HPT commodity categorization and export elements."""
        from departments.pharmacy.moh_647 import classify_tracer_item, aggregate_moh647_monthly

        match1 = classify_tracer_item("Amoxicillin 250mg Capsules")
        self.assertIsNotNone(match1)
        self.assertEqual(match1["category"], "Antimicrobials")

        match2 = classify_tracer_item("Oxytocin 10IU Injection")
        self.assertIsNotNone(match2)
        self.assertEqual(match2["category"], "Maternal & Reproductive")

        self.client.post(
            "/login", data={"username": "test_admin_fhir", "password": "password123"}
        )
        res = self.client.get("/api/khis/export/dhis2_json")
        self.assertEqual(res.status_code, 200)
        data = res.json

        element_names = [dv["dataElement"] for dv in data["dataValues"]]
        self.assertIn("MOH647_TOTAL_TRACER_ITEMS_MONITORED", element_names)
        self.assertIn("MOH647_TRACER_ITEMS_IN_STOCK", element_names)
        self.assertIn("MOH647_TRACER_ITEMS_STOCKOUT_COUNT", element_names)


if __name__ == "__main__":
    unittest.main()

