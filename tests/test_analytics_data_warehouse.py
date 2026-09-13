"""
tests/test_analytics_data_warehouse.py
───────────────────────────────────────
Unit & Integration tests for HIMSS EMRAM Stage 6-7 Data Warehouse & Population Health Analytics.
Tests:
  - ETL execution (run_daily_kpi_etl) with ALOS, bed occupancy, readmissions, and disease surveillance.
  - BI Dashboard view endpoint (/analytics/dashboard)
  - Executive BI API endpoint (/analytics/api/dashboard)
  - Population Health Analytics API endpoint (/analytics/api/population-health)
  - MoH DHIS2 export endpoint in JSON and CSV formats (/analytics/api/dhis2-export)
  - Manual ETL trigger endpoint (/api/analytics/etl/trigger)
"""

from datetime import date, datetime, timedelta

from departments.analytics.etl import run_daily_kpi_etl
from departments.models.encounter import Encounter
from departments.models.medicine import SOAPNote
from departments.models.records import Patient
from extensions import db


class TestAnalyticsETL:
    """Test read-optimized ETL aggregation pipeline."""

    def test_run_daily_kpi_etl_calculation(self, app):
        with app.app_context():
            today = date.today()
            start_dt = datetime.combine(today, datetime.min.time())

            p = Patient.query.filter_by(patient_id="PTETL01").first()
            if not p:
                p = Patient(patient_id="PTETL01", name="ETL Patient", sex="Male")
                db.session.add(p)
                db.session.commit()

            # Create IPD encounter started and ended today
            enc1 = Encounter(
                patient_id="PTETL01",
                encounter_type="IPD",
                started_at=start_dt - timedelta(days=3),
                ended_at=start_dt + timedelta(hours=5),
                status="DISCHARGED",
            )
            db.session.add(enc1)

            # Create active IPD encounter
            enc2 = Encounter(
                patient_id="PTETL01",
                encounter_type="IPD",
                started_at=start_dt + timedelta(hours=1),
                status="ACTIVE",
            )
            db.session.add(enc2)

            # Create SOAP note with diagnosis
            note = SOAPNote(
                patient_id="PTETL01",
                assessment="Severe Malaria with Fever",
                created_at=start_dt + timedelta(hours=2),
            )
            db.session.add(note)
            db.session.commit()

            snapshot = run_daily_kpi_etl(today)

            assert snapshot.snapshot_date == today
            assert snapshot.total_admissions >= 1
            assert snapshot.avg_length_of_stay > 0
            assert snapshot.bed_occupancy_rate >= 1.0
            assert "Severe Malaria with Fever" in snapshot.top_diagnoses_json
            assert snapshot.disease_surveillance_json.get("Malaria") >= 1


class TestAnalyticsRoutes:
    """Test Analytics & Population Health BI Endpoints."""

    def test_dashboard_view_authenticated(self, client, app):
        with client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["role"] = "admin"

        resp = client.get("/analytics/dashboard")
        assert resp.status_code == 200
        assert b"HIMSS EMRAM Stage 7 BI Console" in resp.data

    def test_api_dashboard_data(self, client, app):
        with client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["role"] = "admin"

        resp = client.get("/analytics/api/dashboard")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert "total_outpatient_visits" in data["data"]
        assert "bed_occupancy_rate" in data["data"]

    def test_api_population_health(self, client, app):
        with client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["role"] = "admin"

        resp = client.get("/analytics/api/population-health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert "disease_surveillance" in data
        assert "risk_stratification" in data

    def test_api_dhis2_export_json(self, client, app):
        with client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["role"] = "admin"

        resp = client.get("/analytics/api/dhis2-export?format=json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["orgUnit"] == "HOSPITAL_MAIN_KE"
        assert len(data["dataValues"]) >= 5

    def test_api_dhis2_export_csv(self, client, app):
        with client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["role"] = "admin"

        resp = client.get("/analytics/api/dhis2-export?format=csv")
        assert resp.status_code == 200
        assert resp.content_type == "text/csv; charset=utf-8"
        assert b"MOH_OPD_TOTAL" in resp.data

    def test_trigger_etl_endpoint(self, client, app):
        with client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["role"] = "admin"

        resp = client.post("/api/analytics/etl/trigger", json={"date": date.today().isoformat()})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
