"""
tests/test_theatre_surgical_module.py
──────────────────────────────────────
Comprehensive Unit and Integration Test Suite for Johns Hopkins–Grade
Theatre & Surgical Module (Gap #9):
- WHO 3-stage Surgical Safety Checklist (Sign In, Time Out, Sign Out)
- Anaesthetic Record, ASA Classification, Airway Grade, & Intraoperative Vitals
- Post-Operative Operative Note & PACU Aldrete Recovery Score (0-10)
- Surgical Instrument, Sponge, and Needle Reconciliation & Discrepancy Detection
- REST API & Flask Route Integration
"""

from datetime import date

import pytest

from departments.models.medicine import TheatreList, TheatreProcedure
from departments.models.records import Patient
from departments.models.theatre import (
    AnaestheticRecord,
    PostOpNote,
    SurgicalInstrumentCount,
    WhoSurgicalChecklist,
)
from extensions import db


@pytest.fixture
def sample_patient_id(app):
    with app.app_context():
        p = Patient(
            patient_id="P-SURG-001",
            name="John Surgical Patient",
            sex="Male",
            date_of_birth=date(1980, 5, 15),
            contact="0712345678",
        )
        db.session.add(p)
        db.session.commit()
        return "P-SURG-001"


@pytest.fixture
def sample_procedure_id(app):
    with app.app_context():
        proc = TheatreProcedure(
            name="Laparoscopic Appendectomy",
            type="General Surgery",
            cost=45000.00,
        )
        db.session.add(proc)
        db.session.commit()
        return proc.id


@pytest.fixture
def sample_theatre_entry_id(app, sample_patient_id, sample_procedure_id):
    with app.app_context():
        from departments.medicine.inpatients import create_surgical_encounter

        surgical_enc = create_surgical_encounter(
            patient_id=sample_patient_id,
            provider_id="1",
            chief_complaint="Surgical procedure: Laparoscopic Appendectomy",
        )

        entry = TheatreList(
            patient_id=sample_patient_id,
            procedure_id=sample_procedure_id,
            status=0,
            notes_on_book="Acute Appendicitis",
            encounter_id=surgical_enc.id,
        )
        db.session.add(entry)
        db.session.commit()
        return entry.id


class TestWhoSurgicalChecklist:
    def test_create_who_checklist_default_state(self, app, sample_theatre_entry_id):
        with app.app_context():
            entry = db.session.get(TheatreList, sample_theatre_entry_id)
            checklist = WhoSurgicalChecklist(
                theatre_entry_id=entry.id,
                patient_id=entry.patient_id,
                encounter_id=entry.encounter_id,
            )
            db.session.add(checklist)
            db.session.commit()

            assert checklist.id is not None
            assert checklist.sign_in_completed is False
            assert checklist.time_out_completed is False
            assert checklist.sign_out_completed is False
            assert checklist.is_fully_completed() is False

    def test_sign_in_stage_completion(self, app, sample_theatre_entry_id):
        with app.app_context():
            entry = db.session.get(TheatreList, sample_theatre_entry_id)
            checklist = WhoSurgicalChecklist(
                theatre_entry_id=entry.id,
                patient_id=entry.patient_id,
                patient_identity_confirmed=True,
                site_marked=True,
                anaesthesia_safety_check_completed=True,
                pulse_oximeter_functioning=True,
                sign_in_completed=True,
            )
            db.session.add(checklist)
            db.session.commit()

            assert checklist.sign_in_completed is True
            assert checklist.patient_identity_confirmed is True
            assert checklist.is_fully_completed() is False

    def test_time_out_and_sign_out_full_progression(self, app, sample_theatre_entry_id):
        with app.app_context():
            entry = db.session.get(TheatreList, sample_theatre_entry_id)
            checklist = WhoSurgicalChecklist(
                theatre_entry_id=entry.id,
                patient_id=entry.patient_id,
                sign_in_completed=True,
                time_out_completed=True,
                sign_out_completed=True,
            )
            db.session.add(checklist)
            db.session.commit()

            assert checklist.is_fully_completed() is True


class TestAnaestheticRecord:
    def test_create_anaesthetic_record_defaults(self, app, sample_theatre_entry_id):
        with app.app_context():
            entry = db.session.get(TheatreList, sample_theatre_entry_id)
            record = AnaestheticRecord(
                theatre_entry_id=entry.id,
                patient_id=entry.patient_id,
                asa_status="ASA II",
                technique="General",
            )
            db.session.add(record)
            db.session.commit()

            assert record.id is not None
            assert record.asa_status == "ASA II"
            assert record.technique == "General"

    def test_agents_and_vitals_json_properties(self, app, sample_theatre_entry_id):
        with app.app_context():
            entry = db.session.get(TheatreList, sample_theatre_entry_id)
            record = AnaestheticRecord(
                theatre_entry_id=entry.id,
                patient_id=entry.patient_id,
                asa_status="ASA I",
            )
            record.agents_administered = [
                {"agent": "Propofol", "dose": "150 mg", "time": "08:30"},
                {"agent": "Fentanyl", "dose": "100 mcg", "time": "08:32"},
            ]
            record.vitals_series = [
                {"time": "08:35", "hr": 72, "bp": "120/80", "spo2": 99, "etco2": 35},
                {"time": "08:45", "hr": 68, "bp": "115/75", "spo2": 100, "etco2": 36},
            ]
            db.session.add(record)
            db.session.commit()

            fetched = db.session.get(AnaestheticRecord, record.id)
            assert len(fetched.agents_administered) == 2
            assert fetched.agents_administered[0]["agent"] == "Propofol"
            assert len(fetched.vitals_series) == 2
            assert fetched.vitals_series[1]["spo2"] == 100


class TestPostOpNote:
    def test_aldrete_score_calculation(self, app, sample_theatre_entry_id):
        with app.app_context():
            entry = db.session.get(TheatreList, sample_theatre_entry_id)
            note = PostOpNote(
                theatre_entry_id=entry.id,
                patient_id=entry.patient_id,
                preop_diagnosis="Acute Appendicitis",
                postop_diagnosis="Acute Suppurative Appendicitis",
                procedure_performed="Laparoscopic Appendectomy",
                surgical_findings="Inflamed appendix identified and removed.",
                aldrete_activity=2,
                aldrete_respiration=2,
                aldrete_circulation=2,
                aldrete_consciousness=2,
                aldrete_spo2=2,
            )
            db.session.add(note)
            db.session.commit()

            assert note.total_aldrete_score == 10
            assert note.is_fit_for_pacu_discharge() is True

    def test_aldrete_score_below_threshold_fails_pacu_discharge(
        self, app, sample_theatre_entry_id
    ):
        with app.app_context():
            entry = db.session.get(TheatreList, sample_theatre_entry_id)
            note = PostOpNote(
                theatre_entry_id=entry.id,
                patient_id=entry.patient_id,
                preop_diagnosis="Acute Appendicitis",
                postop_diagnosis="Acute Appendicitis",
                procedure_performed="Laparoscopic Appendectomy",
                surgical_findings="Uneventful.",
                aldrete_activity=1,
                aldrete_respiration=1,
                aldrete_circulation=2,
                aldrete_consciousness=1,
                aldrete_spo2=2,
            )
            db.session.add(note)
            db.session.commit()

            assert note.total_aldrete_score == 7
            assert note.is_fit_for_pacu_discharge() is False


class TestSurgicalInstrumentCount:
    def test_count_reconciliation_success(self, app, sample_theatre_entry_id):
        with app.app_context():
            entry = db.session.get(TheatreList, sample_theatre_entry_id)
            counts = SurgicalInstrumentCount(
                theatre_entry_id=entry.id,
                sponges_initial=10,
                sponges_added=5,
                sponges_closing_cavity=15,
                sponges_closing_skin=15,
                needles_initial=6,
                needles_added=0,
                needles_closing_cavity=6,
                needles_closing_skin=6,
                instruments_initial=25,
                instruments_added=0,
                instruments_closing_cavity=25,
                instruments_closing_skin=25,
            )
            result = counts.calculate_reconciliation()
            db.session.add(counts)
            db.session.commit()

            assert result["reconciled"] is True
            assert counts.count_reconciled is True

    def test_count_reconciliation_discrepancy_detection(
        self, app, sample_theatre_entry_id
    ):
        with app.app_context():
            entry = db.session.get(TheatreList, sample_theatre_entry_id)
            counts = SurgicalInstrumentCount(
                theatre_entry_id=entry.id,
                sponges_initial=10,
                sponges_added=0,
                sponges_closing_cavity=9,  # 1 sponge missing!
                sponges_closing_skin=9,
                needles_initial=6,
                needles_closing_skin=6,
                instruments_initial=25,
                instruments_closing_skin=25,
            )
            result = counts.calculate_reconciliation()
            db.session.add(counts)
            db.session.commit()

            assert result["reconciled"] is False
            assert result["sponge_diff"] == -1
            assert counts.count_reconciled is False


class TestTheatreModuleRoutes:
    def test_surgical_workbench_requires_auth(self, client):
        resp = client.get("/theatre/workbench/1")
        assert resp.status_code in (302, 401)

    def test_who_checklist_api_endpoint(
        self, client, app, admin_user, sample_theatre_entry_id
    ):
        with client:
            resp = client.post(
                f"/theatre/checklist/{sample_theatre_entry_id}",
                json={
                    "stage": "sign_in",
                    "patient_identity_confirmed": True,
                    "site_marked": True,
                    "anaesthesia_safety_check_completed": True,
                    "pulse_oximeter_functioning": True,
                },
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["status"] == "success"

    def test_anaesthetic_record_api_endpoint(
        self, client, app, admin_user, sample_theatre_entry_id
    ):
        with client:
            resp = client.post(
                f"/theatre/anaesthetic/{sample_theatre_entry_id}",
                json={
                    "asa_status": "ASA III",
                    "technique": "Spinal",
                    "estimated_blood_loss_ml": 150,
                },
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["asa_status"] == "ASA III"

    def test_postop_note_api_endpoint(
        self, client, app, admin_user, sample_theatre_entry_id
    ):
        with client:
            resp = client.post(
                f"/theatre/postop/{sample_theatre_entry_id}",
                json={
                    "preop_diagnosis": "Appendicitis",
                    "postop_diagnosis": "Acute Appendicitis",
                    "surgical_findings": "Appendix removed cleanly.",
                    "aldrete_activity": 2,
                    "aldrete_respiration": 2,
                    "aldrete_circulation": 2,
                    "aldrete_consciousness": 2,
                    "aldrete_spo2": 2,
                },
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["total_aldrete_score"] == 10
            assert data["fit_for_pacu_discharge"] is True

    def test_instrument_count_api_endpoint(
        self, client, app, admin_user, sample_theatre_entry_id
    ):
        with client:
            resp = client.post(
                f"/theatre/instruments/{sample_theatre_entry_id}",
                json={
                    "sponges_initial": 10,
                    "sponges_closing_skin": 10,
                    "needles_initial": 5,
                    "needles_closing_skin": 5,
                    "instruments_initial": 20,
                    "instruments_closing_skin": 20,
                },
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["status"] == "success"
            assert data["reconciliation"]["reconciled"] is True
