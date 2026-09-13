"""
tests/test_adt_bed_management.py
──────────────────────────────────
Unit and integration test suite for Inpatient Bed Management, HL7 ADT events (A01, A02, A03, A08),
bed cleaning turnaround state machine, and REST API endpoints (Gap #12 Roadmap).
"""

from datetime import date

from extensions import db
from departments.models.records import Patient
from departments.models.medicine import (
    Bed,
    Ward,
    WardRoom,
)
from departments.medicine.adt_engine import ADTEngine





def _setup_wards_and_patient(app) -> tuple[str, int, int, int, int, int, int]:
    """Helper to create test patient and 2 wards with rooms and beds."""
    with app.app_context():
        p = Patient.query.filter_by(patient_id="ADT-PAT-001").first()
        if not p:
            p = Patient(
                patient_id="ADT-PAT-001",
                name="Inpatient Test Subject",
                sex="Male",
                date_of_birth=date(1985, 3, 20),
                emergency_contact="0722222222",
            )
            db.session.add(p)

        w1 = Ward.query.filter_by(name="Male Medical Ward").first()
        if not w1:
            w1 = Ward(name="Male Medical Ward", sex="Male", number_of_beds=5, occupied_beds=0, daily_charge=3000.0)
            db.session.add(w1)
            db.session.commit()

            r1 = WardRoom(ward_id=w1.id, room_number="101")
            db.session.add(r1)
            db.session.commit()

            b1 = Bed(room_id=r1.id, bed_number="101-A", occupied=False, status="AVAILABLE")
            b2 = Bed(room_id=r1.id, bed_number="101-B", occupied=False, status="AVAILABLE")
            db.session.add_all([b1, b2])

        w2 = Ward.query.filter_by(name="Surgical High Dependency Ward").first()
        if not w2:
            w2 = Ward(name="Surgical High Dependency Ward", sex="Mixed", number_of_beds=3, occupied_beds=0, daily_charge=6000.0)
            db.session.add(w2)
            db.session.commit()

            r2 = WardRoom(ward_id=w2.id, room_number="HDU-1")
            db.session.add(r2)
            db.session.commit()

            b3 = Bed(room_id=r2.id, bed_number="HDU-1A", occupied=False, status="AVAILABLE")
            db.session.add(b3)

        db.session.commit()

        b1_obj = Bed.query.filter_by(bed_number="101-A").first()
        b3_obj = Bed.query.filter_by(bed_number="HDU-1A").first()
        r1_obj = WardRoom.query.filter_by(room_number="101").first()
        r2_obj = WardRoom.query.filter_by(room_number="HDU-1").first()
        w1_obj = Ward.query.filter_by(name="Male Medical Ward").first()
        w2_obj = Ward.query.filter_by(name="Surgical High Dependency Ward").first()

        return p.patient_id, w1_obj.id, r1_obj.id, b1_obj.id, w2_obj.id, r2_obj.id, b3_obj.id


def test_adt_a01_admit_patient(app):
    """Test ADT^A01 patient admission & bed allocation."""
    pat_id, w1_id, r1_id, b1_id, _, _, _ = _setup_wards_and_patient(app)
    with app.app_context():
        adm, adt_log = ADTEngine.admit_patient_a01(
            patient_id=pat_id,
            ward_id=w1_id,
            room_id=r1_id,
            bed_id=b1_id,
            admission_criteria="Severe Pneumonia",
            user_id=1,
        )
        assert adm.id is not None
        assert adm.patient_id == pat_id

        bed = db.session.get(Bed, b1_id)
        assert bed.occupied is True
        assert bed.status == "OCCUPIED"

        ward = db.session.get(Ward, w1_id)
        assert ward.occupied_beds >= 1

        assert adt_log.event_type == "A01"
        assert "ADT^A01" in adt_log.hl7_message


def test_adt_a02_transfer_patient(app):
    """Test ADT^A02 patient bed transfer and previous bed dirty status transition."""
    pat_id, w1_id, r1_id, b1_id, w2_id, r2_id, b3_id = _setup_wards_and_patient(app)
    with app.app_context():
        # Setup active admission
        adm, _ = ADTEngine.admit_patient_a01(
            patient_id=pat_id,
            ward_id=w1_id,
            room_id=r1_id,
            bed_id=b1_id,
            admission_criteria="Observation",
        )

        # Transfer to Ward 2 Bed 3
        trans_adm, adt_log = ADTEngine.transfer_patient_a02(
            admission_id=adm.id,
            new_ward_id=w2_id,
            new_room_id=r2_id,
            new_bed_id=b3_id,
            user_id=1,
        )
        assert trans_adm.ward_id == w2_id
        assert trans_adm.bed_id == b3_id

        # Old bed should now be DIRTY
        old_bed = db.session.get(Bed, b1_id)
        assert old_bed.occupied is False
        assert old_bed.status == "DIRTY"

        # New bed should be OCCUPIED
        new_bed = db.session.get(Bed, b3_id)
        assert new_bed.occupied is True
        assert new_bed.status == "OCCUPIED"

        assert adt_log.event_type == "A02"
        assert "ADT^A02" in adt_log.hl7_message


def test_adt_a03_discharge_patient(app):
    """Test ADT^A03 patient discharge and bed dirty transition."""
    pat_id, w1_id, r1_id, b1_id, _, _, _ = _setup_wards_and_patient(app)
    with app.app_context():
        adm, _ = ADTEngine.admit_patient_a01(
            patient_id=pat_id,
            ward_id=w1_id,
            room_id=r1_id,
            bed_id=b1_id,
            admission_criteria="Observation",
        )

        dis_adm, adt_log = ADTEngine.discharge_patient_a03(
            admission_id=adm.id,
            discharge_summary="Patient fully recovered.",
            user_id=1,
        )
        assert dis_adm.discharged_on is not None

        bed = db.session.get(Bed, b1_id)
        assert bed.occupied is False
        assert bed.status == "DIRTY"

        assert adt_log.event_type == "A03"
        assert "ADT^A03" in adt_log.hl7_message


def test_bed_turnaround_state_machine(app):
    """Test housekeeping turnaround state machine: DIRTY -> CLEANING -> AVAILABLE."""
    _, _, _, b1_id, _, _, _ = _setup_wards_and_patient(app)
    with app.app_context():
        # Set to DIRTY
        b1 = ADTEngine.update_bed_status(b1_id, "DIRTY")
        assert b1.status == "DIRTY"

        # Start Cleaning
        b2 = ADTEngine.update_bed_status(b1_id, "CLEANING")
        assert b2.status == "CLEANING"

        # Finish Cleaning -> AVAILABLE
        b3 = ADTEngine.update_bed_status(b1_id, "AVAILABLE")
        assert b3.status == "AVAILABLE"
        assert b3.occupied is False


def test_ward_bed_matrix_aggregation(app):
    """Test live ward bed status matrix calculation."""
    _setup_wards_and_patient(app)
    with app.app_context():
        matrix = ADTEngine.get_ward_bed_matrix()
        assert matrix["total_beds"] >= 3
        assert "overall_occupancy_pct" in matrix
        assert len(matrix["wards"]) >= 2


def test_adt_api_endpoints(client, app, admin_user):
    """Test Inpatient ADT and Bed status HTTP REST API routes."""
    pat_id, w1_id, r1_id, b1_id, w2_id, r2_id, b3_id = _setup_wards_and_patient(app)

    # 1. Admit API (A01)
    res1 = client.post("/medicine/api/adt/admit", json={
        "patient_id": pat_id, "ward_id": w1_id, "room_id": r1_id, "bed_id": b1_id, "admission_criteria": "Acute Fever"
    })
    assert res1.status_code == 201
    adm_id = res1.get_json()["admission_id"]

    # 2. Bed Matrix API
    res2 = client.get("/medicine/api/beds/status")
    assert res2.status_code == 200
    assert res2.get_json()["total_beds"] >= 3

    # 3. Transfer API (A02)
    res3 = client.post("/medicine/api/adt/transfer", json={
        "admission_id": adm_id, "new_ward_id": w2_id, "new_room_id": r2_id, "new_bed_id": b3_id
    })
    assert res3.status_code == 200

    # 4. Update Bed Status API (Housekeeping)
    res4 = client.post(f"/medicine/api/beds/{b1_id}/status", json={"status": "CLEANING"})
    assert res4.status_code == 200
    assert res4.get_json()["new_status"] == "CLEANING"

    # 5. Discharge API (A03)
    res5 = client.post("/medicine/api/adt/discharge", json={
        "admission_id": adm_id, "discharge_summary": "Discharged home."
    })
    assert res5.status_code == 200

    # 6. ADT Events API
    res6 = client.get("/medicine/api/adt/events")
    assert res6.status_code == 200
    assert res6.get_json()["total"] >= 3

    # 7. Ward Grid Console UI
    res7 = client.get("/medicine/inpatients/ward-grid")
    assert res7.status_code == 200
    assert b"Inpatient Ward &amp; Bed Turnaround Console" in res7.data or b"Inpatient Ward & Bed Turnaround Console" in res7.data
