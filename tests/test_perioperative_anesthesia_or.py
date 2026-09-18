"""
tests/test_perioperative_anesthesia_or.py
──────────────────────────────────────────
Unit & integration tests for Gap #9: Perioperative / OR Scheduling & Anesthesia Log Matrix.
"""

from datetime import datetime, timedelta, timezone

from departments.models.medicine import TheatreList, TheatreProcedure
from departments.models.records import Patient
from departments.models.theatre import AnaestheticRecord
from departments.theatre.theatre_engine import TheatreOperationsEngine
from extensions import db


def test_asa_physical_status_scoring():
    """Test ASA Physical Status classification & emergency mortality risk calculation."""
    # ASA I standard
    asa1 = TheatreOperationsEngine.evaluate_asa_score("ASA I", is_emergency=False)
    assert asa1["asa_code"] == "ASA I"
    assert asa1["risk_level"] == "LOW"
    assert asa1["estimated_mortality_pct"] == 0.05

    # ASA III Emergency (-E)
    asa3_e = TheatreOperationsEngine.evaluate_asa_score("ASA III-E")
    assert asa3_e["asa_code"] == "ASA IIIE"
    assert asa3_e["is_emergency"] is True
    assert "EMERGENCY" in asa3_e["risk_level"]
    # 1.8 * 2.0 = 3.6%
    assert asa3_e["estimated_mortality_pct"] == 3.6

    # ASA V Emergency
    asa5_e = TheatreOperationsEngine.evaluate_asa_score("ASA V", is_emergency=True)
    assert asa5_e["is_emergency"] is True
    # 9.4 * 2.0 = 18.8%
    assert asa5_e["estimated_mortality_pct"] == 18.8


def test_anesthesia_timeline_events(app):
    """Test timestamped anesthesia timeline event matrix creation & retrieval."""
    with app.app_context():
        pat = Patient(
            patient_id="PAT_OR_01",
            name="Sarah OR Doe",
            sex="Female",
            date_of_birth=datetime(1992, 4, 10).date(),
        )
        proc = TheatreProcedure(name="Appendectomy", cost=35000.0)
        db.session.add_all([pat, proc])
        db.session.commit()

        entry = TheatreList(
            patient_id="PAT_OR_01", procedure_id=proc.id, or_room="OR 1"
        )
        db.session.add(entry)
        db.session.commit()

        # Record timeline events
        ev1 = TheatreOperationsEngine.record_timeline_event(
            entry.id, "PRE_INDUCTION", notes="Patient premedicated."
        )
        assert ev1["event_type"] == "PRE_INDUCTION"

        ev2 = TheatreOperationsEngine.record_timeline_event(
            entry.id, "INDUCTION", notes="Propofol 150mg + Fentanyl 100mcg."
        )
        assert ev2["event_type"] == "INDUCTION"

        ev3 = TheatreOperationsEngine.record_timeline_event(
            entry.id, "INTUBATION", notes="ETT 7.5 cuffed placed smoothly."
        )
        assert ev3["event_type"] == "INTUBATION"

        ev4 = TheatreOperationsEngine.record_timeline_event(
            entry.id, "INCISION", notes="Skin incision right lower quadrant."
        )
        assert ev4["event_type"] == "INCISION"

        ev5 = TheatreOperationsEngine.record_timeline_event(
            entry.id, "PACU_TRANSFER", notes="Transferred to PACU in stable condition."
        )
        assert ev5["event_type"] == "PACU_TRANSFER"

        # Fetch matrix
        matrix = TheatreOperationsEngine.get_anesthesia_timeline_matrix(entry.id)
        assert len(matrix["events"]) == 5
        event_types = [e["event_type"] for e in matrix["events"]]
        assert "INDUCTION" in event_types
        assert "PACU_TRANSFER" in event_types

        # Verify auto-updated anesthesia start/end times on record
        record = AnaestheticRecord.query.filter_by(theatre_entry_id=entry.id).first()
        assert record.anaesthesia_start_time is not None
        assert record.anaesthesia_end_time is not None


def test_or_room_schedule_conflict_detection(app):
    """Test OR room scheduling & time collision detection."""
    with app.app_context():
        pat = Patient(
            patient_id="PAT_OR_SCH",
            name="Schedule Patient",
            sex="Male",
            date_of_birth=datetime(1988, 1, 1).date(),
        )
        proc = TheatreProcedure(name="Laparoscopy", cost=45000.0)
        db.session.add_all([pat, proc])
        db.session.commit()

        now = datetime.now(timezone.utc)
        start_time = now + timedelta(hours=2)

        # Create scheduled case in OR 2 from T+2h to T+4h (120 mins)
        entry1 = TheatreList(
            patient_id="PAT_OR_SCH",
            procedure_id=proc.id,
            or_room="OR 2",
            scheduled_start_time=start_time,
            estimated_duration_minutes=120,
            status=0,
        )
        db.session.add(entry1)
        db.session.commit()

        # Test collision in OR 2 at overlapping time (T+3h)
        overlapping_start = start_time + timedelta(minutes=60)
        has_conflict, msg = TheatreOperationsEngine.check_room_schedule_conflict(
            or_room="OR 2",
            start_time=overlapping_start,
            duration_minutes=60,
        )
        assert has_conflict is True
        assert "Schedule collision in OR 2" in msg

        # Test non-overlapping time in OR 2 (T+5h)
        non_overlap_start = start_time + timedelta(minutes=180)
        no_conflict, _ = TheatreOperationsEngine.check_room_schedule_conflict(
            or_room="OR 2",
            start_time=non_overlap_start,
            duration_minutes=60,
        )
        assert no_conflict is False

        # Test different room (OR 3) at same time
        no_conflict_diff_room, _ = TheatreOperationsEngine.check_room_schedule_conflict(
            or_room="OR 3",
            start_time=overlapping_start,
            duration_minutes=60,
        )
        assert no_conflict_diff_room is False


def test_or_utilization_metrics(app):
    """Test OR room utilization percentages and ASA distribution summary."""
    with app.app_context():
        metrics = TheatreOperationsEngine.get_or_dashboard_metrics()
        assert "or_utilization_pct" in metrics
        assert "room_stats" in metrics
        assert "asa_counts" in metrics
        assert isinstance(metrics["or_cases"], list)


def test_theatre_api_endpoints(client, app, admin_user):
    """Test HTTP API endpoints for Anesthesia Timeline, ASA Assessment, and OR Scheduling."""
    with app.app_context():
        pat = Patient(
            patient_id="PAT_OR_API",
            name="API Patient",
            sex="Female",
            date_of_birth=datetime(1995, 2, 2).date(),
        )
        proc = TheatreProcedure(name="Cholecystectomy", cost=50000.0)
        db.session.add_all([pat, proc])
        db.session.commit()

        entry = TheatreList(
            patient_id="PAT_OR_API", procedure_id=proc.id, or_room="OR 1"
        )
        db.session.add(entry)
        db.session.commit()
        entry_id = entry.id

    # 1. Test POST /theatre/api/anesthesia-timeline/<id>
    resp_timeline_post = client.post(
        f"/theatre/api/anesthesia-timeline/{entry_id}",
        json={"event_type": "INDUCTION", "notes": "Rapid sequence induction"},
    )
    assert resp_timeline_post.status_code == 201
    assert resp_timeline_post.get_json()["status"] == "success"

    # 2. Test GET /theatre/api/anesthesia-timeline/<id>
    resp_timeline_get = client.get(f"/theatre/api/anesthesia-timeline/{entry_id}")
    assert resp_timeline_get.status_code == 200
    data_timeline = resp_timeline_get.get_json()
    assert len(data_timeline["events"]) >= 1

    # 3. Test GET /theatre/api/asa-assessment/<code>
    resp_asa = client.get("/theatre/api/asa-assessment/ASA III?emergency=true")
    assert resp_asa.status_code == 200
    data_asa = resp_asa.get_json()
    assert data_asa["asa_code"] == "ASA IIIE"
    assert data_asa["is_emergency"] is True

    # 4. Test POST /theatre/api/or-schedule
    sched_time = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    resp_sched = client.post(
        "/theatre/api/or-schedule",
        json={
            "entry_id": entry_id,
            "or_room": "Cardiac OR",
            "scheduled_start_time": sched_time,
            "estimated_duration_minutes": 150,
        },
    )
    assert resp_sched.status_code == 200
    assert resp_sched.get_json()["or_room"] == "Cardiac OR"
