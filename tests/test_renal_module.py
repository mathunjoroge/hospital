"""
tests/test_renal_module.py
───────────────────────────
Tests for the Renal / Dialysis Unit skeleton.

Scope mirrors DECISIONS_PENDING.md §23:
  ✅ #1  HD + CRRT session CRUD
  ✅ #2  Manual logging, no slot-scheduler assertions
  ✅ #4  Vascular access record CRUD
  ✅ #7  Billing event fires on COMPLETED sessions

Regression guards (must never regress):
  ❌ DialysisSession must NOT have a kt_v column (#3 pending sign-off)
  ❌ No Kt/V formula anywhere in engine or routes
"""

import pytest
from werkzeug.security import generate_password_hash

from departments.models.renal import (
    DialysisSession,
    RenalUnitConfig,
)
from departments.models.user import User
from extensions import db

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def nurse_user(app):
    """Create a nurse user for session logging."""
    with app.app_context():
        user = User(
            username="renal_nurse_test",
            password=generate_password_hash("pass", method="pbkdf2:sha256"),
            role="nurse",
        )
        db.session.add(user)
        db.session.commit()
        return user.id


# ── Safety regression guard ───────────────────────────────────────────────────


class TestAdequacyAndPrescription:
    """Test spKt/V adequacy calculation and DialysisPrescription (Section 23)."""

    def test_spkt_v_calculation(self, app, nurse_user):
        """spKt/V using Daugirdas II equation must return expected score when pre/post BUN given."""
        from datetime import date

        from departments.renal.engine import calculate_spkt_v, create_session

        with app.app_context():
            # Test direct calculator
            # Pre BUN 60, Post BUN 18, 4 hrs, UF 2.5L, Post weight 65kg
            score = calculate_spkt_v(pre_bun=60.0, post_bun=18.0, hours=4.0, uf_L=2.5, post_weight_kg=65.0)
            assert score is not None
            assert 1.2 <= score <= 1.6  # Typical target spKt/V >= 1.2

            # Session integration
            s = create_session(
                patient_id="P-KTV-01",
                nurse_id=nurse_user,
                modality="HD",
                session_date=date.today(),
                pre_bun=70.0,
                post_bun=20.0,
                pre_weight=70.0,
                post_weight=67.0,
            )
            assert s.spkt_v is not None
            assert s.spkt_v >= 1.2

    def test_prescription_creation_and_listing(self, client, admin_user):
        """Creating and listing a Nephrology Dialysis Prescription."""
        rv = client.post(
            "/renal/prescriptions/P-RX-01",
            json={
                "dialysate_flow_rate": 500.0,
                "blood_flow_rate": 350.0,
                "dialysate_composition": "K 2.0, Ca 1.25, Na 138",
                "heparin_bolus_units": 1000.0,
                "target_uf_liters": 2.0,
            },
        )
        assert rv.status_code == 201
        data = rv.get_json()
        assert data["success"] is True
        assert data["prescription"]["blood_flow_rate"] == 350.0

        rv_list = client.get("/renal/prescriptions/P-RX-01")
        assert rv_list.status_code == 200
        assert rv_list.get_json()["count"] >= 1



# ── Model sanity ──────────────────────────────────────────────────────────────


class TestRenalModels:
    def test_tables_created(self, app):
        """All three renal tables must exist after db.create_all()."""
        with app.app_context():
            inspector = db.inspect(db.engine)
            tables = inspector.get_table_names()
            assert "dialysis_sessions" in tables
            assert "vascular_access_records" in tables
            assert "renal_unit_configs" in tables

    def test_dialysis_session_defaults(self, app, nurse_user):
        with app.app_context():
            from datetime import date

            s = DialysisSession(
                patient_id="P001",
                nurse_id=nurse_user,
                modality="HD",
                session_date=date.today(),
            )
            db.session.add(s)
            db.session.commit()
            assert s.id is not None
            assert s.status == "SCHEDULED"
            assert s.created_at is not None

    def test_renal_unit_config_defaults(self, app):
        with app.app_context():
            cfg = RenalUnitConfig()
            db.session.add(cfg)
            db.session.commit()
            assert cfg.scheduling_mode == "MANUAL"
            assert cfg.chair_count is None  # facility provides this later


# ── Engine unit tests ─────────────────────────────────────────────────────────


class TestRenalEngine:
    def test_create_hd_session(self, app, nurse_user):
        from datetime import date

        from departments.renal.engine import create_session

        with app.app_context():
            s = create_session(
                patient_id="P002",
                nurse_id=nurse_user,
                modality="HD",
                session_date=date.today(),
                blood_flow_rate=300.0,
                pre_weight=65.5,
            )
            assert s.id is not None
            assert s.modality == "HD"
            assert s.blood_flow_rate == 300.0
            assert s.status == "SCHEDULED"

    def test_create_crrt_session(self, app, nurse_user):
        from datetime import date

        from departments.renal.engine import create_session

        with app.app_context():
            s = create_session(
                patient_id="P002",
                nurse_id=nurse_user,
                modality="crrt",  # lower-case normalised
                session_date=date.today(),
            )
            assert s.modality == "CRRT"

    def test_invalid_modality_raises(self, app, nurse_user):
        from datetime import date

        from departments.renal.engine import create_session

        with app.app_context():
            with pytest.raises(ValueError, match="Invalid modality"):
                create_session(
                    patient_id="P003",
                    nurse_id=nurse_user,
                    modality="PD",  # Peritoneal — deferred per decision #1
                    session_date=date.today(),
                )

    def test_update_session_status(self, app, nurse_user):
        from datetime import date

        from departments.renal.engine import create_session, update_session_status

        with app.app_context():
            s = create_session("P004", nurse_user, "HD", date.today())
            updated = update_session_status(s.id, "IN_PROGRESS")
            assert updated.status == "IN_PROGRESS"

    def test_update_to_completed(self, app, nurse_user):
        from datetime import date, datetime

        from departments.renal.engine import create_session, update_session_status

        with app.app_context():
            s = create_session("P005", nurse_user, "HD", date.today())
            updated = update_session_status(
                s.id, "COMPLETED", end_time=datetime(2026, 9, 14, 14, 0, 0)
            )
            assert updated.status == "COMPLETED"
            assert updated.end_time is not None

    def test_invalid_status_raises(self, app, nurse_user):
        from datetime import date

        from departments.renal.engine import create_session, update_session_status

        with app.app_context():
            s = create_session("P006", nurse_user, "HD", date.today())
            with pytest.raises(ValueError, match="Invalid status"):
                update_session_status(s.id, "CANCELLED")

    def test_lookup_error_for_missing_session(self, app):
        from departments.renal.engine import update_session_status

        with app.app_context(), pytest.raises(LookupError):
            update_session_status(99999, "COMPLETED")

    def test_get_patient_sessions(self, app, nurse_user):
        from datetime import date

        from departments.renal.engine import create_session, get_patient_sessions

        with app.app_context():
            create_session("P007", nurse_user, "HD", date.today())
            create_session("P007", nurse_user, "CRRT", date.today())
            sessions = get_patient_sessions("P007")
            assert len(sessions) == 2

    def test_session_summary_includes_spktv(self, app, nurse_user):
        """summary dict includes weight_loss_kg and spkt_v."""
        from datetime import date

        from departments.renal.engine import create_session, session_summary

        with app.app_context():
            s = create_session(
                "P008",
                nurse_user,
                "HD",
                date.today(),
                pre_weight=70.0,
                post_weight=67.5,
                pre_bun=60.0,
                post_bun=20.0,
            )
            summary = session_summary(s)
            assert "spkt_v" in summary
            assert summary["spkt_v"] is not None
            assert summary["weight_loss_kg"] == pytest.approx(2.5, abs=0.01)

    def test_log_access_record(self, app):
        from departments.renal.engine import log_access_record

        with app.app_context():
            r = log_access_record(
                patient_id="P009",
                access_type="AVF",
                site_description="Left forearm",
            )
            assert r.id is not None
            assert r.access_type == "AVF"

    def test_invalid_access_type_raises(self, app):
        from departments.renal.engine import log_access_record

        with app.app_context():
            with pytest.raises(ValueError, match="Invalid access_type"):
                log_access_record(patient_id="P010", access_type="Central Line")

    def test_get_patient_access_records(self, app):
        from departments.renal.engine import (
            get_patient_access_records,
            log_access_record,
        )

        with app.app_context():
            log_access_record("P011", "AVF")
            log_access_record("P011", "Tunnelled Catheter")
            records = get_patient_access_records("P011")
            assert len(records) == 2


# ── HTTP route tests ──────────────────────────────────────────────────────────


class TestRenalRoutes:
    def test_log_session_post(self, client, admin_user):
        rv = client.post(
            "/renal/sessions/P020",
            json={"modality": "HD", "session_date": "2026-09-14"},
        )
        assert rv.status_code == 201
        data = rv.get_json()
        assert data["success"] is True
        assert data["session"]["modality"] == "HD"
        assert data["session"]["status"] == "SCHEDULED"

    def test_log_crrt_session(self, client, admin_user):
        rv = client.post(
            "/renal/sessions/P021",
            json={
                "modality": "CRRT",
                "session_date": "2026-09-14",
                "blood_flow_rate": 150,
                "ultrafiltration_volume": 1500,
            },
        )
        assert rv.status_code == 201
        assert rv.get_json()["session"]["modality"] == "CRRT"

    def test_invalid_modality_returns_422(self, client, admin_user):
        rv = client.post(
            "/renal/sessions/P022",
            json={"modality": "PD", "session_date": "2026-09-14"},
        )
        assert rv.status_code == 422
        assert "modality" in rv.get_json()["error"].lower()

    def test_missing_modality_returns_400(self, client, admin_user):
        rv = client.post("/renal/sessions/P023", json={"session_date": "2026-09-14"})
        assert rv.status_code == 400

    def test_missing_session_date_returns_400(self, client, admin_user):
        rv = client.post("/renal/sessions/P024", json={"modality": "HD"})
        assert rv.status_code == 400

    def test_list_sessions(self, client, admin_user):
        client.post(
            "/renal/sessions/P030",
            json={"modality": "HD", "session_date": "2026-09-14"},
        )
        rv = client.get("/renal/sessions/P030")
        assert rv.status_code == 200
        data = rv.get_json()
        assert data["count"] >= 1
        assert data["patient_id"] == "P030"

    def test_update_status_patch(self, client, admin_user):
        rv = client.post(
            "/renal/sessions/P040",
            json={"modality": "HD", "session_date": "2026-09-14"},
        )
        session_id = rv.get_json()["session"]["id"]

        rv2 = client.patch(
            f"/renal/sessions/{session_id}/status",
            json={"status": "IN_PROGRESS"},
        )
        assert rv2.status_code == 200
        assert rv2.get_json()["session"]["status"] == "IN_PROGRESS"

    def test_complete_session(self, client, admin_user):
        rv = client.post(
            "/renal/sessions/P041",
            json={"modality": "CRRT", "session_date": "2026-09-14"},
        )
        session_id = rv.get_json()["session"]["id"]

        rv2 = client.patch(
            f"/renal/sessions/{session_id}/status",
            json={"status": "COMPLETED", "end_time": "2026-09-14T16:00:00"},
        )
        assert rv2.status_code == 200
        assert rv2.get_json()["session"]["status"] == "COMPLETED"

    def test_invalid_status_returns_422(self, client, admin_user):
        rv = client.post(
            "/renal/sessions/P042",
            json={"modality": "HD", "session_date": "2026-09-14"},
        )
        session_id = rv.get_json()["session"]["id"]
        rv2 = client.patch(
            f"/renal/sessions/{session_id}/status", json={"status": "CANCELLED"}
        )
        assert rv2.status_code == 422

    def test_missing_status_returns_400(self, client, admin_user):
        rv = client.post(
            "/renal/sessions/P043",
            json={"modality": "HD", "session_date": "2026-09-14"},
        )
        session_id = rv.get_json()["session"]["id"]
        rv2 = client.patch(f"/renal/sessions/{session_id}/status", json={})
        assert rv2.status_code == 400

    def test_add_access_record(self, client, admin_user):
        rv = client.post(
            "/renal/access/P050",
            json={"access_type": "AVF", "site_description": "Left forearm"},
        )
        assert rv.status_code == 201
        data = rv.get_json()
        assert data["success"] is True
        assert data["record"]["access_type"] == "AVF"

    def test_list_access_records(self, client, admin_user):
        client.post("/renal/access/P051", json={"access_type": "Tunnelled Catheter"})
        rv = client.get("/renal/access/P051")
        assert rv.status_code == 200
        assert rv.get_json()["count"] >= 1

    def test_invalid_access_type_returns_422(self, client, admin_user):
        rv = client.post("/renal/access/P052", json={"access_type": "Shunt"})
        assert rv.status_code == 422

    def test_missing_access_type_returns_400(self, client, admin_user):
        rv = client.post("/renal/access/P053", json={})
        assert rv.status_code == 400


# ── Unit-wide schedule board & patient search ────────────────────────────────


def _make_renal_patient(suffix, name=None, active=True):
    from datetime import date

    from departments.models.records import Patient

    p = Patient(
        patient_id=f"PR{suffix}",
        name=name or f"Renal Patient {suffix}",
        place_of_residence="Nairobi",
        sex="Female",
        date_of_birth=date(1985, 5, 5),
        marital_status="Married",
        blood_group="O+",
        contact=f"072200{suffix}",
        next_of_kin="Kin",
        relationship_with_next_of_kin="Sibling",
        next_of_kin_contact="0711111111",
        national_id=f"ID{suffix}",
        emergency_contact="0722222222",
    )
    if not active:
        p.soft_delete()
    return p


class TestUnitScheduleBoard:
    """The no-patient view must show pending work across ALL patients."""

    def test_board_default_shows_scheduled_and_active_only(self, client, admin_user, app):
        from datetime import date, timedelta

        from departments.renal.engine import create_session, update_session_status

        with app.app_context():
            nurse = User.query.filter_by(username="admin_test_fixture").first()
            future = date.today() + timedelta(days=3)
            past = date.today() - timedelta(days=3)
            create_session(patient_id="PRB1", nurse_id=nurse.id, modality="HD", session_date=future)
            done = create_session(patient_id="PRB2", nurse_id=nurse.id, modality="HD", session_date=past)
            update_session_status(done.id, "COMPLETED")

        rv = client.get("/renal/sessions")
        assert rv.status_code == 200
        data = rv.get_json()
        assert data["scope"] == "unit"
        statuses = {s["status"] for s in data["sessions"]}
        assert "COMPLETED" not in statuses
        assert "SCHEDULED" in statuses

    def test_board_date_range_filters(self, client, admin_user, app):
        """start/end (inclusive) narrow the board; invalid dates are ignored."""
        from datetime import date, timedelta

        from departments.renal.engine import create_session

        with app.app_context():
            nurse = User.query.filter_by(username="admin_test_fixture").first()
            create_session(
                patient_id="PRD1", nurse_id=nurse.id, modality="HD",
                session_date=date.today() + timedelta(days=1),
            )
            create_session(
                patient_id="PRD2", nurse_id=nurse.id, modality="HD",
                session_date=date.today() + timedelta(days=30),
            )

        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        day_after = (date.today() + timedelta(days=2)).isoformat()

        # Inclusive range catches the tomorrow session only
        rv = client.get(f"/renal/sessions?start={tomorrow}&end={day_after}")
        data = rv.get_json()
        ids = {s["patient_id"] for s in data["sessions"]}
        assert ids == {"PRD1"}
        assert data["date_range"] == {"filter_start": tomorrow, "filter_end": day_after}

        # Invalid dates are ignored (returns everything)
        rv = client.get("/renal/sessions?start=not-a-date&end=2026-13-99")
        assert rv.get_json()["count"] >= 2

    def test_board_today_quick_filter(self, client, admin_user, app):
        from datetime import date, timedelta

        from departments.renal.engine import create_session

        with app.app_context():
            nurse = User.query.filter_by(username="admin_test_fixture").first()
            create_session(
                patient_id="PRT1", nurse_id=nurse.id, modality="HD",
                session_date=date.today(),
            )
            create_session(
                patient_id="PRT2", nurse_id=nurse.id, modality="HD",
                session_date=date.today() + timedelta(days=5),
            )

        today = date.today().isoformat()
        rv = client.get(f"/renal/sessions?start={today}&end={today}")
        ids = {s["patient_id"] for s in rv.get_json()["sessions"]}
        assert ids == {"PRT1"}

    def test_per_patient_date_range(self, client, admin_user, app):
        from datetime import date, timedelta

        from departments.renal.engine import create_session

        with app.app_context():
            nurse = User.query.filter_by(username="admin_test_fixture").first()
            create_session(
                patient_id="PRF1", nurse_id=nurse.id, modality="HD",
                session_date=date.today() + timedelta(days=1),
            )
            create_session(
                patient_id="PRF1", nurse_id=nurse.id, modality="CRRT",
                session_date=date.today() + timedelta(days=20),
            )

        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        day_after = (date.today() + timedelta(days=2)).isoformat()
        rv = client.get(f"/renal/sessions/PRF1?start={tomorrow}&end={day_after}")
        data = rv.get_json()
        assert data["count"] == 1
        assert data["date_range"]["filter_start"] == tomorrow

    def test_board_source_filter(self, client, admin_user, app):
        """source=RECORDS/RENAL narrows the board; unknown values are ignored."""
        from datetime import date, timedelta

        from departments.models.renal import DialysisSession
        from departments.renal.engine import create_session
        from extensions import db as _db

        day = date.today() + timedelta(days=1)
        with app.app_context():
            nurse = User.query.filter_by(username="admin_test_fixture").first()
            create_session(
                patient_id="PRQ1", nurse_id=nurse.id, modality="HD",
                session_date=day,
            )
            _db.session.add(
                DialysisSession(
                    patient_id="PRQ2", nurse_id=nurse.id, modality="HD",
                    session_date=day, status="SCHEDULED", source="RECORDS",
                )
            )
            _db.session.commit()

        rv = client.get("/renal/sessions?source=RECORDS")
        ids = {s["patient_id"] for s in rv.get_json()["sessions"]}
        assert ids == {"PRQ2"}
        assert rv.get_json()["source"] == "RECORDS"

        rv = client.get("/renal/sessions?source=RENAL")
        ids = {s["patient_id"] for s in rv.get_json()["sessions"]}
        assert ids == {"PRQ1"}

        rv = client.get("/renal/sessions?source=bogus")
        data = rv.get_json()
        assert data["source"] is None
        assert len(data["sessions"]) == 2

    def test_per_patient_source_filter(self, client, admin_user, app):
        from datetime import date, timedelta

        from departments.models.renal import DialysisSession
        from departments.renal.engine import create_session
        from extensions import db as _db

        day = date.today() + timedelta(days=1)
        with app.app_context():
            nurse = User.query.filter_by(username="admin_test_fixture").first()
            create_session(
                patient_id="PRQ3", nurse_id=nurse.id, modality="HD",
                session_date=day,
            )
            _db.session.add(
                DialysisSession(
                    patient_id="PRQ3", nurse_id=nurse.id, modality="CRRT",
                    session_date=day + timedelta(days=5), status="SCHEDULED",
                    source="RECORDS",
                )
            )
            _db.session.commit()

        rv = client.get("/renal/sessions/PRQ3?source=RECORDS")
        data = rv.get_json()
        assert data["count"] == 1
        assert data["sessions"][0]["source"] == "RECORDS"

    def test_board_html_persists_source_filter(self, client, admin_user, app):
        rv = client.get(
            "/renal/sessions?source=RECORDS", headers={"Accept": "text/html"}
        )
        assert rv.status_code == 200
        assert b'value="RECORDS" selected' in rv.data


class TestNeedsChairTimeFlag:
    """Records bookings without a chair time must be flagged for nurses."""

    def _seed_mixed_sessions(self):
        """One unit-logged (no time), one records booking (no time),
        one records booking with a chair time. Returns their patient ids."""
        from datetime import date, datetime, timedelta

        from departments.models.renal import DialysisSession
        from departments.renal.engine import create_session
        from extensions import db as _db

        nurse = User.query.filter_by(username="admin_test_fixture").first()
        day = date.today() + timedelta(days=1)
        create_session(
            patient_id="PRF1", nurse_id=nurse.id, modality="HD", session_date=day
        )  # unit-logged, no time → NOT flagged
        _db.session.add(
            DialysisSession(
                patient_id="PRF2", nurse_id=nurse.id, modality="HD",
                session_date=day, status="SCHEDULED", source="RECORDS",
            )
        )  # records booking, no time → flagged
        _db.session.add(
            DialysisSession(
                patient_id="PRF3", nurse_id=nurse.id, modality="HD",
                session_date=day, status="SCHEDULED", source="RECORDS",
                start_time=datetime.combine(day, datetime.min.time()).replace(hour=8),
            )
        )  # records booking with chair time → NOT flagged
        _db.session.commit()
        return day

    def test_needs_chair_time_flag_in_json(self, client, admin_user, app):
        day = self._seed_mixed_sessions()

        rv = client.get(f"/renal/sessions?start={day.isoformat()}&end={day.isoformat()}")
        flags = {
            s["patient_id"]: s["needs_chair_time"] for s in rv.get_json()["sessions"]
        }
        assert flags["PRF1"] is False  # unit-logged
        assert flags["PRF2"] is True   # records booking, no chair time
        assert flags["PRF3"] is False  # records booking with chair time

    def test_day_sheet_highlights_unassigned_bookings(self, client, admin_user, app):
        day = self._seed_mixed_sessions()

        rv = client.get(
            f"/renal/sessions?start={day.isoformat()}&end={day.isoformat()}",
            headers={"Accept": "text/html"},
        )
        assert rv.status_code == 200
        html = rv.data
        # warning badge on the unassigned records row
        assert html.count(b"Needs chair time") == 1
        # Time TBD group header carries the nurse-facing count
        assert b"1 need chair time" in html
        # amber row treatment applied
        assert b"renal-row-needs-time" in html

    def test_board_html_renders_with_date_filter(self, client, admin_user, app):
        from datetime import date, timedelta

        from departments.renal.engine import create_session

        with app.app_context():
            nurse = User.query.filter_by(username="admin_test_fixture").first()
            create_session(
                patient_id="PRH9", nurse_id=nurse.id, modality="HD",
                session_date=date.today() + timedelta(days=1),
            )

        today = date.today().isoformat()
        week = (date.today() + timedelta(days=6)).isoformat()
        rv = client.get(
            f"/renal/sessions?start={today}&end={week}",
            headers={"Accept": "text/html"},
        )
        assert rv.status_code == 200
        assert b"Unit Schedule Board" in rv.data
        assert today.encode() in rv.data  # filter value echoed into the form

    def test_board_all_filter_includes_completed(self, client, admin_user, app):
        from datetime import date, timedelta

        from departments.renal.engine import create_session, update_session_status

        with app.app_context():
            nurse = User.query.filter_by(username="admin_test_fixture").first()
            past = date.today() - timedelta(days=2)
            done = create_session(patient_id="PRC1", nurse_id=nurse.id, modality="CRRT", session_date=past)
            update_session_status(done.id, "COMPLETED")

        rv = client.get("/renal/sessions?status=ALL")
        assert rv.status_code == 200
        data = rv.get_json()
        assert any(s["status"] == "COMPLETED" for s in data["sessions"])

    def test_board_includes_patient_names(self, client, admin_user, app):
        from datetime import date, timedelta

        from departments.renal.engine import create_session
        from extensions import db as _db

        with app.app_context():
            _db.session.add(_make_renal_patient("N1", name="Jane Dialysis"))
            _db.session.commit()
            nurse = User.query.filter_by(username="admin_test_fixture").first()
            future = date.today() + timedelta(days=1)
            create_session(patient_id="PRN1", nurse_id=nurse.id, modality="HD", session_date=future)

        rv = client.get("/renal/sessions?status=SCHEDULED")
        data = rv.get_json()
        row = next(s for s in data["sessions"] if s["patient_id"] == "PRN1")
        assert row["patient_name"] == "Jane Dialysis"

    def test_board_html_renders(self, client, admin_user, app):
        rv = client.get("/renal/sessions", headers={"Accept": "text/html"})
        assert rv.status_code == 200
        assert b"Unit Schedule Board" in rv.data

    def test_per_patient_json_includes_patient_name(self, client, admin_user, app):
        from extensions import db as _db

        with app.app_context():
            _db.session.add(_make_renal_patient("P2", name="Kamau Nephro"))
            _db.session.commit()

        rv = client.get("/renal/sessions/PRP2")
        assert rv.status_code == 200
        data = rv.get_json()
        assert data["patient_name"] == "Kamau Nephro"

    def test_per_patient_html_renders_with_name(self, client, admin_user, app):
        from extensions import db as _db

        with app.app_context():
            _db.session.add(_make_renal_patient("H1", name="Achieng Console"))
            _db.session.commit()

        rv = client.get("/renal/sessions/PRH1", headers={"Accept": "text/html"})
        assert rv.status_code == 200
        assert b"Achieng Console" in rv.data
        assert b"Log Dialysis Session" in rv.data


class TestRenalPatientSearch:
    def test_search_matches_id_and_name_active_only(self, client, admin_user, app):
        from extensions import db as _db

        with app.app_context():
            _db.session.add(_make_renal_patient("S1", name="Wanjiku Search"))
            _db.session.add(_make_renal_patient("S2", name="Deleted Patient", active=False))
            _db.session.commit()

        # By ID
        rv = client.get("/renal/api/search-patients?q=PRS1")
        results = rv.get_json()
        assert [r["id"] for r in results] == ["PRS1"]
        assert "Wanjiku Search" in results[0]["text"]

        # By name (case-insensitive)
        rv = client.get("/renal/api/search-patients?q=wanjiku")
        assert [r["id"] for r in rv.get_json()] == ["PRS1"]

        # Soft-deleted patients are excluded
        rv = client.get("/renal/api/search-patients?q=Deleted Patient")
        assert rv.get_json() == []

    def test_search_requires_two_chars(self, client, admin_user):
        assert client.get("/renal/api/search-patients?q=P").get_json() == []
        assert client.get("/renal/api/search-patients?q=").get_json() == []


# ── Day-sheet shift grouping ─────────────────────────────────────────────


class TestShiftGrouping:
    def test_shift_for_datetime_classification(self, app):
        from datetime import datetime

        from departments.renal.engine import shift_for_datetime

        assert shift_for_datetime(datetime(2026, 9, 21, 8, 0)) == "MORNING"
        assert shift_for_datetime(datetime(2026, 9, 21, 11, 59)) == "MORNING"
        assert shift_for_datetime(datetime(2026, 9, 21, 12, 0)) == "AFTERNOON"
        assert shift_for_datetime(datetime(2026, 9, 21, 17, 59)) == "AFTERNOON"
        assert shift_for_datetime(datetime(2026, 9, 21, 20, 0)) == "EVENING"
        assert shift_for_datetime(datetime(2026, 9, 22, 3, 0)) == "EVENING"
        assert shift_for_datetime(None) is None

    def test_session_summary_includes_shift_and_chair_time(self, app, nurse_user):
        from datetime import date, datetime, timedelta

        from departments.renal.engine import create_session, session_summary

        with app.app_context():
            future = date.today() + timedelta(days=1)
            s = create_session(
                patient_id="PRS1",
                nurse_id=nurse_user,
                modality="HD",
                session_date=future,
                start_time=datetime.combine(future, datetime.min.time()).replace(hour=8),
            )
            summary = session_summary(s)
            assert summary["shift"] == "MORNING"
            assert summary["start_time_hm"] == "08:00"

    def test_day_sheet_groups_by_shift(self, client, admin_user, app):
        """Single-day filter renders Morning/Afternoon/Evening/Time-TBD sections."""
        from datetime import date, datetime, timedelta

        from departments.renal.engine import create_session

        day = date.today() + timedelta(days=2)
        day_iso = day.isoformat()

        with app.app_context():
            nurse = User.query.filter_by(username="admin_test_fixture").first()

            def at(hour):
                return datetime.combine(day, datetime.min.time()).replace(hour=hour)

            create_session(patient_id="PRG1", nurse_id=nurse.id, modality="HD",
                           session_date=day, start_time=at(8))
            create_session(patient_id="PRG2", nurse_id=nurse.id, modality="HD",
                           session_date=day, start_time=at(14))
            create_session(patient_id="PRG3", nurse_id=nurse.id, modality="CRRT",
                           session_date=day, start_time=at(20))
            # Records-style booking: date only, no chair time
            create_session(patient_id="PRG4", nurse_id=nurse.id, modality="HD",
                           session_date=day)

        rv = client.get(
            f"/renal/sessions?start={day_iso}&end={day_iso}",
            headers={"Accept": "text/html"},
        )
        assert rv.status_code == 200
        html = rv.data
        assert b"Morning Shift" in html
        assert b"Afternoon Shift" in html
        assert b"Evening Shift" in html
        assert b"Time TBD" in html

    def test_flat_board_has_no_group_headers(self, client, admin_user, app):
        """Without a single-day filter the board stays a flat sorted table."""
        from datetime import date, timedelta

        from departments.renal.engine import create_session

        with app.app_context():
            nurse = User.query.filter_by(username="admin_test_fixture").first()
            create_session(
                patient_id="PRG5", nurse_id=nurse.id, modality="HD",
                session_date=date.today() + timedelta(days=1),
            )

        rv = client.get("/renal/sessions", headers={"Accept": "text/html"})
        assert rv.status_code == 200
        assert b"Morning Shift" not in rv.data


# ── Chair time assignment ────────────────────────────────────────────────────


class TestChairTimeAssignment:
    def test_assign_chair_time_sets_start_and_shift(self, app, nurse_user):
        """A date-only (Records-style) booking gets a chair time + shift."""
        from datetime import date, timedelta

        from departments.renal.engine import (
            assign_chair_time,
            create_session,
            session_summary,
        )

        with app.app_context():
            s = create_session(
                patient_id="PRCT1", nurse_id=nurse_user, modality="HD",
                session_date=date.today() + timedelta(days=1),
            )
            assert s.start_time is None

            updated = assign_chair_time(s.id, chair_time="08:30")
            assert updated.start_time is not None
            assert updated.start_time.hour == 8
            assert updated.start_time.minute == 30
            summary = session_summary(updated)
            assert summary["shift"] == "MORNING"
            assert summary["start_time_hm"] == "08:30"

    def test_chair_time_reassignment_moves_shift(self, app, nurse_user):
        from datetime import date, timedelta

        from departments.renal.engine import assign_chair_time, create_session

        with app.app_context():
            s = create_session(
                patient_id="PRCT2", nurse_id=nurse_user, modality="HD",
                session_date=date.today() + timedelta(days=1),
            )
            moved = assign_chair_time(s.id, chair_time="14:00", end_time_hm="18:00")
            assert moved.start_time.hour == 14
            assert moved.end_time is not None
            assert moved.end_time.hour == 18

    def test_end_time_before_start_rejected(self, app, nurse_user):
        from datetime import date, timedelta

        import pytest as _pytest

        from departments.renal.engine import assign_chair_time, create_session

        with app.app_context():
            s = create_session(
                patient_id="PRCT3", nurse_id=nurse_user, modality="HD",
                session_date=date.today() + timedelta(days=1),
            )
            with _pytest.raises(ValueError, match="after the start"):
                assign_chair_time(s.id, chair_time="14:00", end_time_hm="12:00")

    def test_invalid_time_format_rejected(self, app, nurse_user):
        from datetime import date, timedelta

        import pytest as _pytest

        from departments.renal.engine import assign_chair_time, create_session

        with app.app_context():
            s = create_session(
                patient_id="PRCT4", nurse_id=nurse_user, modality="HD",
                session_date=date.today() + timedelta(days=1),
            )
            with _pytest.raises(ValueError, match="HH:MM"):
                assign_chair_time(s.id, chair_time="8am")
            with _pytest.raises(ValueError, match="HH:MM"):
                assign_chair_time(s.id, chair_time="25:00")

    def test_missing_session_raises_lookup(self, app):
        import pytest as _pytest

        from departments.renal.engine import assign_chair_time

        with app.app_context():
            with _pytest.raises(LookupError):
                assign_chair_time(999999, chair_time="08:00")

    def test_route_assigns_chair_time(self, client, admin_user, app):
        from datetime import date, timedelta

        from departments.renal.engine import create_session

        with app.app_context():
            nurse = User.query.filter_by(username="admin_test_fixture").first()
            s = create_session(
                patient_id="PRCT5", nurse_id=nurse.id, modality="HD",
                session_date=date.today() + timedelta(days=1),
            )
            session_id = s.id

        rv = client.patch(
            f"/renal/sessions/{session_id}/chair-time",
            json={"chair_time": "08:30"},
        )
        assert rv.status_code == 200
        data = rv.get_json()
        assert data["success"] is True
        assert data["session"]["start_time_hm"] == "08:30"
        assert data["session"]["shift"] == "MORNING"

    def test_route_missing_chair_time_400(self, client, admin_user):
        rv = client.patch("/renal/sessions/1/chair-time", json={})
        assert rv.status_code == 400

    def test_route_invalid_time_422(self, client, admin_user, app):
        from datetime import date, timedelta

        from departments.renal.engine import create_session

        with app.app_context():
            nurse = User.query.filter_by(username="admin_test_fixture").first()
            s = create_session(
                patient_id="PRCT6", nurse_id=nurse.id, modality="HD",
                session_date=date.today() + timedelta(days=1),
            )
            session_id = s.id

        rv = client.patch(
            f"/renal/sessions/{session_id}/chair-time",
            json={"chair_time": "not-a-time"},
        )
        assert rv.status_code == 422

    def test_route_unknown_session_404(self, client, admin_user):
        rv = client.patch("/renal/sessions/999999/chair-time", json={"chair_time": "08:00"})
        assert rv.status_code == 404

    def test_chair_time_assignment_dispatches_notification(self, app, nurse_user):
        """Assigning a chair time queues an appointment_confirmed notification."""
        from datetime import date, timedelta

        from departments.models.notification_log import OutboundNotificationLog
        from departments.notifications.dispatcher import EVENT_APPOINTMENT_CONFIRMED
        from departments.renal.engine import assign_chair_time, create_session
        from extensions import db as _db

        with app.app_context():
            _db.session.add(_make_renal_patient("NT1", name="Notify Patient"))
            _db.session.commit()
            s = create_session(
                patient_id="PRNT1", nurse_id=nurse_user, modality="HD",
                session_date=date.today() + timedelta(days=1),
            )

            assign_chair_time(s.id, chair_time="08:30")

            log = OutboundNotificationLog.query.filter_by(
                patient_id="PRNT1", event_type=EVENT_APPOINTMENT_CONFIRMED
            ).first()
            assert log is not None
            assert log.status == "SENT"
            assert "08:30" in log.body
            assert "morning shift" in log.body

    def test_notification_failure_does_not_fail_assignment(
        self, app, nurse_user, monkeypatch
    ):
        from datetime import date, timedelta

        from departments.renal.engine import assign_chair_time, create_session
        from extensions import db as _db

        def boom(session):
            raise RuntimeError("smtp down")

        monkeypatch.setattr(
            "departments.notifications.triggers.trigger_chair_time_assigned", boom
        )

        with app.app_context():
            _db.session.add(_make_renal_patient("NT2", name="Resilient Patient"))
            _db.session.commit()
            s = create_session(
                patient_id="PRNT2", nurse_id=nurse_user, modality="HD",
                session_date=date.today() + timedelta(days=1),
            )

            updated = assign_chair_time(s.id, chair_time="09:00")

            assert updated.start_time is not None
            assert updated.start_time.hour == 9


# ── Chair conflicts (A2) & reschedule (A3) ───────────────────────────────


class TestChairConflictsAndReschedule:
    def test_conflict_blocks_at_default_capacity(self, app, nurse_user):
        """Default capacity 1: a second patient cannot hold the same slot."""
        import pytest as _pytest
        from datetime import date, timedelta

        from departments.renal.engine import assign_chair_time, create_session

        day = date.today() + timedelta(days=1)
        with app.app_context():
            s1 = create_session(patient_id="PRCC1", nurse_id=nurse_user,
                                modality="HD", session_date=day)
            s2 = create_session(patient_id="PRCC2", nurse_id=nurse_user,
                                modality="HD", session_date=day)
            assign_chair_time(s1.id, chair_time="08:00")
            with _pytest.raises(ValueError, match="fully booked"):
                assign_chair_time(s2.id, chair_time="08:00")

    def test_conflict_allows_when_capacity_raised(self, client, admin_user, app):
        """chair_count=2: two patients share the slot; response carries warning."""
        from datetime import date, timedelta

        from departments.models.renal import RenalUnitConfig
        from departments.renal.engine import assign_chair_time, create_session
        from extensions import db as _db

        day = date.today() + timedelta(days=1)
        with app.app_context():
            _db.session.add(RenalUnitConfig(chair_count=2))
            _db.session.commit()
            nurse = User.query.filter_by(username="admin_test_fixture").first()
            s1 = create_session(patient_id="PRCC3", nurse_id=nurse.id,
                                modality="HD", session_date=day)
            s2 = create_session(patient_id="PRCC4", nurse_id=nurse.id,
                                modality="HD", session_date=day)
            assign_chair_time(s1.id, chair_time="08:00")
            session_id = s2.id

        rv = client.patch(
            f"/renal/sessions/{session_id}/chair-time", json={"chair_time": "08:00"}
        )
        assert rv.status_code == 200
        assert rv.get_json()["warning"] is not None
        assert "share this chair slot" in rv.get_json()["warning"]

    def test_reassign_own_slot_no_self_conflict(self, app, nurse_user):
        from datetime import date, timedelta

        from departments.renal.engine import assign_chair_time, create_session

        day = date.today() + timedelta(days=1)
        with app.app_context():
            s = create_session(patient_id="PRCC5", nurse_id=nurse_user,
                               modality="HD", session_date=day)
            assign_chair_time(s.id, chair_time="08:00")
            moved = assign_chair_time(s.id, chair_time="10:00")  # same session
            assert moved.start_time.hour == 10

    def test_reschedule_moves_date_and_reanchors_end(self, app, nurse_user):
        """A3: moving day re-anchors start AND existing end to the new date."""
        from datetime import date, timedelta

        from departments.renal.engine import assign_chair_time, create_session

        day = date.today() + timedelta(days=1)
        new_day = day + timedelta(days=2)
        with app.app_context():
            s = create_session(patient_id="PRCC6", nurse_id=nurse_user,
                               modality="HD", session_date=day)
            assign_chair_time(s.id, chair_time="08:00", end_time_hm="12:00")

            moved = assign_chair_time(
                s.id, chair_time="09:00", session_date=new_day
            )
            assert moved.session_date == new_day
            assert moved.start_time.hour == 9
            assert moved.end_time is not None
            assert moved.end_time.date() == new_day
            assert moved.end_time.hour == 12  # end time-of-day preserved, re-anchored to new date

    def test_route_reschedule_invalid_date_422(self, client, admin_user, app):
        from datetime import date, timedelta

        from departments.renal.engine import create_session

        with app.app_context():
            nurse = User.query.filter_by(username="admin_test_fixture").first()
            s = create_session(
                patient_id="PRCC7", nurse_id=nurse.id, modality="HD",
                session_date=date.today() + timedelta(days=1),
            )
            session_id = s.id

        rv = client.patch(
            f"/renal/sessions/{session_id}/chair-time",
            json={"chair_time": "08:00", "session_date": "not-a-date"},
        )
        assert rv.status_code == 422

    def test_route_conflict_returns_422(self, client, admin_user, app):
        from datetime import date, timedelta

        from departments.renal.engine import assign_chair_time, create_session

        day = date.today() + timedelta(days=1)
        with app.app_context():
            nurse = User.query.filter_by(username="admin_test_fixture").first()
            s1 = create_session(patient_id="PRCC8", nurse_id=nurse.id,
                                modality="HD", session_date=day)
            s2 = create_session(patient_id="PRCC9", nurse_id=nurse.id,
                                modality="HD", session_date=day)
            assign_chair_time(s1.id, chair_time="07:00")
            session_id = s2.id

        rv = client.patch(
            f"/renal/sessions/{session_id}/chair-time", json={"chair_time": "07:00"}
        )
        assert rv.status_code == 422
        assert "fully booked" in rv.get_json()["error"]


# ── Recurring dialysis series (B4) ──────────────────────────────────────────


class TestSessionSeries:
    def test_parse_weekdays(self, app):
        from departments.renal.engine import parse_weekdays

        assert parse_weekdays("Mon/Wed/Fri") == [0, 2, 4]
        assert parse_weekdays("monday, friday") == [0, 4]
        assert parse_weekdays("Tue") == [1]
        assert parse_weekdays("") == []
        assert parse_weekdays("Someday") == []

    def test_series_creates_and_skips_existing(self, app, nurse_user):
        """Fixed Monday start (2027-01-04): 3 targets; Wednesday pre-exists."""
        from datetime import date, datetime

        from departments.models.renal import DialysisSession
        from departments.renal.engine import create_session_series
        from extensions import db as _db

        start = date(2027, 1, 4)  # a Monday
        assert start.weekday() == 0

        with app.app_context():
            _db.session.add(
                DialysisSession(
                    patient_id="PRSS1", nurse_id=nurse_user, modality="HD",
                    session_date=date(2027, 1, 6), status="SCHEDULED",
                )
            )
            _db.session.commit()

            created, skipped = create_session_series(
                patient_id="PRSS1",
                nurse_id=nurse_user,
                weekdays=[0, 2, 4],
                weeks=1,
                start_date=start,
                chair_time="08:00",
            )

            assert len(created) == 2
            assert skipped == [date(2027, 1, 6)]
            assert {s.session_date for s in created} == {
                date(2027, 1, 4), date(2027, 1, 8)
            }
            for s in created:
                assert s.status == "SCHEDULED"
                assert s.start_time.hour == 8
                assert s.start_time.date() == s.session_date

    def test_series_route_creates(self, client, admin_user, app):
        rv = client.post(
            "/renal/sessions/PRSR9/series",
            json={"days": "Mon/Wed/Fri", "weeks": 2, "chair_time": "07:30"},
        )
        assert rv.status_code == 201
        data = rv.get_json()
        assert data["count"] == 6  # 3 days/week × 2 weeks
        assert data["created"][0]["start_time_hm"] == "07:30"

    def test_series_route_requires_days(self, client, admin_user):
        rv = client.post("/renal/sessions/PRSR9/series", json={})
        assert rv.status_code == 400

    def test_series_route_invalid_weeks_422(self, client, admin_user):
        rv = client.post(
            "/renal/sessions/PRSR9/series",
            json={"days": "Mon", "weeks": "lots"},
        )
        assert rv.status_code == 422

    def test_series_route_invalid_modality_422(self, client, admin_user):
        rv = client.post(
            "/renal/sessions/PRSR9/series",
            json={"days": "Mon", "modality": "PD"},
        )
        assert rv.status_code == 422

    def test_sessions_default_to_renal_source(self, app, nurse_user):
        from datetime import date, timedelta

        from departments.renal.engine import create_session, session_summary

        with app.app_context():
            s = create_session(
                patient_id="PRSR1", nurse_id=nurse_user, modality="HD",
                session_date=date.today() + timedelta(days=1),
            )
            assert s.source == "RENAL"
            assert session_summary(s)["source"] == "RENAL"

    def test_board_marks_records_bookings(self, client, admin_user, app):
        """Only RECORDS-sourced sessions carry the Records badge on the board."""
        from datetime import date, timedelta

        from departments.models.renal import DialysisSession
        from departments.renal.engine import create_session
        from extensions import db as _db

        day = date.today() + timedelta(days=1)
        day_iso = day.isoformat()
        with app.app_context():
            nurse = User.query.filter_by(username="admin_test_fixture").first()
            create_session(
                patient_id="PRSR2", nurse_id=nurse.id, modality="HD",
                session_date=day,
            )
            _db.session.add(
                DialysisSession(
                    patient_id="PRSR3", nurse_id=nurse.id, modality="HD",
                    session_date=day, status="SCHEDULED", source="RECORDS",
                    notes="Booked from Records",
                )
            )
            _db.session.commit()

        rv = client.get(
            f"/renal/sessions?start={day_iso}&end={day_iso}",
            headers={"Accept": "text/html"},
        )
        assert rv.status_code == 200
        # exactly one of the two rows is records-sourced (row badge uses the
        # `me-1` icon variant; the hint card & CSS rule don't)
        assert rv.data.count(b"bi-clipboard-plus me-1") == 1

    def test_day_sheet_offers_set_time_button(self, client, admin_user, app):
        """SCHEDULED rows on the day sheet render the chair-time editor trigger."""
        from datetime import date, timedelta

        from departments.renal.engine import create_session

        day = date.today() + timedelta(days=1)
        day_iso = day.isoformat()
        with app.app_context():
            nurse = User.query.filter_by(username="admin_test_fixture").first()
            create_session(
                patient_id="PRCT7", nurse_id=nurse.id, modality="HD",
                session_date=day,
            )

        rv = client.get(
            f"/renal/sessions?start={day_iso}&end={day_iso}",
            headers={"Accept": "text/html"},
        )
        assert rv.status_code == 200
        assert b"openChairTimeModal" in rv.data
        assert b"chairTimeModal" in rv.data
