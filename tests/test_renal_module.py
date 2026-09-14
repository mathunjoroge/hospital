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

class TestRegressionGuards:
    """Ensure out-of-scope fields/columns are never present."""

    def test_dialysis_session_has_no_ktv_column(self, app):
        """DialysisSession must NOT have a kt_v field — decision #3 is pending."""
        with app.app_context():
            cols = {c.name for c in DialysisSession.__table__.columns}
            assert "kt_v" not in cols, "kt_v column must not exist until Nephrology Lead signs off"
            assert "ktv" not in cols
            assert "kt_over_v" not in cols

    def test_no_ktv_in_engine(self):
        """Engine source must not contain any Kt/V formula."""
        import inspect

        import departments.renal.engine as engine_mod
        source = inspect.getsource(engine_mod)
        assert "kt_v" not in source.lower()
        assert "ktv" not in source.lower()
        # Allow the word in comments only — check no callable uses it
        for name, obj in inspect.getmembers(engine_mod, inspect.isfunction):
            fn_src = inspect.getsource(obj)
            assert "kt_v" not in fn_src.lower(), f"{name}() contains kt_v"


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
                modality="crrt",   # lower-case normalised
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
                    modality="PD",   # Peritoneal — deferred per decision #1
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
        with app.app_context():
            with pytest.raises(LookupError):
                update_session_status(99999, "COMPLETED")

    def test_get_patient_sessions(self, app, nurse_user):
        from datetime import date

        from departments.renal.engine import create_session, get_patient_sessions
        with app.app_context():
            create_session("P007", nurse_user, "HD", date.today())
            create_session("P007", nurse_user, "CRRT", date.today())
            sessions = get_patient_sessions("P007")
            assert len(sessions) == 2

    def test_session_summary_no_ktv(self, app, nurse_user):
        """summary dict must never contain kt_v key."""
        from datetime import date

        from departments.renal.engine import create_session, session_summary
        with app.app_context():
            s = create_session("P008", nurse_user, "HD", date.today(), pre_weight=70.0, post_weight=67.5)
            summary = session_summary(s)
            assert "kt_v" not in summary
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
        client.post("/renal/sessions/P030", json={"modality": "HD", "session_date": "2026-09-14"})
        rv = client.get("/renal/sessions/P030")
        assert rv.status_code == 200
        data = rv.get_json()
        assert data["count"] >= 1
        assert data["patient_id"] == "P030"

    def test_update_status_patch(self, client, admin_user):
        rv = client.post("/renal/sessions/P040", json={"modality": "HD", "session_date": "2026-09-14"})
        session_id = rv.get_json()["session"]["id"]

        rv2 = client.patch(
            f"/renal/sessions/{session_id}/status",
            json={"status": "IN_PROGRESS"},
        )
        assert rv2.status_code == 200
        assert rv2.get_json()["session"]["status"] == "IN_PROGRESS"

    def test_complete_session(self, client, admin_user):
        rv = client.post("/renal/sessions/P041", json={"modality": "CRRT", "session_date": "2026-09-14"})
        session_id = rv.get_json()["session"]["id"]

        rv2 = client.patch(
            f"/renal/sessions/{session_id}/status",
            json={"status": "COMPLETED", "end_time": "2026-09-14T16:00:00"},
        )
        assert rv2.status_code == 200
        assert rv2.get_json()["session"]["status"] == "COMPLETED"

    def test_invalid_status_returns_422(self, client, admin_user):
        rv = client.post("/renal/sessions/P042", json={"modality": "HD", "session_date": "2026-09-14"})
        session_id = rv.get_json()["session"]["id"]
        rv2 = client.patch(f"/renal/sessions/{session_id}/status", json={"status": "CANCELLED"})
        assert rv2.status_code == 422

    def test_missing_status_returns_400(self, client, admin_user):
        rv = client.post("/renal/sessions/P043", json={"modality": "HD", "session_date": "2026-09-14"})
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
