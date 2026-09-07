"""
tests/test_break_glass.py
──────────────────────────
Phase D — Break-glass Emergency Access Test Suite

Tests:
  1. invoke_break_glass — creates a valid grant with all required fields
  2. grant is_valid / expiry — time-box enforced correctly
  3. check_break_glass — returns active grant, ignores expired/inactive ones
  4. break_glass_required decorator — normal RBAC path still works
  5. break_glass_required decorator — override path granted for eligible role
  6. break_glass_required decorator — ineligible role without grant → 403
  7. POST /emergency/break-glass/invoke — success (201) for eligible role
  8. POST /emergency/break-glass/invoke — 400 when reason is missing
  9. POST /emergency/break-glass/invoke — 403 for ineligible role
 10. GET  /emergency/break-glass/status — lists active grants
 11. POST /emergency/break-glass/<id>/revoke — revokes own grant
 12. GET  /admin/break-glass — admin audit view (HTML and JSON)
 13. expire_stale_grants — marks expired-by-time grants inactive
 14. supervisor notification recorded on break-glass invocation
"""

from datetime import datetime, timedelta, timezone

import pytest
from werkzeug.security import generate_password_hash

from departments.emergency.break_glass import (
    check_break_glass,
    expire_stale_grants,
    invoke_break_glass,
)
from departments.models.break_glass import BreakGlassAccessLog
from departments.models.records import Patient
from departments.models.user import User
from extensions import db

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_user(role, suffix=""):
    return User(
        username=f"{role}_bg_test{suffix}",
        password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
        role=role,
    )


@pytest.fixture
def doctor_user(app):
    with app.app_context():
        u = _make_user("medicine", "_doc")
        db.session.add(u)
        db.session.commit()
        yield u


@pytest.fixture
def nursing_user(app):
    with app.app_context():
        u = _make_user("nursing", "_nurse")
        db.session.add(u)
        db.session.commit()
        yield u


@pytest.fixture
def admin_user(app):
    with app.app_context():
        u = _make_user("admin", "_adm")
        db.session.add(u)
        db.session.commit()
        yield u


@pytest.fixture
def records_user(app):
    """Records clerk — NOT eligible to invoke break-glass."""
    with app.app_context():
        u = _make_user("records", "_rec")
        db.session.add(u)
        db.session.commit()
        yield u


@pytest.fixture
def sample_patient(app):
    with app.app_context():
        p = Patient(
            patient_id="P-BG-001",
            name="Test Patient BG",
            sex="Male",
            date_of_birth=datetime(1990, 1, 1).date(),
            marital_status="Single",
            contact="0700000001",
            place_of_residence="Nairobi",
            national_id="NID-BG-001",
            next_of_kin="Kin Name",
            relationship_with_next_of_kin="Sibling",
            next_of_kin_contact="0700000002",
            emergency_contact="0700000002",
        )
        db.session.add(p)
        db.session.commit()
        yield p


# ─────────────────────────────────────────────────────────────────────────────
# 1. invoke_break_glass creates a valid grant
# ─────────────────────────────────────────────────────────────────────────────


def test_invoke_break_glass_creates_grant(app, doctor_user, sample_patient):
    """invoke_break_glass() returns a BreakGlassAccessLog with is_active=True."""
    with app.app_context():
        grant = invoke_break_glass(
            reason="Patient unconscious, need immediate history",
            patient_id=sample_patient.patient_id,
            user=doctor_user,
        )
        assert grant.id is not None
        assert grant.is_active is True
        assert grant.patient_id == sample_patient.patient_id
        assert grant.reason == "Patient unconscious, need immediate history"
        expires = (
            grant.expires_at
            if grant.expires_at.tzinfo is not None
            else grant.expires_at.replace(tzinfo=timezone.utc)
        )
        assert expires > datetime.now(timezone.utc)
        assert grant.supervisor_notified is True


# ─────────────────────────────────────────────────────────────────────────────
# 2. is_valid() respects expiry
# ─────────────────────────────────────────────────────────────────────────────


def test_break_glass_grant_validity(app, doctor_user):
    """is_valid() returns False for expired or revoked grants."""
    with app.app_context():
        now = datetime.now(timezone.utc)
        expired_grant = BreakGlassAccessLog(
            user_id=doctor_user.id,
            username=doctor_user.username,
            user_role=doctor_user.role,
            reason="Test expired",
            invoked_at=now - timedelta(hours=6),
            expires_at=now - timedelta(hours=2),
            is_active=True,
        )
        db.session.add(expired_grant)
        db.session.commit()
        assert expired_grant.is_valid() is False

        revoked_grant = BreakGlassAccessLog(
            user_id=doctor_user.id,
            username=doctor_user.username,
            user_role=doctor_user.role,
            reason="Test revoked",
            invoked_at=now - timedelta(minutes=30),
            expires_at=now + timedelta(hours=2),
            is_active=False,
        )
        db.session.add(revoked_grant)
        db.session.commit()
        assert revoked_grant.is_valid() is False


# ─────────────────────────────────────────────────────────────────────────────
# 3. check_break_glass finds active grant, ignores expired/inactive
# ─────────────────────────────────────────────────────────────────────────────


def test_check_break_glass(app, doctor_user):
    """check_break_glass() returns valid grant and ignores expired ones."""
    with app.app_context():
        now = datetime.now(timezone.utc)

        # Active grant
        active = BreakGlassAccessLog(
            user_id=doctor_user.id,
            username=doctor_user.username,
            user_role=doctor_user.role,
            reason="Active grant",
            invoked_at=now,
            expires_at=now + timedelta(hours=4),
            is_active=True,
        )
        # Expired grant
        expired = BreakGlassAccessLog(
            user_id=doctor_user.id,
            username=doctor_user.username,
            user_role=doctor_user.role,
            reason="Expired grant",
            invoked_at=now - timedelta(hours=5),
            expires_at=now - timedelta(hours=1),
            is_active=True,
        )
        db.session.add_all([active, expired])
        db.session.commit()

        found = check_break_glass(doctor_user.id)
        assert found is not None
        assert found.id == active.id


# ─────────────────────────────────────────────────────────────────────────────
# 4–6. POST /emergency/break-glass/invoke — HTTP endpoint tests
# ─────────────────────────────────────────────────────────────────────────────


def test_invoke_endpoint_success(client, app, doctor_user, sample_patient):
    """POST /emergency/break-glass/invoke succeeds for eligible role."""
    client.post(
        "/login", data={"username": doctor_user.username, "password": "Password123!"}
    )
    resp = client.post(
        "/emergency/break-glass/invoke",
        json={
            "reason": "Emergency: patient collapsed",
            "patient_id": sample_patient.patient_id,
        },
    )
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["grant"]["is_active"] is True
    assert data["grant"]["patient_id"] == sample_patient.patient_id


def test_invoke_endpoint_missing_reason(client, doctor_user):
    """POST /emergency/break-glass/invoke returns 400 when reason is absent."""
    client.post(
        "/login", data={"username": doctor_user.username, "password": "Password123!"}
    )
    resp = client.post("/emergency/break-glass/invoke", json={"patient_id": "P-BG-001"})
    assert resp.status_code == 400
    assert "reason" in resp.get_json()["error"].lower()


def test_invoke_endpoint_ineligible_role(client, records_user):
    """POST /emergency/break-glass/invoke returns 403 for ineligible role."""
    client.post(
        "/login", data={"username": records_user.username, "password": "Password123!"}
    )
    resp = client.post(
        "/emergency/break-glass/invoke",
        json={"reason": "I shouldn't be able to do this"},
    )
    assert resp.status_code == 403


# ─────────────────────────────────────────────────────────────────────────────
# 7. GET /emergency/break-glass/status
# ─────────────────────────────────────────────────────────────────────────────


def test_status_endpoint(client, app, doctor_user):
    """GET /emergency/break-glass/status returns the user's active grants."""
    client.post(
        "/login", data={"username": doctor_user.username, "password": "Password123!"}
    )
    # Invoke one first
    client.post(
        "/emergency/break-glass/invoke",
        json={"reason": "Status test reason"},
    )
    resp = client.get("/emergency/break-glass/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "active_grants" in data
    assert len(data["active_grants"]) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 8. POST /emergency/break-glass/<id>/revoke
# ─────────────────────────────────────────────────────────────────────────────


def test_revoke_endpoint(client, app, doctor_user):
    """POST /emergency/break-glass/<id>/revoke deactivates the grant."""
    client.post(
        "/login", data={"username": doctor_user.username, "password": "Password123!"}
    )
    inv_resp = client.post(
        "/emergency/break-glass/invoke",
        json={"reason": "Revoke test"},
    )
    grant_id = inv_resp.get_json()["grant"]["id"]

    revoke_resp = client.post(f"/emergency/break-glass/{grant_id}/revoke")
    assert revoke_resp.status_code == 200

    with app.app_context():
        grant = db.session.get(BreakGlassAccessLog, grant_id)
        assert grant.is_active is False


# ─────────────────────────────────────────────────────────────────────────────
# 9. GET /admin/break-glass (HTML + JSON)
# ─────────────────────────────────────────────────────────────────────────────


def test_admin_audit_view(client, admin_user):
    """GET /admin/break-glass is admin-only and returns audit data."""
    client.post(
        "/login", data={"username": admin_user.username, "password": "Password123!"}
    )
    resp = client.get("/admin/break-glass")
    assert resp.status_code == 200
    assert b"Break-glass" in resp.data


def test_admin_audit_json(client, admin_user):
    """GET /admin/break-glass?format=json returns JSON with break_glass_events key."""
    client.post(
        "/login", data={"username": admin_user.username, "password": "Password123!"}
    )
    resp = client.get("/admin/break-glass?format=json")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "break_glass_events" in data


def test_admin_audit_rbac(client, records_user):
    """Non-admin cannot access /admin/break-glass."""
    client.post(
        "/login", data={"username": records_user.username, "password": "Password123!"}
    )
    resp = client.get("/admin/break-glass")
    assert resp.status_code in (403, 302)


# ─────────────────────────────────────────────────────────────────────────────
# 10. expire_stale_grants
# ─────────────────────────────────────────────────────────────────────────────


def test_expire_stale_grants(app, doctor_user):
    """expire_stale_grants() deactivates grants whose expires_at is in the past."""
    with app.app_context():
        now = datetime.now(timezone.utc)
        stale = BreakGlassAccessLog(
            user_id=doctor_user.id,
            username=doctor_user.username,
            user_role=doctor_user.role,
            reason="Stale grant",
            invoked_at=now - timedelta(hours=5),
            expires_at=now - timedelta(seconds=1),
            is_active=True,
        )
        db.session.add(stale)
        db.session.commit()
        stale_id = stale.id

        count = expire_stale_grants()
        assert count >= 1

        refreshed = db.session.get(BreakGlassAccessLog, stale_id)
        assert refreshed.is_active is False


# ─────────────────────────────────────────────────────────────────────────────
# 11. ValueError on empty reason
# ─────────────────────────────────────────────────────────────────────────────


def test_invoke_empty_reason_raises(app, doctor_user):
    """invoke_break_glass raises ValueError for empty/blank reason."""
    with app.app_context():
        with pytest.raises(ValueError, match="reason"):
            invoke_break_glass(reason="   ", user=doctor_user)


# ─────────────────────────────────────────────────────────────────────────────
# 12. PermissionError for ineligible role
# ─────────────────────────────────────────────────────────────────────────────


def test_invoke_ineligible_role_raises(app, records_user):
    """invoke_break_glass raises PermissionError for non-eligible roles."""
    with app.app_context():
        with pytest.raises(PermissionError, match="authorised"):
            invoke_break_glass(
                reason="Attempting unauthorised override", user=records_user
            )


# ─────────────────────────────────────────────────────────────────────────────
# 13. AUDIT trail — BREAK_GLASS_OVERRIDE action recorded
# ─────────────────────────────────────────────────────────────────────────────


def test_break_glass_audit_trail(app, doctor_user, sample_patient):
    """invoking break-glass writes a BREAK_GLASS_OVERRIDE audit entry."""
    from departments.models.compliance import AuditLog

    with app.app_context():
        invoke_break_glass(
            reason="Audit trail test",
            patient_id=sample_patient.patient_id,
            user=doctor_user,
        )
        entry = (
            AuditLog.query.filter_by(action="BREAK_GLASS_OVERRIDE")
            .order_by(AuditLog.id.desc())
            .first()
        )
        assert entry is not None
        assert entry.action == "BREAK_GLASS_OVERRIDE"
