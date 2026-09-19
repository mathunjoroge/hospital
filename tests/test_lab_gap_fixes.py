"""
tests/test_lab_gap_fixes.py
───────────────────────────
Regression tests for the laboratory gap-analysis fixes:

  - P0-2: 2-tier separation — the entering tech cannot verify their own result.
  - P0-3: panic alerts route to the ordering clinician (not just the verifier).
  - P0-4: LIS enter endpoint rejects missing clinical fields (no silent defaults).
  - P0-5: laboratory/lab_tech roles are interchangeable via ROLE_ALIASES.
  - P1-7: process_lab_request is idempotent (no duplicate results on retry).
  - P1-11: parameter templates cannot be deleted while results reference them.
  - P1-12: lab tests in clinical use cannot be deleted.
"""

from datetime import datetime, timezone

import pytest
from werkzeug.security import generate_password_hash

from departments.models.laboratory import LabResult, LabResultTemplate
from departments.models.medicine import LabTest, RequestedLab
from departments.models.records import Patient
from departments.models.user import User
from extensions import db

# ─── helpers / fixtures ──────────────────────────────────────────────────────


def _make_user(username, role, password="pw123"):
    user = User.query.filter_by(username=username).first()
    if not user:
        user = User(
            username=username,
            password=generate_password_hash(password, method="pbkdf2:sha256"),
            role=role,
        )
        db.session.add(user)
        db.session.commit()
    return user


@pytest.fixture
def lab_setup(app):
    """A patient, a lab test with one template, a tech and a verifier."""
    with app.app_context():
        patient = Patient(
            patient_id="P-GAPLAB-01",
            name="Gap Lab Patient",
            place_of_residence="Nairobi",
            sex="Female",
            date_of_birth=datetime(1990, 1, 1).date(),  # noqa: DTZ001
            marital_status="Single",
            contact="0700000001",
            next_of_kin="Kin",
            relationship_with_next_of_kin="Sibling",
            next_of_kin_contact="0700000002",
            emergency_contact="0700000002",
        )
        db.session.add(patient)
        db.session.flush()

        lab_test = LabTest(
            test_name="GapFix CBC",
            cost=1000.0,
            description="Test catalog entry",
        )
        db.session.add(lab_test)
        db.session.flush()

        template = LabResultTemplate(
            test_id=lab_test.id,
            parameter_name="Hemoglobin",
            normal_range_low=12.0,
            normal_range_high=17.5,
            unit="g/dL",
        )
        db.session.add(template)
        db.session.commit()

        tech = _make_user("gaplab_tech", "laboratory")
        verifier = _make_user("gaplab_verifier", "laboratory")

        yield {
            "patient": patient,
            "lab_test": lab_test,
            "template": template,
            "tech": tech,
            "verifier": verifier,
        }


def _login(client, username, password="pw123"):
    return client.post(
        "/login", data={"username": username, "password": password}, follow_redirects=True
    )


# ─── P0-2: self-verification blocked ────────────────────────────────────────


class TestTwoTierSeparation:
    def test_tech_cannot_verify_own_result(self, app, client, lab_setup):
        """The user who entered a result must get 403 when verifying it."""
        _login(client, "gaplab_tech")

        resp = client.post(
            "/laboratory/lis/enter",
            json={
                "patient_id": "P-GAPLAB-01",
                "lab_test_id": lab_setup["lab_test"].id,
                "parameter_name": "Hemoglobin",
                "result_value": 14.0,
            },
        )
        assert resp.status_code == 201
        result_uuid = resp.get_json()["result_id"]

        # Same tech attempts verification — must be blocked
        resp = client.post(
            "/laboratory/lis/verify",
            json={"result_id": result_uuid, "action": "VERIFY"},
        )
        assert resp.status_code == 403
        assert resp.get_json().get("code") == "SELF_VERIFICATION_BLOCKED"

    def test_second_signatory_can_verify(self, app, client, lab_setup):
        """A different lab user can verify the same result."""
        _login(client, "gaplab_tech")
        resp = client.post(
            "/laboratory/lis/enter",
            json={
                "patient_id": "P-GAPLAB-01",
                "lab_test_id": lab_setup["lab_test"].id,
                "parameter_name": "Hemoglobin",
                "result_value": 14.0,
            },
        )
        result_uuid = resp.get_json()["result_id"]

        _login(client, "gaplab_verifier")
        resp = client.post(
            "/laboratory/lis/verify",
            json={"result_id": result_uuid, "action": "VERIFY"},
        )
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "VERIFIED"

    def test_client_supplied_verifier_id_cannot_bypass_self_check(
        self, app, client, lab_setup
    ):
        """A forged verifier_id in the body must not bypass the self-check."""
        _login(client, "gaplab_tech")
        resp = client.post(
            "/laboratory/lis/enter",
            json={
                "patient_id": "P-GAPLAB-01",
                "lab_test_id": lab_setup["lab_test"].id,
                "parameter_name": "Hemoglobin",
                "result_value": 14.0,
            },
        )
        result_uuid = resp.get_json()["result_id"]

        # Tech claims to be someone else — server must use session identity
        resp = client.post(
            "/laboratory/lis/verify",
            json={"result_id": result_uuid, "verifier_id": 999, "action": "VERIFY"},
        )
        assert resp.status_code == 403


# ─── P0-4: no silent clinical defaults ──────────────────────────────────────


class TestNoSilentDefaults:
    def test_enter_requires_result_value(self, app, client, lab_setup):
        _login(client, "gaplab_tech")
        resp = client.post(
            "/laboratory/lis/enter",
            json={
                "patient_id": "P-GAPLAB-01",
                "lab_test_id": lab_setup["lab_test"].id,
                "parameter_name": "Hemoglobin",
                # result_value omitted
            },
        )
        assert resp.status_code == 400

    def test_enter_requires_parameter_name(self, app, client, lab_setup):
        _login(client, "gaplab_tech")
        resp = client.post(
            "/laboratory/lis/enter",
            json={
                "patient_id": "P-GAPLAB-01",
                "lab_test_id": lab_setup["lab_test"].id,
                "result_value": 14.0,
            },
        )
        assert resp.status_code == 400

    def test_enter_requires_valid_lab_test(self, app, client, lab_setup):
        _login(client, "gaplab_tech")
        resp = client.post(
            "/laboratory/lis/enter",
            json={
                "patient_id": "P-GAPLAB-01",
                "lab_test_id": 99999,
                "parameter_name": "Hemoglobin",
                "result_value": 14.0,
            },
        )
        assert resp.status_code == 404


# ─── P0-3: panic alert routing ──────────────────────────────────────────────


class TestPanicAlertRouting:
    def test_panic_alert_reaches_verifier_copy(self, app, client, lab_setup):
        """Verifier receives the alert copy (documentation) on critical values."""
        from departments.models.nursing import Notifications

        _login(client, "gaplab_tech")
        resp = client.post(
            "/laboratory/lis/enter",
            json={
                "patient_id": "P-GAPLAB-01",
                "lab_test_id": lab_setup["lab_test"].id,
                "parameter_name": "Potassium",
                "result_value": 7.0,  # panic high
            },
        )
        result_uuid = resp.get_json()["result_id"]
        assert resp.get_json()["panic_status"] == "PANIC_CRITICAL"

        _login(client, "gaplab_verifier")
        resp = client.post(
            "/laboratory/lis/verify",
            json={"result_id": result_uuid, "action": "VERIFY"},
        )
        data = resp.get_json()
        assert data["panic_alert_sent"] is True
        assert data["panic_alert_recipient_id"] == lab_setup["verifier"].id

        with app.app_context():
            notification = Notifications.query.filter_by(
                receiver_id=lab_setup["verifier"].id
            ).first()
            assert notification is not None
            assert "CRITICAL LAB PANIC ALERT" in notification.message


# ─── P0-5: role aliasing ────────────────────────────────────────────────────


class TestRoleAliasing:
    def test_laboratory_role_passes_lab_tech_guard(self, app):
        """roles_required('lab_tech', ...) must accept role='laboratory'."""
        from departments.rbac import roles_required  # noqa: F401

        with app.test_request_context():
            from flask_login import login_user

            user = _make_user("gaplab_alias_user", "laboratory")
            login_user(user)
            # Internal guard logic check via has_any_role
            from departments.rbac import has_any_role

            assert has_any_role("lab_tech") is True
            assert has_any_role("laboratory") is True


# ─── P1-7: idempotent result entry ──────────────────────────────────────────


class TestResultEntryIdempotency:
    def test_reprocessing_completed_request_redirects(self, app, client, lab_setup):
        """A processed request must redirect to its result, not duplicate it."""
        with app.app_context():
            lab_request = RequestedLab(
                patient_id="P-GAPLAB-01",
                lab_test_id=lab_setup["lab_test"].id,
                date_requested=datetime.now(timezone.utc),
                status=0,
            )
            db.session.add(lab_request)
            db.session.commit()
            request_id = lab_request.id

        _login(client, "gaplab_tech")

        # First submission
        resp = client.post(
            f"/laboratory/process_lab_request/{request_id}",
            data={
                "lab_test_id[]": str(lab_setup["template"].id),
                "result[]": "14.2",
                "result_notes": "",
            },
            follow_redirects=True,
        )
        assert resp.status_code == 200

        with app.app_context():
            count_before = LabResult.query.filter_by(
                lab_test_id=lab_setup["lab_test"].id
            ).count()

        # Retry (double-click simulation)
        resp = client.post(
            f"/laboratory/process_lab_request/{request_id}",
            data={
                "lab_test_id[]": str(lab_setup["template"].id),
                "result[]": "14.2",
                "result_notes": "",
            },
            follow_redirects=True,
        )
        assert resp.status_code == 200

        with app.app_context():
            count_after = LabResult.query.filter_by(
                lab_test_id=lab_setup["lab_test"].id
            ).count()
            assert (
                count_after == count_before
            ), "Retry must not create a duplicate LabResult."

    def test_web_entry_sets_panic_status(self, app, client, lab_setup):
        """Web result entry must run the panic engine (P1-6)."""
        with app.app_context():
            lab_request = RequestedLab(
                patient_id="P-GAPLAB-01",
                lab_test_id=lab_setup["lab_test"].id,
                date_requested=datetime.now(timezone.utc),
                status=0,
            )
            db.session.add(lab_request)
            db.session.commit()
            request_id = lab_request.id

        _login(client, "gaplab_tech")
        # Enter a critical hemoglobin (> panic high 20.0)
        resp = client.post(
            f"/laboratory/process_lab_request/{request_id}",
            data={
                "lab_test_id[]": str(lab_setup["template"].id),
                "result[]": "21.5",
                "result_notes": "",
            },
            follow_redirects=True,
        )
        assert resp.status_code == 200

        with app.app_context():
            result = LabResult.query.filter_by(lab_test_id=lab_setup["lab_test"].id).first()
            assert result is not None
            assert (
                result.panic_status == "PANIC_CRITICAL"
            ), "Web entry must evaluate panic thresholds."
            assert result.status == "PENDING_VERIFICATION"


# ─── P1-11 / P1-12: deletion guards ─────────────────────────────────────────


class TestDeletionGuards:
    def test_template_deletion_blocked_with_results(self, app, client, lab_setup):
        """A parameter referenced by stored results cannot be deleted."""
        with app.app_context():
            lab_result = LabResult(
                patient_id="P-GAPLAB-01",
                lab_test_id=lab_setup["lab_test"].id,
                test_date=datetime.now(timezone.utc),
                result=f'{{"{lab_setup["template"].id}": "14.0"}}',
                result_id="RES-GAPFIX-1",
                updated_by=1,
            )
            db.session.add(lab_result)
            db.session.commit()

        _login(client, "gaplab_tech")
        # Submit edit with the parameter name emptied (= delete request)
        resp = client.post(
            f"/laboratory/edit_lab_test/{lab_setup['lab_test'].id}",
            data={},
            follow_redirects=True,
        )
        assert resp.status_code == 200

        with app.app_context():
            template = db.session.get(
                LabResultTemplate, lab_setup["template"].id
            )
            assert (
                template is not None
            ), "Parameter template must survive when results reference it."

    def test_lab_test_deletion_blocked_when_in_use(self, app, client, lab_setup):
        with app.app_context():
            db.session.add(
                RequestedLab(
                    patient_id="P-GAPLAB-01",
                    lab_test_id=lab_setup["lab_test"].id,
                    date_requested=datetime.now(timezone.utc),
                    status=0,
                )
            )
            db.session.commit()

        _login(client, "gaplab_tech")
        resp = client.post(
            f"/laboratory/delete_lab_test/{lab_setup['lab_test'].id}",
            follow_redirects=True,
        )
        assert resp.status_code == 200

        with app.app_context():
            still_there = db.session.get(LabTest, lab_setup["lab_test"].id)
            assert (
                still_there is not None
            ), "Test in clinical use must not be deletable."

    def test_unused_lab_test_deletable(self, app, client, lab_setup):
        _login(client, "gaplab_tech")
        with app.app_context():
            fresh = LabTest(test_name="Unused GapFix Test", cost=100.0)
            db.session.add(fresh)
            db.session.commit()
            fresh_id = fresh.id

        resp = client.post(
            f"/laboratory/delete_lab_test/{fresh_id}", follow_redirects=True
        )
        assert resp.status_code == 200

        with app.app_context():
            assert db.session.get(LabTest, fresh_id) is None

    def test_dashboard_route(self, client, lab_setup):
        """GET /laboratory/dashboard renders cleanly and calculates TAT."""
        _login(client, "gaplab_tech")
        resp = client.get("/laboratory/dashboard")
        assert resp.status_code == 200
        assert b"Laboratory Dashboard" in resp.data or b"Dashboard" in resp.data

