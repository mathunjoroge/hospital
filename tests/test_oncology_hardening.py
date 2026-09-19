"""
tests/test_oncology_hardening.py
────────────────────────────────
Regression tests for the oncology module review.

Every test here corresponds to a defect that was reproduced against the
original code (most of them on real PostgreSQL) and would FAIL on that code.
Groups:
  A. Chemotherapy engine     – input validation, units, cumulative exposure,
                                fail-closed history, AC-T phases
  B. Chemo API               – patient existence, strict parsing, cycle rules,
                                toxicity override, duplicates, audit, status flow
  C. Patient chart           – identity mapping (patient number, not int PK),
                                edit persistence, note void/amend, AI-summary consent
  D. Bookings / legacy Rx    – exact patient match, integer FK mapping, provenance
  E. Access control          – non-clinical roles, anonymous
"""

import json
from datetime import date

import pytest
from flask import g
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text

from departments.medicine import chemotherapy_engine as engine
from departments.medicine.chemotherapy_engine import (
    ChemoInputError,
    CumulativeHistoryUnavailable,
    calculate_bsa,
    calculate_regimen_doses,
    get_patient_cumulative_doses,
)
from departments.models.compliance import AuditLog, PatientConsent
from departments.models.medicine import (
    CancerStage,
    CancerType,
    CancerTypeStage,
    OncologyBooking,
    OncologyDrug,
    OncologyNote,
    OncologyRegimen,
    OncoPatient,
    OncoPrescription,
    RegimenDrugAssociation,
)
from departments.models.oncology_models import ChemotherapyRegimenOrder
from departments.models.records import Patient
from departments.models.user import User
from extensions import db

# ─────────────────────────────── fixtures / helpers ───────────────────────────


def _mk_user(role, name=None):
    user = User(username=name or f"u_{role}", password="x", role=role)
    db.session.add(user)
    db.session.commit()
    return user.id


def _mk_patient(pid, name="Test Patient"):
    p = Patient(patient_id=pid, name=name, sex="Female", date_of_birth=date(1985, 5, 5))
    db.session.add(p)
    db.session.commit()
    return p


def _login(client, user_id):
    """Log `user_id` in. Requests reuse the test's app context, so also drop
    flask-login's per-context cached user or a previous login would leak."""
    g.pop("_login_user", None)
    with client.session_transaction() as sess:
        sess["_user_id"] = str(user_id)
        sess["_fresh"] = True


@pytest.fixture
def doctor(app):
    return _mk_user("doctor")


@pytest.fixture
def patient(app):
    return _mk_patient("P0001", "Amy First")


def _order(
    patient_id="P0001",
    doses=None,
    bsa=1.82,
    status="ORDERED",
    cycle=1,
    protocol="AC-T",
    physician_id=1,
):
    order = ChemotherapyRegimenOrder(
        patient_id=patient_id,
        physician_id=physician_id,
        protocol_name=protocol,
        weight_kg=70.0,
        height_cm=170.0,
        bsa_m2=bsa,
        cycle_number=cycle,
        status=status,
        calculated_doses_json=json.dumps(
            doses or [{"drug_name": "Doxorubicin", "calculated_dose": 100.0}]
        ),
    )
    db.session.add(order)
    db.session.commit()
    return order


SAVE = "/medicine/oncology/api/save-chemo-order"
CALC = "/medicine/oncology/api/calculate-chemo"


def _save_body(**over):
    body = {
        "patient_id": "P0001",
        "protocol_name": "CHOP",
        "height_cm": 170,
        "weight_kg": 70,
        "bsa_formula": "mosteller",
        "cycle_number": 1,
        "total_cycles": 6,
    }
    body.update(over)
    return body


# ══════════════════════════════════════════════════════════════════════════════
# A. Chemotherapy engine
# ══════════════════════════════════════════════════════════════════════════════


class TestBsaInputValidation:
    """There must be no silent 'standard adult' fallback (was: return 1.73)."""

    @pytest.mark.parametrize(
        "height,weight",
        [
            (0, 0),
            (-5, 70),
            (170, -1),
            (float("nan"), 70),
            (170, float("inf")),
            (None, 70),
            ("", 70),
            ("abc", 70),
            (True, 70),
            (17, 700),  # unit mistakes: plausible-looking BSA, absurd inputs
            (1700, 70),
        ],
    )
    def test_rejects_bad_biometrics(self, height, weight):
        with pytest.raises(ChemoInputError):
            calculate_bsa(height, weight)

    def test_rejects_unknown_formula(self):
        with pytest.raises(ChemoInputError):
            calculate_bsa(170, 70, formula="xyz")

    def test_valid_values_unchanged(self):
        assert calculate_bsa(170, 70, "mosteller") == 1.82
        assert calculate_bsa(170, 70, "DuBois") == 1.81  # case-insensitive
        assert calculate_bsa("170", "70") == 1.82  # numeric strings from forms


class TestDoseUnitsAndPhases:
    def test_absolute_dose_is_not_labelled_per_m2(self, app):
        res = calculate_regimen_doses("P0001", "AC-T", 170, 70)
        dox = next(d for d in res["drugs"] if d["drug_name"] == "Doxorubicin")
        assert dox["calculated_dose"] == 109.2
        assert dox["dose_display"] == "109.2 mg"  # was "109.2 mg/m2"
        assert dox["dose_unit"] == "mg"
        assert dox["unit"] == "mg/m2"  # protocol basis is still per m2

    def test_bleomycin_units(self, app):
        res = calculate_regimen_doses("P0001", "ABVD", 170, 70)
        bleo = next(d for d in res["drugs"] if d["drug_name"] == "Bleomycin")
        assert bleo["dose_display"].endswith(" units")
        assert "/m2" not in bleo["dose_display"]

    def test_bleomycin_cap_message_uses_absolute_unit(self, app):
        _order(doses=[{"drug_name": "Bleomycin", "calculated_dose": 299.0}])
        res = calculate_regimen_doses("P0001", "ABVD", 170, 70)
        assert res["has_toxicity_warning"] is True
        msg = " ".join(res["toxicity_warnings"])
        # 299 prior + 18.2 (10 units/m2 x 1.82) = 317.2 absolute units
        assert "(317.2 units)" in msg
        assert "cap of 300.0 units" in msg
        assert "units/m2" not in msg

    @pytest.mark.parametrize(
        "cycle,expected",
        [
            (1, ["Doxorubicin", "Cyclophosphamide"]),
            (4, ["Doxorubicin", "Cyclophosphamide"]),
            (5, ["Paclitaxel"]),
            (8, ["Paclitaxel"]),
        ],
    )
    def test_ac_t_is_sequential_not_concurrent(self, app, cycle, expected):
        res = calculate_regimen_doses("P0001", "AC-T", 170, 70, cycle_number=cycle)
        assert [d["drug_name"] for d in res["drugs"]] == expected

    def test_ac_t_rejects_cycle_beyond_protocol(self, app):
        res = calculate_regimen_doses("P0001", "AC-T", 170, 70, cycle_number=9)
        assert res["error"] is True and res["code"] == "CYCLE_OUT_OF_RANGE"

    @pytest.mark.parametrize("cycle", [0, -3, 31, 1.5, "x", None])
    def test_rejects_bad_cycle_numbers(self, app, cycle):
        res = calculate_regimen_doses("P0001", "CHOP", 170, 70, cycle_number=cycle)
        assert res["error"] is True and res["code"] == "CYCLE_OUT_OF_RANGE"

    def test_engine_returns_error_dict_not_defaults(self, app):
        res = calculate_regimen_doses("P0001", "CHOP", None, 70)
        assert res["error"] is True and res["code"] == "INVALID_INPUT"

    def test_default_total_cycles_exposed(self, app):
        assert (
            calculate_regimen_doses("P0001", "AC-T", 170, 70)["default_total_cycles"]
            == 8
        )


class TestCumulativeExposure:
    def test_cancelled_orders_do_not_count(self, app):
        _order(
            doses=[{"drug_name": "Doxorubicin", "calculated_dose": 800.0}],
            status="CANCELLED",
        )
        assert get_patient_cumulative_doses("P0001") == {}
        res = calculate_regimen_doses("P0001", "AC-T", 170, 70)
        assert res["has_toxicity_warning"] is False

    def test_active_orders_count(self, app):
        _order(doses=[{"drug_name": "Doxorubicin", "calculated_dose": 800.0}])
        assert get_patient_cumulative_doses("P0001") == {"Doxorubicin": 800.0}

    def test_per_m2_uses_each_orders_own_bsa(self, app):
        """
        Prior: 400 mg at BSA 1.0  = 400 mg/m2.
        New  : 60 mg/m2 at BSA 2.0 = 120 mg.
        True lifetime = 400 + 60 = 460 mg/m2 (> 450 cap).
        Dividing the mg total by the NEW bsa gives (400+120)/2.0 = 260 -> cap missed.
        """
        _order(doses=[{"drug_name": "Doxorubicin", "calculated_dose": 400.0}], bsa=1.0)
        res = calculate_regimen_doses("P0001", "AC-T", 180, 80)  # BSA 2.00
        assert res["bsa_m2"] == 2.0
        dox = next(d for d in res["drugs"] if d["drug_name"] == "Doxorubicin")
        assert dox["new_cumulative_per_m2"] == 460.0
        assert dox["cap_exceeded"] is True and res["has_toxicity_warning"] is True

    def test_legacy_low_history_still_within_cap(self, app):
        _order(doses=[{"drug_name": "Doxorubicin", "calculated_dose": 200.0}])
        res = calculate_regimen_doses("P0001", "AC-T", 170, 70)
        assert res["has_toxicity_warning"] is False

    def test_fails_closed_when_history_table_is_missing(self, app):
        """Production reality before the migration: the table did not exist and
        the old code returned {} -> caps silently skipped."""
        ChemotherapyRegimenOrder.__table__.drop(db.engine)
        with pytest.raises(CumulativeHistoryUnavailable):
            get_patient_cumulative_doses("P0001")
        res = calculate_regimen_doses("P0001", "AC-T", 170, 70)
        assert res["error"] is True and res["code"] == "CUMULATIVE_HISTORY_UNAVAILABLE"

    def test_fails_closed_on_corrupt_history_json(self, app):
        order = _order()
        order.calculated_doses_json = "{not json"
        db.session.commit()
        with pytest.raises(CumulativeHistoryUnavailable):
            get_patient_cumulative_doses("P0001")

    def test_vincristine_cap_applies_even_without_protocol_level_cap(
        self, app, monkeypatch
    ):
        proto = json.loads(json.dumps(engine.CHEMO_PROTOCOLS["CHOP"]))
        for d in proto["drugs"]:
            d.pop("cap_max_mg", None)
        monkeypatch.setitem(engine.CHEMO_PROTOCOLS, "CHOP", proto)
        res = calculate_regimen_doses("P0001", "CHOP", 200, 120)
        vinc = next(d for d in res["drugs"] if d["drug_name"] == "Vincristine")
        assert vinc["calculated_dose"] == 2.0


# ══════════════════════════════════════════════════════════════════════════════
# B. Chemo API
# ══════════════════════════════════════════════════════════════════════════════


class TestCalculateEndpoint:
    def test_unknown_patient_404(self, client, doctor):
        _login(client, doctor)
        r = client.get(f"{CALC}?patient_id=NOPE&protocol=CHOP&height=170&weight=70")
        assert r.status_code == 404 and r.get_json()["code"] == "PATIENT_NOT_FOUND"

    @pytest.mark.parametrize(
        "qs",
        [
            "height=nan&weight=70",
            "height=inf&weight=70",
            "height=abc&weight=xyz",
            "height=-5&weight=70",
            "height=17&weight=700",
            "",  # missing entirely — used to silently become 170 cm / 70 kg
            "height=170",
        ],
    )
    def test_bad_or_missing_biometrics_rejected(self, client, doctor, patient, qs):
        _login(client, doctor)
        r = client.get(f"{CALC}?patient_id=P0001&protocol=CHOP&{qs}")
        assert r.status_code == 400
        assert r.get_json()["code"] == "INVALID_INPUT"

    def test_missing_protocol_rejected(self, client, doctor, patient):
        _login(client, doctor)
        r = client.get(f"{CALC}?patient_id=P0001&height=170&weight=70")
        assert r.status_code == 400 and r.get_json()["code"] == "UNKNOWN_PROTOCOL"

    def test_valid_request_and_cycle_param(self, client, doctor, patient):
        _login(client, doctor)
        r = client.get(
            f"{CALC}?patient_id=P0001&protocol=AC-T&height=170&weight=70&cycle=5"
        )
        body = r.get_json()
        assert r.status_code == 200 and body["bsa_m2"] == 1.82
        assert [d["drug_name"] for d in body["drugs"]] == ["Paclitaxel"]

    def test_history_unavailable_is_503_not_silent_pass(self, client, doctor, patient):
        ChemotherapyRegimenOrder.__table__.drop(db.engine)
        _login(client, doctor)
        r = client.get(f"{CALC}?patient_id=P0001&protocol=AC-T&height=170&weight=70")
        assert r.status_code == 503
        assert r.get_json()["code"] == "CUMULATIVE_HISTORY_UNAVAILABLE"

    def test_response_is_valid_json_never_nan(self, client, doctor, patient):
        _login(client, doctor)
        r = client.get(f"{CALC}?patient_id=P0001&protocol=CHOP&height=nan&weight=70")
        assert b"NaN" not in r.data and b"Infinity" not in r.data


class TestSaveEndpoint:
    def test_happy_path_persists_and_audits(self, client, doctor, patient):
        _login(client, doctor)
        r = client.post(SAVE, json=_save_body())
        assert r.status_code == 201
        order = db.session.get(ChemotherapyRegimenOrder, r.get_json()["order_id"])
        assert order.status == "ORDERED" and order.physician_id == doctor
        assert order.patient_id == "P0001" and order.bsa_formula == "mosteller"
        audit = AuditLog.query.filter_by(action="CHEMO_ORDER_CREATED").one()
        assert audit.resource_id == str(order.id) and audit.user_id == doctor

    def test_unknown_patient_rejected_and_nothing_saved(self, client, doctor):
        _login(client, doctor)
        r = client.post(SAVE, json=_save_body(patient_id="NOPE"))
        assert r.status_code == 404
        assert ChemotherapyRegimenOrder.query.count() == 0

    def test_bogus_formula_rejected(self, client, doctor, patient):
        _login(client, doctor)
        r = client.post(SAVE, json=_save_body(bsa_formula="xyz"))
        assert r.status_code == 400
        assert ChemotherapyRegimenOrder.query.count() == 0

    def test_formula_is_normalised(self, client, doctor, patient):
        _login(client, doctor)
        client.post(SAVE, json=_save_body(bsa_formula="DuBois"))
        assert ChemotherapyRegimenOrder.query.one().bsa_formula == "dubois"

    @pytest.mark.parametrize(
        "override",
        [
            {"cycle_number": -3},
            {"cycle_number": 0},
            {"cycle_number": 2.5},
            {"cycle_number": "x"},
            {"total_cycles": 0},
            {"total_cycles": 99},
            {"total_cycles": 1, "cycle_number": 3},
            {"height_cm": "tall"},  # used to raise ValueError -> HTTP 500
            {"height_cm": None},
            {"weight_kg": float("nan")},
            {"weight_kg": 0},
        ],
    )
    def test_invalid_numbers_are_400_not_500(self, client, doctor, patient, override):
        _login(client, doctor)
        r = client.post(SAVE, json=_save_body(**override))
        assert r.status_code == 400, r.get_json()
        assert ChemotherapyRegimenOrder.query.count() == 0

    def test_total_cycles_defaults_from_protocol(self, client, doctor, patient):
        _login(client, doctor)
        body = _save_body(protocol_name="AC-T")
        body.pop("total_cycles")
        client.post(SAVE, json=body)
        assert ChemotherapyRegimenOrder.query.one().total_cycles == 8

    def test_duplicate_active_cycle_conflicts(self, client, doctor, patient):
        _login(client, doctor)
        assert client.post(SAVE, json=_save_body()).status_code == 201
        r = client.post(SAVE, json=_save_body())  # double-click / retry
        assert r.status_code == 409 and r.get_json()["code"] == "DUPLICATE_CYCLE_ORDER"
        assert ChemotherapyRegimenOrder.query.count() == 1

    def test_db_enforces_single_active_order_even_if_app_check_is_bypassed(
        self, app, patient
    ):
        _order(protocol="CHOP", cycle=1)
        db.session.add(
            ChemotherapyRegimenOrder(
                patient_id="P0001",
                physician_id=1,
                protocol_name="CHOP",
                weight_kg=70,
                height_cm=170,
                bsa_m2=1.82,
                cycle_number=1,
                calculated_doses_json="[]",
                status="ORDERED",
            )
        )
        with pytest.raises(Exception, match="(?i)unique"):
            db.session.commit()
        db.session.rollback()

    def test_cancelled_duplicate_can_be_reordered(self, client, doctor, patient):
        _login(client, doctor)
        first = client.post(SAVE, json=_save_body()).get_json()["order_id"]
        client.post(
            f"/medicine/oncology/api/chemo-order/{first}/status",
            json={"status": "CANCELLED", "reason": "entered in error"},
        )
        assert client.post(SAVE, json=_save_body()).status_code == 201


class TestToxicityOverride:
    def _over_cap(self):
        _order(doses=[{"drug_name": "Doxorubicin", "calculated_dose": 800.0}])

    def _ac_t(self, **over):
        return _save_body(protocol_name="AC-T", total_cycles=8, cycle_number=2, **over)

    def test_cap_exceeded_blocks_save_without_reason(self, client, doctor, patient):
        self._over_cap()
        _login(client, doctor)
        r = client.post(SAVE, json=self._ac_t())
        assert r.status_code == 409
        body = r.get_json()
        assert (
            body["code"] == "TOXICITY_OVERRIDE_REQUIRED" and body["toxicity_warnings"]
        )
        assert ChemotherapyRegimenOrder.query.count() == 1  # only the seeded one

    def test_short_reason_is_not_enough(self, client, doctor, patient):
        self._over_cap()
        _login(client, doctor)
        r = client.post(SAVE, json=self._ac_t(toxicity_override_reason="ok"))
        assert r.status_code == 409

    def test_documented_override_saves_and_is_audited(self, client, doctor, patient):
        self._over_cap()
        _login(client, doctor)
        why = "Curative intent, cardiology cleared, dexrazoxane cover"
        r = client.post(SAVE, json=self._ac_t(toxicity_override_reason=why))
        assert r.status_code == 201 and r.get_json()["has_toxicity_warning"] is True
        order = db.session.get(ChemotherapyRegimenOrder, r.get_json()["order_id"])
        assert order.toxicity_override_reason == why
        audit = AuditLog.query.filter_by(action="CHEMO_ORDER_CREATED").one()
        assert why in audit.details

    def test_stray_override_reason_ignored_when_no_warning(
        self, client, doctor, patient
    ):
        _login(client, doctor)
        client.post(SAVE, json=_save_body(toxicity_override_reason="not needed here"))
        assert ChemotherapyRegimenOrder.query.one().toxicity_override_reason is None

    def test_order_not_saved_when_audit_cannot_be_written(
        self, client, doctor, patient, monkeypatch
    ):
        import departments.medicine.oncology as onc

        monkeypatch.setattr(onc, "log_audit_event", lambda **kw: None)
        _login(client, doctor)
        r = client.post(SAVE, json=_save_body())
        assert r.status_code == 503 and r.get_json()["code"] == "AUDIT_UNAVAILABLE"


class TestOrderStatusWorkflow:
    def _new_order(self, client):
        return client.post(SAVE, json=_save_body()).get_json()["order_id"]

    def _set(self, client, oid, status, reason=None):
        body = {"status": status}
        if reason:
            body["reason"] = reason
        return client.post(
            f"/medicine/oncology/api/chemo-order/{oid}/status", json=body
        )

    def test_full_lifecycle(self, client, doctor, patient):
        _login(client, doctor)
        oid = self._new_order(client)
        assert self._set(client, oid, "PREPARED").status_code == 200
        r = self._set(client, oid, "ADMINISTERED")
        assert r.status_code == 200 and r.get_json()["status"] == "ADMINISTERED"
        assert (
            AuditLog.query.filter_by(action="CHEMO_ORDER_STATUS_CHANGED").count() == 2
        )

    def test_cannot_skip_or_leave_terminal_states(self, client, doctor, patient):
        _login(client, doctor)
        oid = self._new_order(client)
        assert (
            self._set(client, oid, "ADMINISTERED").status_code == 409
        )  # skipped PREPARED
        self._set(client, oid, "PREPARED")
        self._set(client, oid, "ADMINISTERED")
        assert self._set(client, oid, "CANCELLED", "too late now").status_code == 409

    def test_cancel_requires_reason_and_removes_from_cumulative(
        self, client, doctor, patient
    ):
        _login(client, doctor)
        oid = self._new_order(client)
        assert self._set(client, oid, "CANCELLED").status_code == 400
        assert self._set(client, oid, "CANCELLED", "duplicate entry").status_code == 200
        order = db.session.get(ChemotherapyRegimenOrder, oid)
        assert order.status == "CANCELLED" and order.status_reason == "duplicate entry"
        assert order.status_changed_by == doctor
        assert get_patient_cumulative_doses("P0001") == {}  # excluded, row retained

    def test_nurse_may_administer_but_not_cancel(self, client, doctor, patient):
        _login(client, doctor)
        oid = self._new_order(client)
        self._set(client, oid, "PREPARED")
        _login(client, _mk_user("nurse"))
        assert self._set(client, oid, "CANCELLED", "no good reason").status_code == 403
        assert self._set(client, oid, "ADMINISTERED").status_code == 200

    def test_unknown_order_and_status(self, client, doctor, patient):
        _login(client, doctor)
        assert self._set(client, 9999, "PREPARED").status_code == 404
        oid = self._new_order(client)
        assert self._set(client, oid, "BOGUS").status_code == 400

    def test_list_endpoint(self, client, doctor, patient):
        _login(client, doctor)
        oid = self._new_order(client)
        self._set(client, oid, "CANCELLED", "entered in error")
        r = client.get("/medicine/oncology/api/chemo-orders/P0001")
        orders = r.get_json()["orders"]
        assert r.status_code == 200 and orders[0]["order_id"] == oid
        assert orders[0]["status"] == "CANCELLED"
        assert client.get("/medicine/oncology/api/chemo-orders/NOPE").status_code == 404


class TestChemoBuilderPage:
    def test_builder_requires_existing_patient(self, client, doctor):
        _login(client, doctor)
        assert client.get("/medicine/oncology/chemo-builder/NOPE").status_code == 404

    def test_builder_shows_patient_name_and_no_prefilled_biometrics(
        self, client, doctor, patient
    ):
        _login(client, doctor)
        html = client.get("/medicine/oncology/chemo-builder/P0001").get_data(
            as_text=True
        )
        assert "Amy First" in html
        assert 'id="heightCm" value=' not in html and 'id="weightKg" value=' not in html


# ══════════════════════════════════════════════════════════════════════════════
# C. Patient chart
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def cancer_ref(app):
    """A cancer type with two linked stages plus one unrelated stage."""
    breast = CancerType(code="BRE", name="Breast")
    st1 = CancerStage(code="S2", label="Stage II")
    st2 = CancerStage(code="S3", label="Stage III")
    other = CancerStage(code="AA1", label="Ann Arbor I")
    db.session.add_all([breast, st1, st2, other])
    db.session.flush()
    db.session.add_all(
        [
            CancerTypeStage(cancer_type_id=breast.id, cancer_stage_id=st1.id),
            CancerTypeStage(cancer_type_id=breast.id, cancer_stage_id=st2.id),
        ]
    )
    db.session.commit()
    return {"type": breast.id, "s2": st1.id, "s3": st2.id, "other": other.id}


def _enrol(pid="P0001", **over):
    row = dict(
        patient_id=pid,
        diagnosis="Invasive ductal carcinoma",
        diagnosis_date=date(2026, 1, 1),
        cancer_type="Breast",
        stage="Stage II",
    )
    row.update(over)
    onco = OncoPatient(**row)
    db.session.add(onco)
    db.session.commit()
    return onco


def _encounter(pid="P0001"):
    return f"/medicine/oncology/encounter/{pid}"


def _book(pid="P0001"):
    """encounter.html renders the oncology forms and notes only for patients that
    have an oncology booking."""
    db.session.add(
        OncologyBooking(
            patient_id=pid,
            booking_date=date(2030, 1, 1),
            purpose="Follow-up",
            status="Scheduled",
        )
    )
    db.session.commit()


class TestEncounterIdentityMapping:
    """
    OncoPatient.patient_id is a string FK to patients.patient_id.  The encounter
    view used the integer primary key, which cannot even be compared to that
    column on PostgreSQL (varchar = integer -> HTTP 500).
    """

    def test_encounter_finds_enrolment_by_patient_number(self, client, doctor, patient):
        _enrol()
        _book()
        _login(client, doctor)
        r = client.get(_encounter())
        assert r.status_code == 200
        assert b"Invasive ductal carcinoma" in r.data  # prepopulated from the record

    def test_new_enrolment_stores_patient_number_not_int_pk(
        self, client, doctor, patient, cancer_ref
    ):
        _login(client, doctor)
        r = client.post(
            _encounter(),
            data={
                "submit_update": "1",
                "diagnosis": "DCIS",
                "diagnosis_date": "2026-02-02",
                "cancer_type": cancer_ref["type"],
                "stage": cancer_ref["s2"],
                "status": "Active",
            },
        )
        assert r.status_code == 302  # PRG redirect
        row = OncoPatient.query.one()
        assert row.patient_id == "P0001"  # not "1"
        assert row.cancer_type == "Breast" and row.stage == "Stage II"  # names, not ids
        assert AuditLog.query.filter_by(action="ONCOLOGY_ENROLLED").count() == 1

    def test_editing_persists_the_submitted_values(
        self, client, doctor, patient, cancer_ref
    ):
        """The form used to be re-populated from the DB BEFORE reading POST data,
        so every edit was silently discarded while 'updated successfully' showed."""
        _enrol()
        _login(client, doctor)
        client.post(
            _encounter(),
            data={
                "submit_update": "1",
                "diagnosis": "Changed dx",
                "diagnosis_date": "2026-03-03",
                "cancer_type": cancer_ref["type"],
                "stage": cancer_ref["s3"],
                "status": "Completed",
            },
        )
        row = OncoPatient.query.one()
        assert (row.diagnosis, row.stage, row.status) == (
            "Changed dx",
            "Stage III",
            "Completed",
        )
        assert OncoPatient.query.count() == 1  # updated in place, not duplicated

    def test_stage_must_belong_to_cancer_type(
        self, client, doctor, patient, cancer_ref
    ):
        _login(client, doctor)
        client.post(
            _encounter(),
            data={
                "submit_update": "1",
                "diagnosis": "x",
                "diagnosis_date": "2026-02-02",
                "cancer_type": cancer_ref["type"],
                "stage": cancer_ref["other"],
                "status": "Active",
            },
        )
        assert OncoPatient.query.count() == 0

    def test_form_prefill_selects_stored_type_and_stage(
        self, client, doctor, patient, cancer_ref
    ):
        _enrol()
        _book()
        _login(client, doctor)
        html = client.get(_encounter()).get_data(as_text=True)
        assert f'<option selected value="{cancer_ref["type"]}">Breast' in html
        assert f'<option selected value="{cancer_ref["s2"]}">Stage II' in html

    def test_stage_filter_script_targets_real_route(self, client, doctor, patient):
        _book()
        _login(client, doctor)
        html = client.get(_encounter()).get_data(as_text=True)
        assert "/medicine/get_stages/" in html  # was "/get_stages/" -> 404

    def test_refreshing_after_post_does_not_duplicate_the_note(
        self, client, doctor, patient
    ):
        _login(client, doctor)
        r = client.post(
            _encounter(),
            data={
                "submit_note": "1",
                "note_date": "2026-04-04",
                "note_content": "Cycle 2 tolerated well",
            },
        )
        assert r.status_code == 302
        assert client.get(r.headers["Location"]).status_code == 200
        assert OncologyNote.query.count() == 1

    def test_encounter_view_is_audited(self, client, doctor, patient):
        _login(client, doctor)
        client.get(_encounter())
        assert AuditLog.query.filter_by(action="ONCOLOGY_ENCOUNTER_VIEW").count() == 1

    def test_enrol_form_posts_patient_number(self, client, doctor, patient):
        """add_onco_patient.html posted the integer PK, which violates the FK on PostgreSQL."""
        _login(client, doctor)
        html = client.get("/medicine/oncology/add").get_data(as_text=True)
        assert 'value="P0001"' in html
        assert f'value="{db.session.get(Patient, 1).id}">Amy First</option>' not in html


class TestEnrolRoute:
    BASE = dict(
        patient_id="P0001",
        diagnosis="Ca",
        cancer_type="Breast",
        stage="II",
        diagnosis_date="2026-01-01",
    )

    def test_happy_path_uses_patient_number(self, client, doctor, patient):
        _login(client, doctor)
        assert client.post("/medicine/oncology/add", data=self.BASE).status_code == 302
        assert OncoPatient.query.one().patient_id == "P0001"

    @pytest.mark.parametrize(
        "field", ["patient_id", "diagnosis", "cancer_type", "stage", "diagnosis_date"]
    )
    def test_missing_field_is_a_validation_message_not_a_500(
        self, client, doctor, patient, field
    ):
        _login(client, doctor)
        data = {**self.BASE, field: ""}
        assert client.post("/medicine/oncology/add", data=data).status_code == 302
        assert OncoPatient.query.count() == 0

    def test_bad_date_future_date_and_unknown_patient(self, client, doctor, patient):
        _login(client, doctor)
        for over in (
            {"diagnosis_date": "01/02/2026"},
            {"diagnosis_date": "2999-01-01"},
            {"patient_id": "NOPE"},
            {"patient_id": "Amy"},
        ):
            client.post("/medicine/oncology/add", data={**self.BASE, **over})
        assert OncoPatient.query.count() == 0

    def test_second_active_enrolment_refused(self, client, doctor, patient):
        _login(client, doctor)
        client.post("/medicine/oncology/add", data=self.BASE)
        client.post("/medicine/oncology/add", data=self.BASE)
        assert OncoPatient.query.count() == 1


class TestNoteVoidAndAmend:
    def _note(self, text_="Original note"):
        n = OncologyNote(
            patient_id="P0001", note_date=date(2026, 1, 1), note_content=text_
        )
        db.session.add(n)
        db.session.commit()
        return n.id

    def _void(self, client, nid, reason="Entered in error"):
        return client.post(
            f"/medicine/oncology/note/{nid}/void", data={"void_reason": reason}
        )

    def test_void_is_really_persisted(self, client, doctor, patient):
        """`is_voided` etc. were never mapped columns: the route reported success
        and stored nothing."""
        nid = self._note()
        _login(client, doctor)
        assert self._void(client, nid).status_code == 302
        db.session.expire_all()
        row = db.session.execute(
            text(
                "select is_voided, voided_reason, voided_by, voided_at, note_content "
                "from oncology_notes where id=:i"
            ),
            {"i": nid},
        ).one()
        assert row.is_voided in (1, True) and row.voided_reason == "Entered in error"
        assert row.voided_by == doctor and row.voided_at is not None
        assert row.note_content == "Original note"  # void, never delete
        assert AuditLog.query.filter_by(action="ONCOLOGY_NOTE_VOIDED").count() == 1

    def test_void_columns_exist_in_schema(self, app):
        cols = {c["name"] for c in sa_inspect(db.engine).get_columns("oncology_notes")}
        assert {"is_voided", "voided_by", "voided_at", "voided_reason"} <= cols

    def test_void_requires_reason(self, client, doctor, patient):
        nid = self._note()
        _login(client, doctor)
        self._void(client, nid, reason="  ")
        assert db.session.get(OncologyNote, nid).is_voided is False

    def test_voided_note_cannot_be_edited_or_voided_again(
        self, client, doctor, patient
    ):
        nid = self._note()
        _login(client, doctor)
        self._void(client, nid)
        client.post(
            f"/medicine/oncology/note/{nid}/edit",
            data={
                "submit_note": "1",
                "note_date": "2026-05-05",
                "note_content": "tampered",
            },
        )
        db.session.expire_all()
        assert db.session.get(OncologyNote, nid).note_content == "Original note"
        self._void(client, nid, reason="again")
        assert AuditLog.query.filter_by(action="ONCOLOGY_NOTE_VOIDED").count() == 1

    def test_voided_note_is_flagged_in_encounter_ui(self, client, doctor, patient):
        _book()
        nid = self._note("wrong-patient text")
        _login(client, doctor)
        self._void(client, nid, reason="wrong patient")
        html = client.get(_encounter()).get_data(as_text=True)
        assert "VOIDED" in html and "wrong patient" in html

    def test_void_form_actually_sends_a_reason(self, client, doctor, patient):
        """The old 'Delete' form posted no void_reason, so voiding from the UI could never work."""
        _book()
        self._note()
        _login(client, doctor)
        html = client.get(_encounter()).get_data(as_text=True)
        assert 'name="void_reason"' in html

    def test_amendment_keeps_previous_text_in_audit_trail(
        self, client, doctor, patient
    ):
        nid = self._note("first version")
        _login(client, doctor)
        client.post(
            f"/medicine/oncology/note/{nid}/edit",
            data={
                "submit_note": "1",
                "note_date": "2026-05-05",
                "note_content": "second version",
            },
        )
        assert db.session.get(OncologyNote, nid).note_content == "second version"
        audit = AuditLog.query.filter_by(action="ONCOLOGY_NOTE_AMENDED").one()
        assert "first version" in audit.details

    def test_no_void_when_audit_cannot_be_written(
        self, client, doctor, patient, monkeypatch
    ):
        import departments.medicine.oncology as onc

        nid = self._note()
        monkeypatch.setattr(onc, "log_audit_event", lambda **kw: None)
        _login(client, doctor)
        self._void(client, nid)
        db.session.expire_all()
        assert db.session.get(OncologyNote, nid).is_voided is False


class TestAiSummaryConsentScope:
    """Consent was checked for the URL id, but the patient was then resolved with
    ILIKE '%id%' on id OR name — so consent for 'P1' released the notes of 'P10'."""

    @pytest.fixture
    def two_patients(self, app):
        _mk_patient(
            "P10", "Zed Tenth"
        )  # inserted first: ILIKE '%P1%'.first() prefers it
        _mk_patient("P1", "Amy First")
        db.session.add(
            PatientConsent(
                patient_id="P1", consent_type="ai_diagnosis", is_granted=True
            )
        )
        db.session.add(
            OncologyNote(
                patient_id="P10",
                note_date=date(2026, 1, 1),
                note_content="SECRET-NOTE-OF-P10",
            )
        )
        db.session.add(
            OncologyNote(
                patient_id="P1", note_date=date(2026, 1, 1), note_content="note of P1"
            )
        )
        db.session.commit()

    @pytest.fixture
    def captured(self, monkeypatch):
        import departments.medicine.oncology as onc

        seen = {}
        monkeypatch.setattr(
            onc.Summarizer,
            "answer",
            lambda t: seen.setdefault("text", t) and "stub",
            raising=False,
        )
        return seen

    def test_only_the_consented_patients_notes_reach_the_model(
        self, client, doctor, two_patients, captured
    ):
        _login(client, doctor)
        r = client.get("/medicine/oncology/ai_summary/P1")
        assert r.status_code == 200
        assert captured["text"] == "note of P1"
        assert "SECRET-NOTE-OF-P10" not in captured["text"]

    def test_partial_id_or_name_does_not_resolve_a_patient(
        self, client, doctor, two_patients, captured
    ):
        _login(client, doctor)
        db.session.add(
            PatientConsent(
                patient_id="Amy", consent_type="ai_diagnosis", is_granted=True
            )
        )
        db.session.commit()
        assert client.get("/medicine/oncology/ai_summary/Amy").status_code == 404
        assert "text" not in captured

    def test_voided_notes_are_never_sent(self, client, doctor, two_patients, captured):
        db.session.add(
            OncologyNote(
                patient_id="P1",
                note_date=date(2026, 2, 2),
                note_content="ENTERED-IN-ERROR",
                is_voided=True,
            )
        )
        db.session.commit()
        _login(client, doctor)
        client.get("/medicine/oncology/ai_summary/P1")
        assert "ENTERED-IN-ERROR" not in captured["text"]

    def test_disclosure_is_audited_and_no_audit_means_no_llm_call(
        self, client, doctor, two_patients, captured, monkeypatch
    ):
        _login(client, doctor)
        client.get("/medicine/oncology/ai_summary/P1")
        assert (
            AuditLog.query.filter_by(action="ONCOLOGY_AI_SUMMARY_GENERATED").count()
            == 1
        )
        captured.clear()
        import departments.medicine.oncology as onc

        monkeypatch.setattr(onc, "log_audit_event", lambda **kw: None)
        r = client.get("/medicine/oncology/ai_summary/P1")
        assert r.status_code == 503 and "text" not in captured

    def test_prompt_size_is_bounded(self, client, doctor, two_patients, captured):
        for i in range(30):
            db.session.add(
                OncologyNote(
                    patient_id="P1",
                    note_date=date(2026, 3, 1),
                    note_content=f"{i:02d}" + "x" * 900,
                )
            )
        db.session.commit()
        _login(client, doctor)
        client.get("/medicine/oncology/ai_summary/P1")
        assert len(captured["text"]) <= 13000


# ══════════════════════════════════════════════════════════════════════════════
# D. Bookings and the legacy prescription flow
# ══════════════════════════════════════════════════════════════════════════════

BOOK = "/medicine/bookings/new"


class TestBookings:
    def _post(self, client, pid, **over):
        data = {
            "patient_id": pid,
            "booking_date": "2030-01-01",
            "purpose": "Chemotherapy",
            "status": "Scheduled",
        }
        data.update(over)
        return client.post(BOOK, data=data)

    def test_exact_patient_number_stored_canonically(self, client, doctor, patient):
        _login(client, doctor)
        assert self._post(client, " P0001 ").status_code == 302
        b = OncologyBooking.query.one()
        assert b.patient_id == "P0001"
        assert AuditLog.query.filter_by(action="ONCOLOGY_BOOKING_CREATED").count() == 1

    @pytest.mark.parametrize("fragment", ["Amy", "P00", "amy first", "%", "P0001%"])
    def test_fuzzy_input_is_rejected(self, client, doctor, patient, fragment):
        """Name/partial-id used to pass validation and the RAW text was stored:
        an FK violation (HTTP 500) on PostgreSQL, an orphaned booking on SQLite."""
        _login(client, doctor)
        assert self._post(client, fragment).status_code == 302
        assert OncologyBooking.query.count() == 0

    def test_booking_list_only_shows_joinable_rows(self, client, doctor, patient):
        _login(client, doctor)
        self._post(client, "P0001")
        assert b"Amy First" in client.get("/medicine/bookings/").data


@pytest.fixture
def rx_setup(app, patient, cancer_ref):
    onco = _enrol()
    booking = OncologyBooking(
        patient_id="P0001",
        booking_date=date(2030, 1, 1),
        purpose="Chemotherapy",
        status="Scheduled",
    )
    drug = OncologyDrug(name="Doxorubicin", dosage_form="Injection", strength="50mg")
    other_drug = OncologyDrug(name="Unrelated", dosage_form="Tablet", strength="10mg")
    regimen = OncologyRegimen(name="AC", cycle_duration_days=21, total_cycles=4)
    db.session.add_all([booking, drug, other_drug, regimen])
    db.session.flush()
    db.session.add(
        RegimenDrugAssociation(regimen_id=regimen.id, drug_id=drug.id, sequence=1)
    )
    db.session.commit()
    return {
        "onco": onco.id,
        "booking": booking.id,
        "drug": drug.id,
        "other_drug": other_drug.id,
        "regimen": regimen.id,
    }


def _rx_form(ids, **over):
    d = ids["drug"]
    # `prescribed_by` is part of the baseline because the ORIGINAL route required it; without
    # it every negative test below would be rejected for that reason alone (i.e. vacuously).
    data = {
        "booking_id": ids["booking"],
        "regimen_id": ids["regimen"],
        "start_date": "2030-01-02",
        "prescribed_by": "Dr Baseline",
        "drug_id": [d],
        f"dosage_{d}": "60 mg/m2",
        f"calculated_dose_{d}": "109.2 mg",
    }
    data.update(over)
    return data


class TestLegacyPrescriptions:
    """onco_prescriptions.onco_patient_id is an INTEGER FK to onco_patients.id;
    the routes wrote/compared the string patient number (crashes on PostgreSQL)."""

    def test_prescription_links_to_the_enrolment_row(self, client, doctor, rx_setup):
        _login(client, doctor)
        r = client.post("/medicine/prescriptions/new", data=_rx_form(rx_setup))
        assert r.status_code == 302
        rx = OncoPrescription.query.one()
        assert rx.onco_patient_id == rx_setup["onco"] and isinstance(
            rx.onco_patient_id, int
        )
        assert (
            AuditLog.query.filter_by(action="CHEMO_PRESCRIPTION_CREATED").count() == 1
        )

    def test_prescriber_comes_from_session_not_from_the_form(
        self, client, doctor, rx_setup
    ):
        _login(client, doctor)
        client.post(
            "/medicine/prescriptions/new",
            data=_rx_form(rx_setup, prescribed_by="Dr Someone Else"),
        )
        assert OncoPrescription.query.one().prescribed_by == "u_doctor"

    def test_drug_must_belong_to_the_regimen(self, client, doctor, rx_setup):
        _login(client, doctor)
        bad = rx_setup["other_drug"]
        data = _rx_form(
            rx_setup,
            drug_id=[bad],
            **{f"dosage_{bad}": "1", f"calculated_dose_{bad}": "1"},
        )
        client.post("/medicine/prescriptions/new", data=data)
        assert OncoPrescription.query.count() == 0

    def test_at_least_one_drug_required(self, client, doctor, rx_setup):
        _login(client, doctor)
        client.post("/medicine/prescriptions/new", data=_rx_form(rx_setup, drug_id=[]))
        assert OncoPrescription.query.count() == 0

    def test_deprecated_regimen_cannot_be_prescribed(self, client, doctor, rx_setup):
        db.session.get(OncologyRegimen, rx_setup["regimen"]).status = "Deprecated"
        db.session.commit()
        _login(client, doctor)
        client.post("/medicine/prescriptions/new", data=_rx_form(rx_setup))
        assert OncoPrescription.query.count() == 0

    def test_patient_must_be_enrolled_in_oncology(self, client, doctor, rx_setup):
        OncoPatient.query.delete()
        db.session.commit()
        _login(client, doctor)
        client.post("/medicine/prescriptions/new", data=_rx_form(rx_setup))
        assert OncoPrescription.query.count() == 0

    def test_list_shows_prescriptions_under_the_right_patient(
        self, client, doctor, rx_setup
    ):
        _login(client, doctor)
        client.post("/medicine/prescriptions/new", data=_rx_form(rx_setup))
        r = client.get("/medicine/prescriptions/P0001")
        assert (
            r.status_code == 200 and b"Amy First" in r.data and b"Doxorubicin" in r.data
        )

    def test_list_does_not_fuzzy_match_another_patient(self, client, doctor, rx_setup):
        _mk_patient("P00010", "Other Person")
        _login(client, doctor)
        assert client.get("/medicine/prescriptions/P0001").status_code == 200
        r = client.get("/medicine/prescriptions/Other")  # a name fragment
        assert r.status_code == 302  # "Patient does not exist", not a fuzzy hit

    def test_all_prescriptions_page_joins_via_onco_patient(
        self, client, doctor, rx_setup
    ):
        _login(client, doctor)
        client.post("/medicine/prescriptions/new", data=_rx_form(rx_setup))
        r = client.get("/medicine/prescriptions")
        assert r.status_code == 200 and b"Amy First" in r.data


# ══════════════════════════════════════════════════════════════════════════════
# E. Access control
# ══════════════════════════════════════════════════════════════════════════════

# (method, url, kwargs).  Everything below handles patient data or clinical
# writes: a non-clinical role must get 403, an anonymous caller must be bounced.
PROTECTED = [
    ("get", "/medicine/oncology", {}),
    ("get", "/medicine/oncology/add", {}),
    ("post", "/medicine/oncology/add", {"data": {}}),
    ("get", "/medicine/oncology/encounter/P0001", {}),
    ("post", "/medicine/oncology/encounter/P0001", {"data": {}}),
    ("get", "/medicine/oncology/ai_summary/P0001", {}),
    ("get", "/medicine/oncology/note/1/edit", {}),
    ("post", "/medicine/oncology/note/1/void", {"data": {}}),
    ("get", "/medicine/bookings/", {}),
    ("get", BOOK, {}),
    ("post", BOOK, {"data": {}}),
    ("get", "/medicine/prescriptions", {}),
    ("get", "/medicine/prescriptions/P0001", {}),
    ("get", "/medicine/prescriptions/new", {}),
    ("get", "/medicine/oncology/chemo-builder/P0001", {}),
    ("get", CALC, {}),
    ("post", SAVE, {"json": {}}),
    ("get", "/medicine/oncology/api/chemo-orders/P0001", {}),
    ("post", "/medicine/oncology/api/chemo-order/1/status", {"json": {}}),
    ("post", "/medicine/diseases/add", {"data": {}}),
    ("get", "/medicine/get_stages/1", {}),
]


class TestAccessControl:
    @pytest.mark.parametrize("method,url,kw", PROTECTED)
    def test_anonymous_is_bounced(self, client, patient, method, url, kw):
        r = getattr(client, method)(url, **kw)
        assert r.status_code in (301, 302, 401), (url, r.status_code)

    @pytest.mark.parametrize(
        "method,url,kw", [p for p in PROTECTED if "get_stages" not in p[1]]
    )
    def test_non_clinical_role_is_forbidden(self, client, patient, method, url, kw):
        _login(client, _mk_user("lab_tech"))
        r = getattr(client, method)(url, **kw)
        assert r.status_code == 403, (method, url, r.status_code)

    def test_lab_tech_cannot_enrol_a_patient(self, client, patient):
        _login(client, _mk_user("lab_tech"))
        client.post("/medicine/oncology/add", data=TestEnrolRoute.BASE)
        assert OncoPatient.query.count() == 0

    def test_nurse_can_read_the_chart_but_not_change_it(
        self, client, patient, cancer_ref
    ):
        nurse = _mk_user("nurse")
        _enrol()
        _login(client, nurse)
        assert client.get(_encounter()).status_code == 200
        r = client.post(
            _encounter(),
            data={
                "submit_note": "1",
                "note_date": "2026-04-04",
                "note_content": "nurse edit",
            },
        )
        assert r.status_code == 403 and OncologyNote.query.count() == 0

    def test_nurse_cannot_reach_the_chemo_order_builder_or_api(self, client, patient):
        _login(client, _mk_user("nurse"))
        assert client.get("/medicine/oncology/chemo-builder/P0001").status_code == 403
        assert client.post(SAVE, json=_save_body()).status_code == 403

    def test_get_stages_needs_a_session_but_any_role_may_read_reference_data(
        self, client, patient
    ):
        assert client.get("/medicine/get_stages/1").status_code in (301, 302, 401)
        _login(client, _mk_user("lab_tech"))
        assert client.get("/medicine/get_stages/1").status_code == 200
