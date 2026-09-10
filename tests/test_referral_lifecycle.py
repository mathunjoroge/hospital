"""
tests/test_referral_lifecycle.py
─────────────────────────────────
T3.3 — Referral Lifecycle: source encounter transitions to REFERRED_OUT
and a new REFERRAL encounter is opened for the receiving facility.
"""
from datetime import datetime, timezone

import pytest

from departments.models.encounter import Encounter
from departments.models.records import Patient, PatientWaitingList
from departments.referrals.engine import ReferralEngine, ReferralStatus
from departments.shared.queue_constants import QueueStatus
from extensions import db


# ── helpers ────────────────────────────────────────────────────────


def _patient_with_encounter(patient_id: str, stage: str = "IN_CONSULTATION"):
    """Create a patient with an active OPD encounter at the given stage."""
    p = Patient(patient_id=patient_id, name=f"Test {patient_id}", sex="M",
                date_of_birth=datetime(1985, 6, 15))
    db.session.add(p)
    db.session.add(PatientWaitingList(patient_id=patient_id, seen=QueueStatus.WAITING_TRIAGE))
    enc = Encounter(
        patient_id=patient_id,
        encounter_type="OPD",
        stage=stage,
        status="ACTIVE",
    )
    db.session.add(enc)
    db.session.commit()
    return p, enc


# ── T3.3 tests ─────────────────────────────────────────────────────


def test_referral_acceptance_marks_source_as_referred_out(app):
    """Accepting a referral should set the source encounter stage to REFERRED_OUT."""
    with app.app_context():
        _, src_enc = _patient_with_encounter("P-REF-01")
        src_id = src_enc.id

        engine = ReferralEngine()
        referral = engine.initiate(
            patient_id="P-REF-01",
            referring_facility="Kenyatta National Hospital",
            receiving_facility="Moi Teaching & Referral Hospital",
            reason="Specialist surgical review",
            clinical_summary="Patient requires cardiothoracic assessment.",
        )

        engine.update_status(referral.id, ReferralStatus.ACCEPTED)

        updated_src = Encounter.query.get(src_id)
        assert updated_src.stage == "REFERRED_OUT", (
            f"Expected REFERRED_OUT, got {updated_src.stage}"
        )
        # status is DISCHARGED because we used close()
        assert updated_src.status == "DISCHARGED"


def test_referral_acceptance_creates_receiving_encounter(app):
    """Accepting a referral must create a new REFERRAL encounter for the patient."""
    with app.app_context():
        _patient_with_encounter("P-REF-02")

        engine = ReferralEngine()
        referral = engine.initiate(
            patient_id="P-REF-02",
            referring_facility="Coast General Hospital",
            receiving_facility="Aga Khan Hospital Nairobi",
            reason="Oncology second opinion",
            clinical_summary="Stage II breast cancer — needs multidisciplinary review.",
        )

        engine.update_status(referral.id, ReferralStatus.ACCEPTED)

        receiving_enc = Encounter.query.filter_by(
            patient_id="P-REF-02", encounter_type="REFERRAL"
        ).first()

        assert receiving_enc is not None, "Receiving REFERRAL encounter was not created"
        assert receiving_enc.stage == "IN_CONSULTATION"
        assert receiving_enc.status == "ACTIVE"
        assert receiving_enc.chief_complaint == "Oncology second opinion"


def test_referral_rejection_does_not_change_encounter(app):
    """Rejecting a referral must NOT close the source encounter."""
    with app.app_context():
        _, src_enc = _patient_with_encounter("P-REF-03")
        src_id = src_enc.id
        original_stage = src_enc.stage

        engine = ReferralEngine()
        referral = engine.initiate(
            patient_id="P-REF-03",
            referring_facility="Nakuru Level 5",
            receiving_facility="Nairobi Hospital",
            reason="Post-op complication",
            clinical_summary="Wound dehiscence post-appendectomy.",
        )

        engine.update_status(referral.id, ReferralStatus.REJECTED)

        src_after = Encounter.query.get(src_id)
        assert src_after.stage == original_stage, (
            "Rejection must not alter the source encounter stage"
        )
        # No REFERRAL encounter should be created
        ref_enc = Encounter.query.filter_by(
            patient_id="P-REF-03", encounter_type="REFERRAL"
        ).first()
        assert ref_enc is None, "No receiving encounter should be created on rejection"


def test_referral_acceptance_no_active_encounter_is_safe(app):
    """Accepting a referral when no active source encounter exists must not crash."""
    with app.app_context():
        # Patient exists but has no encounter
        p = Patient(patient_id="P-REF-04", name="No Enc", sex="F",
                    date_of_birth=datetime(1992, 3, 1))
        db.session.add(p)
        db.session.commit()

        engine = ReferralEngine()
        referral = engine.initiate(
            patient_id="P-REF-04",
            referring_facility="Sub-district clinic",
            receiving_facility="County Referral",
            reason="Complicated delivery",
            clinical_summary="Prolonged labour — needs theatre.",
        )

        # Should not raise
        result = engine.update_status(referral.id, ReferralStatus.ACCEPTED)
        assert result is not None
        assert result.status == ReferralStatus.ACCEPTED

        # Receiving encounter still created
        receiving_enc = Encounter.query.filter_by(
            patient_id="P-REF-04", encounter_type="REFERRAL"
        ).first()
        assert receiving_enc is not None


def test_invalid_referral_status_raises(app):
    """Passing an unknown status string must raise ValueError."""
    with app.app_context():
        engine = ReferralEngine()
        referral = engine.initiate(
            patient_id="P-REF-05",
            referring_facility="A", receiving_facility="B",
            reason="Test", clinical_summary="Test",
        )
        with pytest.raises(ValueError, match="Invalid referral status"):
            engine.update_status(referral.id, "NONSENSE")
