"""T3.2 — SURGICAL encounter creation and stage transition tests."""
from departments.medicine.inpatients import (
    create_surgical_encounter,
    transition_surgical_stage,
)
from departments.models.user import User
from extensions import db


def _login(client, role="doctor"):
    user = User(username=f"test_{role}_{id(client)}", password="password", role=role)
    db.session.add(user)
    db.session.commit()
    with client.session_transaction() as sess:
        sess["_user_id"] = str(user.id)
        sess["_fresh"] = True


def test_create_surgical_encounter_starts_in_pre_op(client):
    """Ensure surgical encounters are created in PRE_OP stage."""
    _login(client)
    enc = create_surgical_encounter(patient_id="TEST_P123", provider_id="DOC1")
    db.session.commit()
    assert enc.encounter_type == "SURGICAL"
    assert enc.stage == "PRE_OP"
    assert enc.status == "ACTIVE"


def test_surgical_stage_transitions(client):
    """Ensure legal stage transitions work: PRE_OP -> INTRA_OP -> POST_OP."""
    _login(client)
    enc = create_surgical_encounter(patient_id="TEST_P124")
    db.session.commit()

    transition_surgical_stage(enc.id, "INTRA_OP")
    db.session.refresh(enc)
    assert enc.stage == "INTRA_OP"

    transition_surgical_stage(enc.id, "POST_OP")
    db.session.refresh(enc)
    assert enc.stage == "POST_OP"


def test_invalid_surgical_stage_transition(client):
    """Ensure jumping stages (PRE_OP -> POST_OP) is blocked by set_stage."""
    _login(client)
    enc = create_surgical_encounter(patient_id="TEST_P125")
    db.session.commit()

    # set_stage refuses illegal transitions silently (logs a warning, no exception).
    # Assert the stage is unchanged rather than expecting a raise.
    transition_surgical_stage(enc.id, "POST_OP")
    db.session.refresh(enc)
    assert enc.stage == "PRE_OP", f"Expected PRE_OP but got {enc.stage}"
