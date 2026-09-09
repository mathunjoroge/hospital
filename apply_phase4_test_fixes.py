#!/usr/bin/env python3
"""Update legacy tests to align with Phase 4 (Encounter as source of truth)."""
import os

ROOT = os.getcwd()

# 1. Fix test_patient_flow_integration.py
p1 = os.path.join(ROOT, "tests/test_patient_flow_integration.py")
if os.path.exists(p1):
    src = open(p1, encoding="utf-8").read()
    
    old_assertion = '''            # Check PatientWaitingList entry
            waiting_entry = PatientWaitingList.query.filter_by(patient_id=patient_id).first()
            assert waiting_entry is not None'''
            
    new_assertion = '''            # Phase 4: PatientWaitingList is retired. Check Encounter instead.
            from departments.models.encounter import Encounter
            enc = Encounter.query.filter_by(patient_id=patient_id).first()
            assert enc is not None
            assert enc.stage == "REGISTERED"
            
            # Legacy table should no longer be written to during registration
            waiting_entry = PatientWaitingList.query.filter_by(patient_id=patient_id).first()
            assert waiting_entry is None'''
            
    if old_assertion in src:
        src = src.replace(old_assertion, new_assertion)
        open(p1, "w", encoding="utf-8").write(src)
        print("patched  tests/test_patient_flow_integration.py")
    else:
        print("  [skip] test_patient_flow_integration.py: anchor not found")


# 2. Fix test_phase2_encounter_stage.py
p2 = os.path.join(ROOT, "tests/test_phase2_encounter_stage.py")
if os.path.exists(p2):
    src = open(p2, encoding="utf-8").read()
    
    # Fix the helper: stop creating legacy PatientWaitingList rows
    old_helper = '''def _patient(pid):
    p = Patient(
        patient_id=pid, name=f"Test {pid}", sex="F",
        date_of_birth=date(1990, 1, 1),
    )
    db.session.add(p)
    db.session.add(PatientWaitingList(patient_id=pid, seen=QueueStatus.WAITING_TRIAGE))
    db.session.commit()
    return p'''
    
    new_helper = '''def _patient(pid):
    p = Patient(
        patient_id=pid, name=f"Test {pid}", sex="F",
        date_of_birth=date(1990, 1, 1),
    )
    db.session.add(p)
    # Phase 4: PatientWaitingList is retired; Encounter is the source of truth.
    db.session.commit()
    return p'''
    
    if old_helper in src:
        src = src.replace(old_helper, new_helper)
    
    # Fix the closure test: assert Encounter stage, not legacy seen integer
    old_closure = '''def test_visit_closes_when_no_work_no_debt(app):
    _patient("P0001")
    ScheduleEngine().create_walk_in(patient_id="P0001")
    assert visit_closure.maybe_close_encounter("P0001") is True
    enc = Encounter.query.filter_by(patient_id="P0001").first()
    assert enc.status == "DISCHARGED" and enc.ended_at is not None
    entry = PatientWaitingList.query.filter_by(patient_id="P0001").first()
    assert entry.seen == QueueStatus.DISCHARGED'''
    
    new_closure = '''def test_visit_closes_when_no_work_no_debt(app):
    _patient("P0001")
    ScheduleEngine().create_walk_in(patient_id="P0001")
    assert visit_closure.maybe_close_encounter("P0001") is True
    enc = Encounter.query.filter_by(patient_id="P0001").first()
    assert enc.status == "DISCHARGED" and enc.ended_at is not None
    # Phase 4: Encounter stage is the source of truth for closure
    assert enc.stage == "DISCHARGED"'''
    
    if old_closure in src:
        src = src.replace(old_closure, new_closure)
        
    open(p2, "w", encoding="utf-8").write(src)
    print("patched  tests/test_phase2_encounter_stage.py")


print("\nTest fixes applied. Run: pytest")