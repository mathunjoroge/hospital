#!/usr/bin/env python3
"""Fix Phase 3: add patient relationship to Encounter, fix test logic."""
import os, sys

ROOT = os.getcwd()

# 1. Add patient relationship to Encounter model
enc_path = os.path.join(ROOT, "departments/models/encounter.py")
src = open(enc_path, encoding="utf-8").read()

# Check if relationship already exists
if 'patient = db.relationship' not in src:
    # Add the relationship after the ended_at column
    src = src.replace(
        '    ended_at = db.Column(db.DateTime(timezone=True), nullable=True)\n',
        '    ended_at = db.Column(db.DateTime(timezone=True), nullable=True)\n'
        '\n'
        '    patient = db.relationship("Patient", foreign_keys="Encounter.patient_id",\n'
        '                              primaryjoin="Encounter.patient_id == Patient.patient_id",\n'
        '                              lazy="joined", viewonly=True)\n'
    )
    open(enc_path, "w", encoding="utf-8").write(src)
    print("patched  departments/models/encounter.py (added patient relationship)")
else:
    print("  [skip] encounter.py: patient relationship already exists")


# 2. Fix the test: P0002 needs to be triaged to enter medicine queue
test_path = os.path.join(ROOT, "tests/test_phase3_queue_service.py")
src = open(test_path, encoding="utf-8").read()

# Fix the test to triage P0002 before checking medicine queue
src = src.replace(
    'def test_medicine_queue_includes_waiting_and_in_consult(app):\n'
    '    _patient("P0001")\n'
    '    _patient("P0002")\n'
    '    ScheduleEngine().create_walk_in(patient_id="P0001")\n'
    '    ScheduleEngine().create_walk_in(patient_id="P0002")\n'
    '    ScheduleEngine().mark_triage_complete("P0001")\n'
    '    enc1 = Encounter.query.filter_by(patient_id="P0001").first()\n'
    '    enc1.set_stage("IN_CONSULTATION")\n'
    '    db.session.commit()\n'
    '    out = queue_service.queue_for("medicine")\n'
    '    ids = {e.patient_id for e in out}\n'
    '    assert "P0001" in ids and "P0002" in ids\n',
    'def test_medicine_queue_includes_waiting_and_in_consult(app):\n'
    '    _patient("P0001")\n'
    '    _patient("P0002")\n'
    '    ScheduleEngine().create_walk_in(patient_id="P0001")\n'
    '    ScheduleEngine().create_walk_in(patient_id="P0002")\n'
    '    # Triage both patients so they move from REGISTERED to WAITING_DOCTOR\n'
    '    ScheduleEngine().mark_triage_complete("P0001")\n'
    '    ScheduleEngine().mark_triage_complete("P0002")\n'
    '    enc1 = Encounter.query.filter_by(patient_id="P0001").first()\n'
    '    enc1.set_stage("IN_CONSULTATION")\n'
    '    db.session.commit()\n'
    '    out = queue_service.queue_for("medicine")\n'
    '    ids = {e.patient_id for e in out}\n'
    '    # P0001 is IN_CONSULTATION, P0002 is WAITING_DOCTOR - both in medicine queue\n'
    '    assert "P0001" in ids and "P0002" in ids\n'
)

open(test_path, "w", encoding="utf-8").write(src)
print("patched  tests/test_phase3_queue_service.py (fixed test logic)")

print("\nFixes applied. Run: pytest tests/test_phase3_queue_service.py -v")