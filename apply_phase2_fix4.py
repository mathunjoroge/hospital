#!/usr/bin/env python3
"""Final Phase 2 fixes: set initial stage in create_walk_in, fix LabTest test data."""
import os

ROOT = os.getcwd()

# 1. Fix create_walk_in to set stage="REGISTERED"
engine_path = os.path.join(ROOT, "departments/appointments/engine.py")
src = open(engine_path, encoding="utf-8").read()

old_create = '''        enc = Encounter(
            patient_id=str(patient_id),
            appointment_id=appt.id,
            provider_id=str(provider_id),
            encounter_type="OPD",
            status="ACTIVE",
        )
        db.session.add(enc)
        db.session.commit()
        logger.info("WALK-IN APPOINTMENT & ENCOUNTER CREATED: Patient %s", patient_id)'''

new_create = '''        enc = Encounter(
            patient_id=str(patient_id),
            appointment_id=appt.id,
            provider_id=str(provider_id),
            encounter_type="OPD",
            status="ACTIVE",
            stage="REGISTERED",
        )
        db.session.add(enc)
        db.session.commit()
        logger.info("WALK-IN APPOINTMENT & ENCOUNTER CREATED: Patient %s", patient_id)'''

if old_create in src:
    src = src.replace(old_create, new_create)
    open(engine_path, "w", encoding="utf-8").write(src)
    print("patched  departments/appointments/engine.py (create_walk_in sets stage)")
else:
    print("  [skip] engine.py: anchor not found")

# 2. Fix test data: LabTest requires cost
test_path = os.path.join(ROOT, "tests/test_phase2_encounter_stage.py")
src = open(test_path, encoding="utf-8").read()

# Fix both LabTest creations to include cost
src = src.replace(
    'lt = LabTest(test_name="CBC"); db.session.add(lt); db.session.commit()',
    'lt = LabTest(test_name="CBC", cost=500); db.session.add(lt); db.session.commit()'
)

open(test_path, "w", encoding="utf-8").write(src)
print("patched  tests/test_phase2_encounter_stage.py (LabTest cost)")

print("\nFinal fixes applied. Run: pytest tests/test_phase2_encounter_stage.py -v")