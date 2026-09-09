#!/usr/bin/env python3
"""Final fix: update test_unified_patient_flow to use Encounter instead of PatientWaitingList."""
import os

ROOT = os.getcwd()
p = os.path.join(ROOT, "tests/test_patient_flow_integration.py")

if not os.path.exists(p):
    print(f"FATAL: {p} not found")
    exit(1)

src = open(p, encoding="utf-8").read()

# The exact failing assertion block from the traceback
old_block = """            # Check PatientWaitingList entry
            waiting_entry = PatientWaitingList.query.filter_by(patient_id=patient_id).first()
            assert waiting_entry is not None"""

new_block = """            # Phase 4: PatientWaitingList is retired. Check Encounter instead.
            from departments.models.encounter import Encounter
            enc = Encounter.query.filter_by(patient_id=patient_id).first()
            assert enc is not None, "Encounter should be created on registration"
            assert enc.stage == "REGISTERED", "New encounter should start at REGISTERED stage"
            
            # Legacy table should no longer be written to
            waiting_entry = PatientWaitingList.query.filter_by(patient_id=patient_id).first()
            assert waiting_entry is None, "PatientWaitingList should not be created in Phase 4" """

if old_block in src:
    src = src.replace(old_block, new_block)
    open(p, "w", encoding="utf-8").write(src)
    print("patched  tests/test_patient_flow_integration.py")
else:
    # Fallback: try a looser match
    import re
    pattern = r'# Check PatientWaitingList entry\s+waiting_entry = PatientWaitingList\.query\.filter_by\(patient_id=patient_id\)\.first\(\)\s+assert waiting_entry is not None'
    match = re.search(pattern, src)
    if match:
        src = src[:match.start()] + new_block.strip() + src[match.end():]
        open(p, "w", encoding="utf-8").write(src)
        print("patched  tests/test_patient_flow_integration.py (regex fallback)")
    else:
        print("FATAL: could not find the failing assertion block")
        print("Manual edit needed in tests/test_patient_flow_integration.py around line 61")
        exit(1)

print("\nDone. Run: pytest")