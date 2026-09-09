#!/usr/bin/env python3
"""Final Phase 2 fix: set stage='REGISTERED' in create_walk_in Encounter constructor."""
import os, re

ROOT = os.getcwd()
engine_path = os.path.join(ROOT, "departments/appointments/engine.py")

src = open(engine_path, encoding="utf-8").read()

# Use regex to find the Encounter(...) call inside create_walk_in and inject stage="REGISTERED"
# We look for: enc = Encounter( ... status="ACTIVE" ... )
pattern = r'(enc = Encounter\([\s\S]*?status="ACTIVE")'
replacement = r'\1,\n            stage="REGISTERED"'

new_src, count = re.subn(pattern, replacement, src, count=1)

if count == 0:
    # Try alternative: maybe it's already there?
    if 'stage="REGISTERED"' in src:
        print("  [skip] engine.py: stage='REGISTERED' already present")
    else:
        print("FATAL: Could not find Encounter constructor in create_walk_in")
        print("Manual fix needed in departments/appointments/engine.py:")
        print("  Add stage='REGISTERED' to the Encounter() call in create_walk_in()")
else:
    open(engine_path, "w", encoding="utf-8").write(new_src)
    print("patched  departments/appointments/engine.py (create_walk_in sets stage='REGISTERED')")

print("\nDone. Run: pytest tests/test_phase2_encounter_stage.py -v")