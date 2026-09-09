#!/usr/bin/env python3
"""Fix the broken Patient Kardex link in the nursing vitals template."""
import os, sys

ROOT = os.getcwd()
if not os.path.exists(os.path.join(ROOT, "departments/appointments/engine.py")):
    sys.exit("Run from the repository root.")

tpl = "departments/nursing/templates/nursing/vitals.html"
p = os.path.join(ROOT, tpl)
src = open(p, encoding="utf-8").read()

broken = '{{ url_for(\'nursing.patient_dashboard\') }}?patient_id={{ patient_id }}'
fixed  = '{{ url_for(\'records.patient_profile\', patient_id=patient_id) }}'

count = src.count(broken)
if count == 0:
    print(f"[skip] {tpl}: broken anchor not found (already patched?)")
    sys.exit(0)
if count != 1:
    sys.exit(f"FATAL: {tpl}: broken anchor found {count}x, expected 1. Aborting.")

new_src = src.replace(broken, fixed, 1)
open(p, "w", encoding="utf-8").write(new_src)
print(f"patched  {tpl}")

# Verify the fix
verify = open(p, encoding="utf-8").read()
assert fixed in verify and "patient_dashboard" not in verify, "Verification failed."
print("verified: broken URL removed, patient_profile link in place.")