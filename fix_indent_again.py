#!/usr/bin/env python3
"""Fix indentation for the Phase 4 replacement blocks."""

with open("tests/test_patient_flow_integration.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    stripped = line.lstrip()
    # Target the exact lines we injected in the last script
    if stripped.startswith("# Phase 4: Check Encounter") or \
       stripped.startswith("enc = Encounter.query.filter_by") or \
       stripped.startswith("assert enc is not None") or \
       stripped.startswith('assert enc.status == "ACTIVE"'):
        # Force exactly 8 spaces of indentation (correct for inside `with app.app_context():`)
        new_lines.append("        " + stripped)
    else:
        new_lines.append(line)

with open("tests/test_patient_flow_integration.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Indentation fixed. Run: pytest")