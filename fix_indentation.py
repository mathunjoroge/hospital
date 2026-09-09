#!/usr/bin/env python3
"""Fix the IndentationError in test_patient_flow_integration.py"""
import os

p = "tests/test_patient_flow_integration.py"
with open(p, "r", encoding="utf-8") as f:
    lines = f.readlines()

out = []
in_block = False

for line in lines:
    if "Phase 4: PatientWaitingList is retired" in line:
        in_block = True
        
    if in_block and line.strip():
        # Force exactly 8 spaces of indentation (correct for inside `with app.app_context():`)
        out.append("        " + line.lstrip())
    else:
        out.append(line)
        
    if "assert waiting_entry is None" in line:
        in_block = False

with open(p, "w", encoding="utf-8") as f:
    f.writelines(out)

print("Indentation fixed. Run: pytest")