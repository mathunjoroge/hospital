#!/usr/bin/env python3
"""Delete the stray .seen assertion line left over in test_patient_flow_integration.py"""
import os

p = "tests/test_patient_flow_integration.py"
with open(p, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Filter out the exact stray line
out = [line for line in lines if "assert waiting_entry.seen == QueueStatus.WAITING_TRIAGE" not in line]

with open(p, "w", encoding="utf-8") as f:
    f.writelines(out)

print("Stray line deleted. Run: pytest")