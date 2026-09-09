#!/usr/bin/env python3
"""Replace all remaining PatientWaitingList assertions in the integration test."""
import re

p = "tests/test_patient_flow_integration.py"
with open(p, "r", encoding="utf-8") as f:
    content = f.read()

# Regex to catch any remaining `waiting_entry = PatientWaitingList...` followed by `assert waiting_entry.seen == ...`
pattern = r'waiting_entry = PatientWaitingList\.query\.filter_by\(patient_id=patient_id\)\.first\(\)\s+assert waiting_entry\.seen == QueueStatus\.\w+'

replacement = (
    '# Phase 4: Check Encounter instead of retired PatientWaitingList\n'
    '            enc = Encounter.query.filter_by(patient_id=patient_id).first()\n'
    '            assert enc is not None\n'
    '            assert enc.status == "ACTIVE"'
)

new_content, count = re.subn(pattern, replacement, content)

if count > 0:
    with open(p, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Replaced {count} legacy assertion(s). Run: pytest")
else:
    print("No matching legacy assertions found. The file might have slightly different formatting.")
    print("Manual check required.")