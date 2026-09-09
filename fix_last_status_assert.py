#!/usr/bin/env python3
"""Fix the final assertion: SOAP submit with no pending work auto-closes the visit."""

p = "tests/test_patient_flow_integration.py"
with open(p, "r", encoding="utf-8") as f:
    content = f.read()

# Split on the LAST occurrence of the failing assertion
target = 'assert enc.status == "ACTIVE"'
parts = content.rsplit(target, 1)

if len(parts) == 2:
    # Rejoin with the corrected expectation for Phase 2 auto-closure
    new_content = parts[0] + 'assert enc.status == "DISCHARGED"' + parts[1]
    
    with open(p, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Fixed the final assertion. Run: pytest")
else:
    print("ERROR: Could not find the target assertion in the file.")