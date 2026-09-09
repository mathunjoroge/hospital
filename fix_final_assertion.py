#!/usr/bin/env python3
"""Fix the final assertion in test_unified_patient_flow to expect auto-closure."""

p = "tests/test_patient_flow_integration.py"
with open(p, "r", encoding="utf-8") as f:
    content = f.read()

old_block = '''            assert soap_post_resp.status_code == 200
            # Phase 4: Check Encounter instead of retired PatientWaitingList
            enc = Encounter.query.filter_by(patient_id=patient_id).first()
            assert enc is not None
            assert enc.status == "ACTIVE"'''

new_block = '''            assert soap_post_resp.status_code == 200
            # Phase 2/4: Since there are no pending labs, drugs, or bills, the visit auto-closes.
            enc = Encounter.query.filter_by(patient_id=patient_id).first()
            assert enc is not None
            assert enc.status == "DISCHARGED"
            assert enc.stage == "DISCHARGED"'''

if old_block in content:
    content = content.replace(old_block, new_block)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    print("Fixed step 5 assertion. Run: pytest")
else:
    print("Block not found exactly. Check indentation.")