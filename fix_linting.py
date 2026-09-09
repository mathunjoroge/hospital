#!/usr/bin/env python3
"""
Fix PEP 8 style violations in test_phase2_encounter_stage.py
Splits multiple statements on one line (E702 errors) onto separate lines.
"""

def fix_file():
    file_path = "tests/test_phase2_encounter_stage.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix line 27: db.session.add(enc); db.session.commit()
    content = content.replace(
        "    db.session.add(enc); db.session.commit()",
        "    db.session.add(enc)\n    db.session.commit()"
    )
    
    # Fix line 38: db.session.add(enc); db.session.commit()
    # This pattern appears in test_stage_machine_refuses_illegal_jumps
    content = content.replace(
        "    db.session.add(enc); db.session.commit()\n    assert enc.set_stage(\"DISCHARGED\") is False",
        "    db.session.add(enc)\n    db.session.commit()\n    assert enc.set_stage(\"DISCHARGED\") is False"
    )
    
    # Fix line 59: lt = LabTest(test_name="CBC", cost=500); db.session.add(lt); db.session.commit()
    content = content.replace(
        '    lt = LabTest(test_name="CBC", cost=500); db.session.add(lt); db.session.commit()',
        '    lt = LabTest(test_name="CBC", cost=500)\n    db.session.add(lt)\n    db.session.commit()'
    )
    
    # Fix line 81: db.session.add(inv); db.session.commit()
    content = content.replace(
        "    db.session.add(inv); db.session.commit()\n    db.session.add(InvoiceLineItem",
        "    db.session.add(inv)\n    db.session.commit()\n    db.session.add(InvoiceLineItem"
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Fixed {file_path}")
    print("  - Removed semicolons and split multiple statements onto separate lines")

if __name__ == "__main__":
    fix_file()