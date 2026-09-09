#!/usr/bin/env python3
"""Fix both Phase 2 issues: route collision + migration import order."""
import os, sys

ROOT = os.getcwd()
if not os.path.exists(os.path.join(ROOT, "departments/appointments/engine.py")):
    sys.exit("Run from the repository root.")


def fix_migration():
    p = os.path.join(ROOT, "scripts/migrate_encounter_stage.py")
    src = open(p, encoding="utf-8").read()
    
    # Remove the broken prepended lines
    lines = src.splitlines(keepends=True)
    clean_lines = []
    skip_next = False
    for line in lines:
        if 'sys.path.insert(0, os.path.abspath' in line:
            continue  # skip this broken line
        if line.strip() == 'import sys':
            continue  # skip standalone import sys we added
        clean_lines.append(line)
    
    # Now insert sys.path logic AFTER "import os"
    result = []
    for line in clean_lines:
        result.append(line)
        if line.strip() == 'import os':
            result.append('import sys\n')
            result.append('sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))\n')
    
    open(p, "w", encoding="utf-8").write("".join(result))
    print("fixed  scripts/migrate_encounter_stage.py")


def fix_route_collision():
    p = os.path.join(ROOT, "departments/medicine/consultations.py")
    src = open(p, encoding="utf-8").read()
    
    # Rename the route from /discharge/<patient_id> to /visit-discharge/<patient_id>
    # and rename function from discharge_patient to manual_discharge_visit
    src = src.replace(
        '@bp.route("/discharge/<patient_id>", methods=["POST"])\n'
        "@login_required\n"
        '@roles_required("medicine", "admin")\n'
        "def discharge_patient(patient_id):\n",
        '@bp.route("/visit-discharge/<patient_id>", methods=["POST"])\n'
        "@login_required\n"
        '@roles_required("medicine", "admin")\n'
        "def manual_discharge_visit(patient_id):\n"
    )
    
    open(p, "w", encoding="utf-8").write(src)
    print("fixed  departments/medicine/consultations.py (route renamed)")


def fix_test_file():
    p = os.path.join(ROOT, "tests/test_phase2_encounter_stage.py")
    if not os.path.exists(p):
        return
    src = open(p, encoding="utf-8").read()
    # Update test to use the renamed endpoint
    src = src.replace(
        'resp = client.post("/medicine/discharge/P0001", follow_redirects=True)',
        'resp = client.post("/medicine/visit-discharge/P0001", follow_redirects=True)'
    )
    open(p, "w", encoding="utf-8").write(src)
    print("fixed  tests/test_phase2_encounter_stage.py")


fix_migration()
fix_route_collision()
fix_test_file()
print("\nBoth fixes applied. Run: python scripts/migrate_encounter_stage.py && pytest")