import subprocess
import sys
from pathlib import Path


def run_cmd(cmd):
    print(f"\n{'='*20} Executing: {cmd} {'='*20}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Command failed: {cmd}")
        sys.exit(result.returncode)
    print(f"✅ Command succeeded: {cmd}")

def fix_ui_summary_route():
    print("\n📝 Fixing imports and route in departments/records/routes.py...")
    file_path = Path('departments/records/routes.py')
    content = file_path.read_text(encoding='utf-8')

    # 1. Remove the badly placed imports if they exist
    content = content.replace("from collections import defaultdict\n", "")
    content = content.replace("from departments.models.encounter import Encounter\n", "")
    content = content.replace("from extensions import db\n", "")
    content = content.replace("from flask_login import login_required\n", "")

    # 2. Add them to the top of the file (after existing imports)
    lines = content.split('\n')
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('from ') or line.startswith('import '):
            insert_idx = i + 1

    # Ensure we don't duplicate
    if "from collections import defaultdict" not in content:
        lines.insert(insert_idx, "from collections import defaultdict")
        lines.insert(insert_idx + 1, "from departments.models.encounter import Encounter")
        lines.insert(insert_idx + 2, "from extensions import db")
        lines.insert(insert_idx + 3, "from flask_login import login_required")
        content = '\n'.join(lines)

    # 3. Ensure the route exists and is clean
    if "@bp.route(\"/api/active_encounters_summary\")" not in content:
        injection = """

@bp.route("/api/active_encounters_summary")
@login_required
def active_encounters_summary():
    \"\"\"
    Returns a grouped summary of all ACTIVE encounters by type and stage.
    \"\"\"
    active_encounters = Encounter.query.filter_by(status="ACTIVE").all()

    summary = defaultdict(lambda: defaultdict(int))
    total_active = 0

    for enc in active_encounters:
        enc_type = enc.encounter_type or "UNKNOWN"
        stage = enc.stage or "UNKNOWN"
        summary[enc_type][stage] += 1
        total_active += 1

    return {
        "total_active": total_active,
        "by_type": {k: dict(v) for k, v in summary.items()}
    }
"""
        # Append to the end of the file
        content += injection

    file_path.write_text(content, encoding='utf-8')
    print("✅ UI summary route and imports fixed.")
    return True

def fix_ui_summary_test():
    print("\n📝 Fixing test for active encounters summary...")
    test_file = Path('tests/test_ui_encounter_summary.py')

    test_code = """import pytest
from departments.models.encounter import Encounter
from extensions import db

def test_active_encounters_summary_groups_by_type(client):
    \"\"\"Ensure the summary endpoint correctly groups active encounters.\"\"\"
    # 1. Log in as admin (follow redirects to ensure session is set)
    client.post("/login", data={"username": "admin", "password": "AdminPassword123!"}, follow_redirects=True)

    # 2. Create diverse active encounters
    enc1 = Encounter(patient_id="TEST_P1", encounter_type="SURGICAL", stage="PRE_OP", status="ACTIVE")
    enc2 = Encounter(patient_id="TEST_P2", encounter_type="SURGICAL", stage="INTRA_OP", status="ACTIVE")
    enc3 = Encounter(patient_id="TEST_P3", encounter_type="TELEHEALTH", stage="IN_CONSULTATION", status="ACTIVE")
    enc4 = Encounter(patient_id="TEST_P4", encounter_type="OPD", stage="WAITING_DOCTOR", status="ACTIVE")

    db.session.add_all([enc1, enc2, enc3, enc4])
    db.session.commit()

    # 3. Call the endpoint
    response = client.get("/records/api/active_encounters_summary", follow_redirects=True)

    # If it still redirects, it means login failed. Let's assert 200 to catch that.
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.data}"

    data = response.get_json()
    assert data["total_active"] >= 4
    assert data["by_type"]["SURGICAL"]["PRE_OP"] >= 1
    assert data["by_type"]["SURGICAL"]["INTRA_OP"] >= 1
    assert data["by_type"]["TELEHEALTH"]["IN_CONSULTATION"] >= 1

    # 4. Cleanup
    db.session.delete(enc1)
    db.session.delete(enc2)
    db.session.delete(enc3)
    db.session.delete(enc4)
    db.session.commit()
"""
    test_file.write_text(test_code, encoding='utf-8')
    print("✅ UI summary test fixed.")
    return True

if __name__ == "__main__":
    print("🚀 UI Wiring: Fix Active Encounters Summary...")

    if not fix_ui_summary_route():
        sys.exit(1)
    if not fix_ui_summary_test():
        sys.exit(1)

    run_cmd("python -m pytest tests/test_ui_encounter_summary.py -v")
    run_cmd("python -m ruff check . --fix")
    run_cmd("python -m ruff check .")
    run_cmd("python -m pytest -q")

    run_cmd("git add departments/records/routes.py tests/test_ui_encounter_summary.py")
    run_cmd('git commit -m "fix: UI Wiring - Correct import order and test login for active encounters summary"')

    print("\n🎉 UI Wiring fixed. Tests + ruff green, committed.")
