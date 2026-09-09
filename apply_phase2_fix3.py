#!/usr/bin/env python3
"""Fix the migration script imports and the Phase 2 test date type."""
import os

ROOT = os.getcwd()

# 1. Fix migration script: ensure os and sys are imported before use
mig_path = os.path.join(ROOT, "scripts/migrate_encounter_stage.py")
mig_content = '''"""Idempotent migration: add encounters.stage and backfill legacy rows."""
import os
import sys

# Add repo root to sys.path so we can import app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import inspect, text

from app import app
from extensions import db


def migrate():
    with app.app_context():
        insp = inspect(db.engine)
        cols = [c["name"] for c in insp.get_columns("encounters")]
        with db.engine.begin() as conn:
            if "stage" not in cols:
                conn.execute(text("ALTER TABLE encounters ADD COLUMN stage VARCHAR(30)"))
                print("added encounters.stage")
            else:
                print("encounters.stage already present")
            conn.execute(text(
                "UPDATE encounters SET stage = 'AWAITING_BILLING' "
                "WHERE stage IS NULL AND status = 'ACTIVE'"
            ))
            conn.execute(text(
                "UPDATE encounters SET stage = 'DISCHARGED' "
                "WHERE stage IS NULL AND status <> 'ACTIVE'"
            ))
        print("migration complete")


if __name__ == "__main__":
    migrate()
'''
with open(mig_path, "w", encoding="utf-8") as f:
    f.write(mig_content)
print("fixed  scripts/migrate_encounter_stage.py")

# 2. Fix test file: use date object instead of string for date_of_birth
test_path = os.path.join(ROOT, "tests/test_phase2_encounter_stage.py")
src = open(test_path, encoding="utf-8").read()

if "from datetime import date" not in src:
    src = src.replace(
        "from werkzeug.security import generate_password_hash",
        "from datetime import date\nfrom werkzeug.security import generate_password_hash"
    )

src = src.replace(
    'date_of_birth="1990-01-01"',
    'date_of_birth=date(1990, 1, 1)'
)

with open(test_path, "w", encoding="utf-8") as f:
    f.write(src)
print("fixed  tests/test_phase2_encounter_stage.py")

print("\nDone. Run: python scripts/migrate_encounter_stage.py && pytest")