"""Idempotent migration: add encounters.stage and backfill legacy rows."""
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
