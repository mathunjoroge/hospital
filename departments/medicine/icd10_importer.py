"""
departments/medicine/icd10_importer.py
────────────────────────────────────────
ICD-10 import utilities.

Two entry-points:

  import_from_who_api()   — walk the full WHO ICD-10 tree and upsert all
                            ~14,000 codes into the local `icd10_codes` table.
                            Requires WHO_ICD_CLIENT_ID / WHO_ICD_CLIENT_SECRET
                            in the environment (.env).

  import_icd10_codes(filepath) — legacy CSV importer (kept for backward
                                  compatibility with any existing scripts).

CLI usage (run from project root):
  python -m departments.medicine.icd10_importer

The walk takes 25–40 minutes on first run due to WHO rate limiting (~2 req/s).
Subsequent Celery-triggered nightly re-syncs only upsert changes and complete
in the same time window, but with minimal DB churn when codes are unchanged.
"""
import csv
import logging
import os

from departments.models.terminology import ICD10Code
from extensions import db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WHO API bulk import
# ---------------------------------------------------------------------------
def import_from_who_api(release: str | None = None) -> int:
    """
    Walk the full WHO ICD-10 tree for *release* and upsert every code into the
    local ``icd10_codes`` table.

    Returns the total number of codes upserted (inserted + updated).

    Must be called inside a Flask application context.
    """
    from departments.medicine.who_icd_client import walk_icd10_tree

    if release is None:
        release = os.getenv("WHO_ICD_API_RELEASE", "2019")

    logger.info("Starting WHO ICD-10 bulk import (release %s) …", release)
    print(f"[ICD-10 Import] Walking WHO ICD-10 {release} tree — this takes 25–40 minutes.")
    print("[ICD-10 Import] Progress reported every 500 codes.\n")

    total = 0
    batch: list[dict] = []
    BATCH_SIZE = 200

    def _flush_batch(batch: list[dict]) -> int:
        """Upsert a batch of code dicts into icd10_codes. Returns count upserted."""
        if not batch:
            return 0
        codes_in_batch = [r["code"] for r in batch]
        existing = {
            row.code: row
            for row in ICD10Code.query.filter(ICD10Code.code.in_(codes_in_batch)).all()
        }
        inserted = updated = 0
        for record in batch:
            code_val = record["code"][:20] if record.get("code") else ""
            ch_val = record["chapter"][:500] if record.get("chapter") else None
            bl_val = record["block"][:500] if record.get("block") else None
            if code_val in existing:
                row = existing[code_val]
                row.description = record["description"]
                row.chapter = ch_val
                row.block = bl_val
                updated += 1
            else:
                db.session.add(ICD10Code(
                    code=code_val,
                    description=record["description"],
                    chapter=ch_val,
                    block=bl_val,
                ))
                inserted += 1
        db.session.commit()
        return inserted + updated

    try:
        for code, title, chapter, block in walk_icd10_tree(release):
            batch.append({
                "code": code,
                "description": title,
                "chapter": chapter,
                "block": block,
            })
            total += 1

            if len(batch) >= BATCH_SIZE:
                _flush_batch(batch)
                batch = []

            if total % 500 == 0:
                print(f"[ICD-10 Import] {total:,} codes processed …")
                logger.info("ICD-10 import progress: %d codes", total)

        # Flush any remaining records
        if batch:
            _flush_batch(batch)

    except Exception as exc:  # noqa: BLE001
        logger.error("ICD-10 import failed after %d codes: %s", total, exc)
        db.session.rollback()
        raise

    logger.info("ICD-10 import complete: %d codes upserted.", total)
    print(f"\n[ICD-10 Import] ✓ Complete — {total:,} codes upserted into icd10_codes.")
    return total


# ---------------------------------------------------------------------------
# Legacy CSV importer (backward-compat)
# ---------------------------------------------------------------------------
def load_icd10_from_csv(filepath: str) -> list[dict]:
    """
    Load ICD-10 codes from a CSV file.
    Expected columns: CODE, DESCRIPTION, CHAPTER, BLOCK
    """
    codes = []
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            codes.append({
                'code': row['CODE'].strip(),
                'description': row['DESCRIPTION'].strip(),
                'chapter': row.get('CHAPTER', '').strip(),
                'block': row.get('BLOCK', '').strip(),
            })
    return codes


def import_icd10_codes(filepath: str | None = None) -> int:
    """
    Import ICD-10 codes into the database from a CSV file.
    If filepath is not provided, uses the environment variable ICD10_CSV_PATH
    or a default location.
    Returns the number of codes imported.

    Prefer import_from_who_api() for a complete, up-to-date code set.
    """
    from flask import current_app

    from departments.models.terminology import ICD10Code
    from extensions import db

    if filepath is None:
        filepath = os.getenv('ICD10_CSV_PATH', '/app/data/icd10_codes.csv')

    if not os.path.exists(filepath):
        current_app.logger.warning("ICD-10 CSV file not found at %s", filepath)
        return 0

    codes = load_icd10_from_csv(filepath)
    if not codes:
        current_app.logger.warning("No ICD-10 codes found in the CSV file")
        return 0

    try:
        num_deleted = db.session.query(ICD10Code).delete()
        current_app.logger.info("Deleted %d existing ICD-10 codes", num_deleted)
    except Exception as exc:  # noqa: BLE001
        current_app.logger.error("Error deleting existing ICD-10 codes: %s", exc)
        db.session.rollback()
        return 0

    try:
        for code_dict in codes:
            code = ICD10Code(**code_dict)
            db.session.add(code)
        db.session.commit()
        current_app.logger.info("Imported %d ICD-10 codes from CSV", len(codes))
        return len(codes)
    except Exception as exc:  # noqa: BLE001
        current_app.logger.error("Error importing ICD-10 codes: %s", exc)
        db.session.rollback()
        return 0


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys

    from app import app  # noqa: E402

    release = sys.argv[1] if len(sys.argv) > 1 else None
    with app.app_context():
        count = import_from_who_api(release)
        print(f"Done — {count:,} ICD-10 codes in database.")
