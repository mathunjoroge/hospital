"""
LOINC importer for loading LOINC codes.
"""
import csv
import os
from typing import Dict, List

from extensions import db
from flask import current_app

from departments.models.terminology import LoincCode


def load_loinc_from_csv(filepath: str) -> List[Dict]:
    """
    Load LOINC codes from a CSV file.
    Expected columns: LOINC_NUM, LONG_COMMON_NAME
    """
    codes = []
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            codes.append({
                'code': row['LOINC_NUM'].strip(),
                'description': row['LONG_COMMON_NAME'].strip(),
            })
    return codes


def import_loinc_codes(filepath: str = None) -> int:
    """
    Import LOINC codes into the database.
    If filepath is not provided, uses the environment variable LOINC_CSV_PATH
    or a default location.
    Returns the number of codes imported.
    """
    if filepath is None:
        filepath = os.getenv('LOINC_CSV_PATH', '/app/data/loinc_codes.csv')

    if not os.path.exists(filepath):
        current_app.logger.warning(f"LOINC CSV file not found at {filepath}")
        return 0

    codes = load_loinc_from_csv(filepath)
    if not codes:
        current_app.logger.warning("No LOINC codes found in the CSV file")
        return 0

    # Clear existing table (if any) and insert new codes
    try:
        num_deleted = db.session.query(LoincCode).delete()
        current_app.logger.info(f"Deleted {num_deleted} existing LOINC codes")
    except Exception as e:
        current_app.logger.error(f"Error deleting existing LOINC codes: {e}")
        db.session.rollback()
        return 0

    try:
        for code_dict in codes:
            code = LoincCode(**code_dict)
            db.session.add(code)
        db.session.commit()
        current_app.logger.info(f"Imported {len(codes)} LOINC codes")
        return len(codes)
    except Exception as e:
        current_app.logger.error(f"Error importing LOINC codes: {e}")
        db.session.rollback()
        return 0


if __name__ == '__main__':
    # For testing the importer directly
    import sys
    from app import app
    with app.app_context():
        imported = import_loinc_codes(sys.argv[1] if len(sys.argv) > 1 else None)
        print(f"Imported {imported} LOINC codes")
