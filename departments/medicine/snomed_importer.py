"""
SNOMED CT importer for loading SNOMED CT CORE subset data.
"""
import csv
import os
from typing import Dict, List

from flask import current_app

from departments.models.terminology import SnomedCode
from extensions import db


def load_snomed_from_csv(filepath: str) -> List[Dict]:
    """
    Load SNOMED CT codes from a CSV file.
    Expected columns: CODE, DESCRIPTION
    """
    codes = []
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            codes.append({
                'code': row['CODE'].strip(),
                'description': row['DESCRIPTION'].strip(),
            })
    return codes


def import_snomed_codes(filepath: str = None) -> int:
    """
    Import SNOMED CT codes into the database.
    If filepath is not provided, uses the environment variable SNOMED_CSV_PATH
    or a default location.
    Returns the number of codes imported.
    """
    if filepath is None:
        filepath = os.getenv('SNOMED_CSV_PATH', '/app/data/snomed_codes.csv')

    if not os.path.exists(filepath):
        current_app.logger.warning(f"SNOMED CSV file not found at {filepath}")
        return 0

    codes = load_snomed_from_csv(filepath)
    if not codes:
        current_app.logger.warning("No SNOMED codes found in the CSV file")
        return 0

    # Clear existing table (if any) and insert new codes
    try:
        num_deleted = db.session.query(SnomedCode).delete()
        current_app.logger.info(f"Deleted {num_deleted} existing SNOMED codes")
    except Exception as e:
        current_app.logger.error(f"Error deleting existing SNOMED codes: {e}")
        db.session.rollback()
        return 0

    try:
        for code_dict in codes:
            code = SnomedCode(**code_dict)
            db.session.add(code)
        db.session.commit()
        current_app.logger.info(f"Imported {len(codes)} SNOMED codes")
        return len(codes)
    except Exception as e:
        current_app.logger.error(f"Error importing SNOMED codes: {e}")
        db.session.rollback()
        return 0


if __name__ == '__main__':
    # For testing the importer directly
    import sys

    from app import app
    with app.app_context():
        imported = import_snomed_codes(sys.argv[1] if len(sys.argv) > 1 else None)
        print(f"Imported {imported} SNOMED codes")
