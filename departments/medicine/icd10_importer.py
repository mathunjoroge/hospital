"""
ICD-10 importer for loading WHO ICD-10-CM data from NLM UMLS flat file.
"""
import csv
import os
from typing import List, Dict

from flask import current_app
from extensions import db
from departments.models.terminology import ICD10Code


def load_icd10_from_csv(filepath: str) -> List[Dict]:
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


def import_icd10_codes(filepath: str = None) -> int:
    """
    Import ICD-10 codes into the database.
    If filepath is not provided, uses the environment variable ICD10_CSV_PATH
    or a default location.
    Returns the number of codes imported.
    """
    if filepath is None:
        filepath = os.getenv('ICD10_CSV_PATH', '/app/data/icd10_codes.csv')
    
    if not os.path.exists(filepath):
        current_app.logger.warning(f"ICD-10 CSV file not found at {filepath}")
        return 0

    codes = load_icd10_from_csv(filepath)
    if not codes:
        current_app.logger.warning("No ICD-10 codes found in the CSV file")
        return 0

    # Clear existing table (if any) and insert new codes
    # Note: In production, we might want to do a merge instead of truncate.
    # For simplicity, we truncate and reinsert.
    try:
        num_deleted = db.session.query(ICD10Code).delete()
        current_app.logger.info(f"Deleted {num_deleted} existing ICD-10 codes")
    except Exception as e:
        current_app.logger.error(f"Error deleting existing ICD-10 codes: {e}")
        db.session.rollback()
        return 0

    try:
        for code_dict in codes:
            code = ICD10Code(**code_dict)
            db.session.add(code)
        db.session.commit()
        current_app.logger.info(f"Imported {len(codes)} ICD-10 codes")
        return len(codes)
    except Exception as e:
        current_app.logger.error(f"Error importing ICD-10 codes: {e}")
        db.session.rollback()
        return 0


if __name__ == '__main__':
    # For testing the importer directly
    import sys
    from app import app
    with app.app_context():
        imported = import_icd10_codes(sys.argv[1] if len(sys.argv) > 1 else None)
        print(f"Imported {imported} ICD-10 codes")
