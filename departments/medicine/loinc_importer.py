"""
departments/medicine/loinc_importer.py
───────────────────────────────────────
LOINC import utilities.

Two entry-points:
  1. import_from_umls_api()  — Bulk populate LOINC codes from UMLS REST API & core lab dataset.
  2. import_loinc_codes()    — Legacy CSV file importer.
"""

import csv
import logging
import os
from flask import current_app

from departments.models.terminology import LoincCode
from extensions import db
from departments.medicine.umls_client import get_core_loinc_seed_dataset, search_loinc_live

logger = logging.getLogger(__name__)

# Common laboratory search terms to query UMLS API for rich LOINC coverage
_LOINC_SEARCH_TERMS = [
    "glucose", "hemoglobin", "creatinine", "temperature", "blood pressure",
    "heart rate", "leukocytes", "platelets", "bilirubin", "cholesterol",
    "sodium", "potassium", "chloride", "malaria", "hiv", "urinalysis",
    "troponin", "viral load", "cd4", "urea", "alt", "ast", "tb"
]


def import_from_umls_api(fetch_live_api: bool = True) -> int:
    """
    Populate the local `loinc_codes` PostgreSQL table using the UMLS API & seed dataset.

    :param fetch_live_api: If True, also queries UMLS API for common laboratory categories.
    :return: Number of LOINC codes populated in local DB.
    """
    logger.info("Starting LOINC database population via UMLS API / seed dataset …")
    codes_map: dict[str, str] = {}

    # 1. Load core seed dataset
    for item in get_core_loinc_seed_dataset():
        code = item["code"]
        desc = item["description"]
        if code and desc:
            codes_map[code] = desc

    # 2. Optionally fetch live UMLS API concepts for key lab categories
    if fetch_live_api:
        logger.info("Querying UMLS REST API for additional laboratory concepts …")
        for term in _LOINC_SEARCH_TERMS:
            try:
                live_results = search_loinc_live(term, max_results=15)
                for res in live_results:
                    c_code = res.get("code")
                    c_desc = res.get("description")
                    if c_code and c_desc:
                        codes_map[c_code] = c_desc
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed live UMLS query for LOINC '%s': %s", term, e)

    if not codes_map:
        logger.warning("No LOINC codes collected for import.")
        return 0

    # 3. Upsert into database in batches of 100
    try:
        existing_codes = {c.code: c for c in LoincCode.query.all()}
        new_objects = []
        updated_count = 0

        for code, desc in codes_map.items():
            if code in existing_codes:
                if existing_codes[code].description != desc:
                    existing_codes[code].description = desc
                    updated_count += 1
            else:
                new_objects.append(LoincCode(code=code, description=desc))

        if new_objects:
            batch_size = 100
            for i in range(0, len(new_objects), batch_size):
                db.session.add_all(new_objects[i : i + batch_size])
                db.session.commit()

        if updated_count > 0:
            db.session.commit()

        total = LoincCode.query.count()
        logger.info("LOINC import complete. DB now contains %d codes.", total)
        return len(codes_map)

    except Exception as e:  # noqa: BLE001
        db.session.rollback()
        logger.error("Error during LOINC database import: %s", e)
        return 0


def load_loinc_from_csv(filepath: str) -> list[dict]:
    """Load LOINC codes from a CSV file (Expected columns: LOINC_NUM, LONG_COMMON_NAME)."""
    codes = []
    with open(filepath, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            codes.append({
                "code": row["LOINC_NUM"].strip(),
                "description": row["LONG_COMMON_NAME"].strip(),
            })
    return codes


def import_loinc_codes(filepath: str | None = None) -> int:
    """Import LOINC codes from a CSV file into the database."""
    if filepath is None:
        filepath = os.getenv("LOINC_CSV_PATH", "/app/data/loinc_codes.csv")

    if not os.path.exists(filepath):
        logger.info("LOINC CSV file not found at %s. Falling back to UMLS API import.", filepath)
        return import_from_umls_api(fetch_live_api=True)

    codes = load_loinc_from_csv(filepath)
    if not codes:
        return 0

    try:
        db.session.query(LoincCode).delete()
        for code_dict in codes:
            db.session.add(LoincCode(**code_dict))
        db.session.commit()
        return len(codes)
    except Exception as e:  # noqa: BLE001
        db.session.rollback()
        logger.error("Error importing LOINC codes from CSV: %s", e)
        return 0


if __name__ == "__main__":
    import sys
    from app import app

    with app.app_context():
        if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
            imported = import_loinc_codes(sys.argv[1])
        else:
            imported = import_from_umls_api(fetch_live_api=True)
        print(f"Imported {imported} LOINC codes into database.")
