"""
departments/medicine/snomed_importer.py
────────────────────────────────────────
SNOMED CT import utilities.

Two entry-points:
  1. import_from_umls_api()  — Bulk populate SNOMED CT codes from UMLS REST API & core clinical dataset.
  2. import_snomed_codes()   — Legacy CSV file importer.
"""

import csv
import logging
import os

from departments.medicine.umls_client import (
    get_core_snomed_seed_dataset,
    search_snomed_live,
)
from departments.models.terminology import SnomedCode
from extensions import db

logger = logging.getLogger(__name__)

# Common clinical search terms to query UMLS API for rich SNOMED CT coverage
_SNOMED_SEARCH_TERMS = [
    "hypertension",
    "diabetes",
    "fever",
    "pneumonia",
    "malaria",
    "asthma",
    "tuberculosis",
    "appendicitis",
    "sepsis",
    "anemia",
    "infection",
    "headache",
    "chest pain",
    "cough",
    "dyspnea",
    "cesarean",
    "delivery",
    "fracture",
    "stroke",
    "heart failure",
    "kidney disease",
    "gastritis",
    "cancer",
    "carcinoma",
    "epilepsy",
    "migraine",
    "diarrhea",
    "vomiting",
]


def import_from_umls_api(fetch_live_api: bool = True) -> int:
    """
    Populate the local `snomed_codes` PostgreSQL table using the UMLS API & seed dataset.

    :param fetch_live_api: If True, also queries UMLS API for common clinical categories.
    :return: Number of SNOMED codes populated in local DB.
    """
    logger.info("Starting SNOMED CT database population via UMLS API / seed dataset …")
    codes_map: dict[str, str] = {}

    # 1. Load core seed dataset
    for item in get_core_snomed_seed_dataset():
        code = item["code"]
        desc = item["description"]
        if code and desc:
            codes_map[code] = desc

    # 2. Optionally fetch live UMLS API concepts for key clinical categories
    if fetch_live_api:
        logger.info("Querying UMLS REST API for additional clinical concepts …")
        for term in _SNOMED_SEARCH_TERMS:
            try:
                live_results = search_snomed_live(term, max_results=15)
                for res in live_results:
                    c_code = res.get("code")
                    c_desc = res.get("description")
                    if c_code and c_desc:
                        codes_map[c_code] = c_desc
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed live UMLS query for '%s': %s", term, e)

    if not codes_map:
        logger.warning("No SNOMED CT codes collected for import.")
        return 0

    # 3. Upsert into database in batches of 100
    try:
        existing_codes = {c.code: c for c in SnomedCode.query.all()}
        new_objects = []
        updated_count = 0

        for code, desc in codes_map.items():
            if code in existing_codes:
                if existing_codes[code].description != desc:
                    existing_codes[code].description = desc
                    updated_count += 1
            else:
                new_objects.append(SnomedCode(code=code, description=desc))

        if new_objects:
            batch_size = 100
            for i in range(0, len(new_objects), batch_size):
                db.session.add_all(new_objects[i : i + batch_size])
                db.session.commit()

        if updated_count > 0:
            db.session.commit()

        total = SnomedCode.query.count()
        logger.info("SNOMED CT import complete. DB now contains %d codes.", total)
        return len(codes_map)

    except Exception as e:  # noqa: BLE001
        db.session.rollback()
        logger.error("Error during SNOMED CT database import: %s", e)
        return 0


def load_snomed_from_csv(filepath: str) -> list[dict]:
    """Load SNOMED CT codes from a CSV file (Expected columns: CODE, DESCRIPTION)."""
    codes = []
    with open(filepath, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            codes.append(
                {
                    "code": row["CODE"].strip(),
                    "description": row["DESCRIPTION"].strip(),
                }
            )
    return codes


def import_snomed_codes(filepath: str | None = None) -> int:
    """Import SNOMED CT codes from a CSV file into the database."""
    if filepath is None:
        filepath = os.getenv("SNOMED_CSV_PATH", "/app/data/snomed_codes.csv")

    if not os.path.exists(filepath):
        logger.info(
            "SNOMED CSV file not found at %s. Falling back to UMLS API import.",
            filepath,
        )
        return import_from_umls_api(fetch_live_api=True)

    codes = load_snomed_from_csv(filepath)
    if not codes:
        return 0

    try:
        db.session.query(SnomedCode).delete()
        for code_dict in codes:
            db.session.add(SnomedCode(**code_dict))
        db.session.commit()
        return len(codes)
    except Exception as e:  # noqa: BLE001
        db.session.rollback()
        logger.error("Error importing SNOMED codes from CSV: %s", e)
        return 0


if __name__ == "__main__":
    import sys

    from app import app

    with app.app_context():
        if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
            imported = import_snomed_codes(sys.argv[1])
        else:
            imported = import_from_umls_api(fetch_live_api=True)
        print(f"Imported {imported} SNOMED codes into database.")
