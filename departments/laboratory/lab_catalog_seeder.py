"""
departments/laboratory/lab_catalog_seeder.py
─────────────────────────────────────────────
Seeds standard clinical lab tests into `labtests` table with LOINC mapping
and standard result parameters (`labresults_templates`).
"""

import logging
from typing import Any

from departments.models.laboratory import LabResultTemplate
from departments.models.medicine import LabTest
from extensions import db

logger = logging.getLogger(__name__)

STANDARD_LAB_TESTS: list[dict[str, Any]] = [
    {
        "test_name": "Full Blood Count (FBC/CBC)",
        "loinc_code": "58410-2",
        "cost": 1500.0,
        "description": "Complete Blood Count including WBC, RBC, Hemoglobin, Hematocrit, and Platelets.",
        "parameters": [
            {"parameter_name": "Hemoglobin", "normal_range_low": 12.0, "normal_range_high": 17.5, "unit": "g/dL"},
            {"parameter_name": "Leukocytes (WBC)", "normal_range_low": 4.0, "normal_range_high": 11.0, "unit": "10^3/uL"},
            {"parameter_name": "Platelets", "normal_range_low": 150.0, "normal_range_high": 450.0, "unit": "10^3/uL"},
            {"parameter_name": "Hematocrit (HCT)", "normal_range_low": 36.0, "normal_range_high": 52.0, "unit": "%"},
            {"parameter_name": "Erythrocytes (RBC)", "normal_range_low": 4.2, "normal_range_high": 5.9, "unit": "10^6/uL"},
        ],
    },
    {
        "test_name": "Fasting Blood Glucose",
        "loinc_code": "1558-6",
        "cost": 500.0,
        "description": "Measurement of blood glucose after an overnight fast (minimum 8 hours).",
        "parameters": [
            {"parameter_name": "Fasting Glucose", "normal_range_low": 3.9, "normal_range_high": 5.6, "unit": "mmol/L"},
        ],
    },
    {
        "test_name": "Random Blood Sugar (RBS)",
        "loinc_code": "2345-7",
        "cost": 500.0,
        "description": "Random measurement of plasma glucose level.",
        "parameters": [
            {"parameter_name": "Random Glucose", "normal_range_low": 3.9, "normal_range_high": 7.8, "unit": "mmol/L"},
        ],
    },
    {
        "test_name": "HbA1c (Glycated Hemoglobin)",
        "loinc_code": "4548-4",
        "cost": 2000.0,
        "description": "Average blood glucose levels over the past 2 to 3 months.",
        "parameters": [
            {"parameter_name": "HbA1c Percentage", "normal_range_low": 4.0, "normal_range_high": 5.6, "unit": "%"},
        ],
    },
    {
        "test_name": "Renal Function Test (U&E / Creatinine)",
        "loinc_code": "2160-0",
        "cost": 2500.0,
        "description": "Assessment of kidney function including Serum Creatinine, Urea, and Electrolytes.",
        "parameters": [
            {"parameter_name": "Serum Creatinine", "normal_range_low": 53.0, "normal_range_high": 115.0, "unit": "umol/L"},
            {"parameter_name": "Blood Urea Nitrogen", "normal_range_low": 2.5, "normal_range_high": 7.1, "unit": "mmol/L"},
            {"parameter_name": "Sodium (Na+)", "normal_range_low": 135.0, "normal_range_high": 145.0, "unit": "mmol/L"},
            {"parameter_name": "Potassium (K+)", "normal_range_low": 3.5, "normal_range_high": 5.1, "unit": "mmol/L"},
            {"parameter_name": "Chloride (Cl-)", "normal_range_low": 96.0, "normal_range_high": 106.0, "unit": "mmol/L"},
        ],
    },
    {
        "test_name": "Liver Function Test (LFT)",
        "loinc_code": "1742-6",
        "cost": 2500.0,
        "description": "Evaluates liver enzymes, proteins, and bilirubin levels.",
        "parameters": [
            {"parameter_name": "ALT (SGPT)", "normal_range_low": 7.0, "normal_range_high": 56.0, "unit": "U/L"},
            {"parameter_name": "AST (SGOT)", "normal_range_low": 10.0, "normal_range_high": 40.0, "unit": "U/L"},
            {"parameter_name": "Total Bilirubin", "normal_range_low": 3.4, "normal_range_high": 20.5, "unit": "umol/L"},
            {"parameter_name": "Direct Bilirubin", "normal_range_low": 0.0, "normal_range_high": 5.1, "unit": "umol/L"},
            {"parameter_name": "Alkaline Phosphatase (ALP)", "normal_range_low": 44.0, "normal_range_high": 147.0, "unit": "U/L"},
            {"parameter_name": "Serum Albumin", "normal_range_low": 35.0, "normal_range_high": 50.0, "unit": "g/L"},
        ],
    },
    {
        "test_name": "Lipid Profile",
        "loinc_code": "2093-3",
        "cost": 3000.0,
        "description": "Measures total cholesterol, HDL, LDL, and triglycerides.",
        "parameters": [
            {"parameter_name": "Total Cholesterol", "normal_range_low": 3.0, "normal_range_high": 5.2, "unit": "mmol/L"},
            {"parameter_name": "HDL Cholesterol", "normal_range_low": 1.0, "normal_range_high": 2.1, "unit": "mmol/L"},
            {"parameter_name": "LDL Cholesterol", "normal_range_low": 0.0, "normal_range_high": 3.3, "unit": "mmol/L"},
            {"parameter_name": "Triglycerides", "normal_range_low": 0.4, "normal_range_high": 1.7, "unit": "mmol/L"},
        ],
    },
    {
        "test_name": "Malaria Rapid Diagnostic Test (RDT)",
        "loinc_code": "58941-6",
        "cost": 500.0,
        "description": "Antigen-based rapid test for Plasmodium falciparum and vivax.",
        "parameters": [
            {"parameter_name": "Plasmodium falciparum Ag", "normal_range_low": 0.0, "normal_range_high": 0.0, "unit": "Negative/Positive"},
        ],
    },
    {
        "test_name": "Urinalysis Dipstick & Microscopy",
        "loinc_code": "5804-0",
        "cost": 800.0,
        "description": "Screening test to check urine parameters including pH, protein, glucose, and sediment.",
        "parameters": [
            {"parameter_name": "pH", "normal_range_low": 5.0, "normal_range_high": 8.0, "unit": "pH units"},
            {"parameter_name": "Specific Gravity", "normal_range_low": 1.005, "normal_range_high": 1.030, "unit": "sg"},
            {"parameter_name": "Urine Protein", "normal_range_low": 0.0, "normal_range_high": 0.15, "unit": "g/L"},
            {"parameter_name": "Urine Glucose", "normal_range_low": 0.0, "normal_range_high": 0.8, "unit": "mmol/L"},
            {"parameter_name": "Leukocyte Esterase", "normal_range_low": 0.0, "normal_range_high": 0.0, "unit": "Negative/Positive"},
        ],
    },
    {
        "test_name": "HIV 1/2 Rapid Screening Test",
        "loinc_code": "75622-1",
        "cost": 0.0,
        "description": "Qualitative rapid immunoassay for antibodies to HIV-1 and HIV-2.",
        "parameters": [
            {"parameter_name": "HIV 1/2 Antibody", "normal_range_low": 0.0, "normal_range_high": 0.0, "unit": "Non-Reactive"},
        ],
    },
    {
        "test_name": "HIV-1 Viral Load (Quantification)",
        "loinc_code": "20447-9",
        "cost": 3500.0,
        "description": "Quantitative PCR measurement of HIV-1 RNA copies in plasma.",
        "parameters": [
            {"parameter_name": "HIV-1 RNA Copies", "normal_range_low": 0.0, "normal_range_high": 50.0, "unit": "copies/mL"},
        ],
    },
    {
        "test_name": "CD4+ Absolute T-Cell Count",
        "loinc_code": "24467-3",
        "cost": 2000.0,
        "description": "Flow cytometry determination of absolute CD4 lymphocyte count.",
        "parameters": [
            {"parameter_name": "CD4 Count", "normal_range_low": 500.0, "normal_range_high": 1500.0, "unit": "cells/uL"},
        ],
    },
    {
        "test_name": "TB GeneXpert MTB/RIF Assay",
        "loinc_code": "80367-6",
        "cost": 2500.0,
        "description": "Automated nucleic acid amplification test for Mycobacterium tuberculosis and rifampicin resistance.",
        "parameters": [
            {"parameter_name": "MTB DNA", "normal_range_low": 0.0, "normal_range_high": 0.0, "unit": "Not Detected"},
            {"parameter_name": "Rifampicin Resistance", "normal_range_low": 0.0, "normal_range_high": 0.0, "unit": "Not Detected"},
        ],
    },
    {
        "test_name": "Stool Routine Examination & Parasitology",
        "loinc_code": "10701-1",
        "cost": 800.0,
        "description": "Microscopic and macroscopic examination of stool specimen.",
        "parameters": [
            {"parameter_name": "Ova and Parasites", "normal_range_low": 0.0, "normal_range_high": 0.0, "unit": "Not Seen"},
            {"parameter_name": "Pus Cells (WBC)", "normal_range_low": 0.0, "normal_range_high": 5.0, "unit": "/HPF"},
        ],
    },
    {
        "test_name": "Thyroid Stimulating Hormone (TSH)",
        "loinc_code": "3016-3",
        "cost": 2000.0,
        "description": "Quantitative assay for serum TSH level.",
        "parameters": [
            {"parameter_name": "Serum TSH", "normal_range_low": 0.4, "normal_range_high": 4.2, "unit": "uIU/mL"},
        ],
    },
    {
        "test_name": "C-Reactive Protein (CRP)",
        "loinc_code": "1988-5",
        "cost": 1200.0,
        "description": "Inflammatory marker evaluation.",
        "parameters": [
            {"parameter_name": "CRP Level", "normal_range_low": 0.0, "normal_range_high": 10.0, "unit": "mg/L"},
        ],
    },
    {
        "test_name": "Erythrocyte Sedimentation Rate (ESR)",
        "loinc_code": "4537-7",
        "cost": 700.0,
        "description": "Nonspecific measurement of systemic inflammation.",
        "parameters": [
            {"parameter_name": "ESR Rate", "normal_range_low": 0.0, "normal_range_high": 20.0, "unit": "mm/hr"},
        ],
    },
]


def seed_lab_test_catalog() -> int:
    """Populates standard clinical lab tests and result templates."""
    logger.info("Seeding standard LabTest catalog with LOINC mappings...")
    seeded_count = 0

    try:
        for item in STANDARD_LAB_TESTS:
            existing = LabTest.query.filter(
                (LabTest.test_name == item["test_name"])
                | (LabTest.loinc_code == item["loinc_code"])
            ).first()

            if not existing:
                lab_test = LabTest(
                    test_name=item["test_name"],
                    cost=item["cost"],
                    description=item["description"],
                    loinc_code=item["loinc_code"],
                )
                db.session.add(lab_test)
                db.session.flush()
                seeded_count += 1
            else:
                lab_test = existing
                if not lab_test.loinc_code:
                    lab_test.loinc_code = item["loinc_code"]
                if not lab_test.description:
                    lab_test.description = item["description"]

            # Seed result templates if none exist for this test
            existing_templates = LabResultTemplate.query.filter_by(
                test_id=lab_test.id
            ).all()
            if not existing_templates:
                for param in item.get("parameters", []):
                    template = LabResultTemplate(
                        test_id=lab_test.id,
                        parameter_name=param["parameter_name"],
                        normal_range_low=param["normal_range_low"],
                        normal_range_high=param["normal_range_high"],
                        unit=param["unit"],
                    )
                    db.session.add(template)

        db.session.commit()
        logger.info("Successfully seeded %d lab tests into catalog.", seeded_count)
        return seeded_count

    except Exception as e:  # noqa: BLE001
        db.session.rollback()
        logger.error("Failed to seed lab test catalog: %s", e)
        return 0


if __name__ == "__main__":
    from app import app

    with app.app_context():
        count = seed_lab_test_catalog()
        print(f"✅ Seeding complete: {count} new lab tests created.")
