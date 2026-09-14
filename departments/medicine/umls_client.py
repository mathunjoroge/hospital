"""
departments/medicine/umls_client.py
────────────────────────────────────
UMLS (National Library of Medicine UTS) REST API Client.

Interfaces with UMLS UTS REST API (https://uts-ws.nlm.nih.gov/rest) to search:
  - SNOMED CT (sABs=SNOMEDCT_US)
  - LOINC (sABs=LNC)

Provides live typeahead search fallback as well as curated seed datasets
for bulk population of local PostgreSQL terminology tables.
"""

import logging
import os

import requests
from flask import current_app

logger = logging.getLogger(__name__)

_UMLS_SEARCH_URL = "https://uts-ws.nlm.nih.gov/rest/search/current"
_UMLS_CUI_ATOMS_URL = "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}/atoms"


def get_umls_api_key() -> str:
    """Retrieve UMLS API Key from Flask app config or environment variables."""
    try:
        key = current_app.config.get("UMLS_API_KEY")
        if key:
            return key
    except RuntimeError:
        pass
    return os.getenv("UMLS_API_KEY", "")


def search_umls(query: str, sab: str = "SNOMEDCT_US", max_results: int = 20) -> list[dict]:
    """
    Search UMLS REST API for concepts matching query in specified source vocabulary (sab).

    :param query: Free-text search string (e.g. "fever", "glucose")
    :param sab: Target vocabulary code ('SNOMEDCT_US' or 'LNC')
    :param max_results: Maximum results to return
    :return: List of dicts with keys 'code' and 'description'
    """
    api_key = get_umls_api_key()
    if not api_key:
        logger.warning("UMLS_API_KEY not configured. Live search unavailable.")
        return []

    q = (query or "").strip()
    if not q:
        return []

    params = {
        "string": q,
        "sABs": sab,
        "language": "ENG",
        "pageSize": max_results,
        "apiKey": api_key,
    }

    try:
        response = requests.get(_UMLS_SEARCH_URL, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        raw_results = data.get("result", {}).get("results", [])

        results = []
        seen_codes = set()

        for item in raw_results:
            name = item.get("name", "").strip()
            cui = item.get("ui", "").strip()
            if not name or not cui:
                continue

            # Attempt to extract precise concept code via CUI atom lookup if CUI provided
            code = _extract_code_for_cui(cui, sab, api_key) or cui

            if code not in seen_codes:
                seen_codes.add(code)
                results.append({
                    "code": code,
                    "description": name,
                    "cui": cui,
                })
                if len(results) >= max_results:
                    break

        return results

    except Exception as e:  # noqa: BLE001
        logger.warning("UMLS API search error (%s, query='%s'): %s", sab, q, e)
        return []


def _extract_code_for_cui(cui: str, sab: str, api_key: str) -> str | None:
    """Helper to resolve exact source code (SNOMED ID / LOINC Num) from UMLS CUI atoms."""
    if not cui.startswith("C"):
        return cui

    try:
        url = _UMLS_CUI_ATOMS_URL.format(cui=cui)
        params = {"sABs": sab, "language": "ENG", "apiKey": api_key}
        res = requests.get(url, params=params, timeout=5)
        if res.status_code == 200:
            atoms = res.json().get("result", [])
            for a in atoms:
                if a.get("rootSource") == sab:
                    code_val = a.get("code", "")
                    if "/" in code_val:
                        return code_val.split("/")[-1]
                    if code_val:
                        return code_val
    except Exception:  # noqa: BLE001
        pass
    return None


def search_snomed_live(query: str, max_results: int = 20) -> list[dict]:
    """Search SNOMED CT concepts via UMLS API."""
    return search_umls(query, sab="SNOMEDCT_US", max_results=max_results)


def search_loinc_live(query: str, max_results: int = 20) -> list[dict]:
    """Search LOINC concepts via UMLS API."""
    return search_umls(query, sab="LNC", max_results=max_results)


def get_core_snomed_seed_dataset() -> list[dict]:
    """
    Curated core dataset of essential SNOMED CT clinical findings & procedures.
    Used to seed the local database for offline fast searching.
    """
    return [
        {"code": "38341003", "description": "Hypertensive disorder"},
        {"code": "44054006", "description": "Type 2 diabetes mellitus"},
        {"code": "424754009", "description": "Fever"},
        {"code": "22298006", "description": "Myocardial infarction"},
        {"code": "195967001", "description": "Asthma"},
        {"code": "233604007", "description": "Pneumonia"},
        {"code": "61462000", "description": "Malaria"},
        {"code": "85189001", "description": "Acute appendicitis"},
        {"code": "86406008", "description": "Human immunodeficiency virus infection"},
        {"code": "56717001", "description": "Tuberculosis"},
        {"code": "91302008", "description": "Sepsis"},
        {"code": "271737000", "description": "Anemia"},
        {"code": "68566005", "description": "Urinary tract infection"},
        {"code": "25374005", "description": "Gastroenteritis"},
        {"code": "25064002", "description": "Headache"},
        {"code": "29857009", "description": "Chest pain"},
        {"code": "49727002", "description": "Cough"},
        {"code": "267036007", "description": "Dyspnea"},
        {"code": "11466000", "description": "Cesarean section"},
        {"code": "199994002", "description": "Normal spontaneous vaginal delivery"},
        {"code": "71620000", "description": "Fracture of femur"},
        {"code": "404684003", "description": "Clinical finding"},
        {"code": "71388002", "description": "Procedure"},
        {"code": "128053003", "description": "Deep vein thrombosis"},
        {"code": "230690007", "description": "Stroke"},
        {"code": "42343007", "description": "Congestive heart failure"},
        {"code": "13645005", "description": "Chronic obstructive lung disease"},
        {"code": "73211009", "description": "Diabetes mellitus"},
        {"code": "46635009", "description": "Diabetes mellitus type 1"},
        {"code": "90708001", "description": "Kidney disease"},
        {"code": "709044004", "description": "Chronic kidney disease"},
        {"code": "235856003", "description": "Liver disease"},
        {"code": "371087003", "description": "Gastritis"},
        {"code": "13213009", "description": "Peptic ulcer disease"},
        {"code": "58848006", "description": "Pancreatitis"},
        {"code": "84757009", "description": "Epilepsy"},
        {"code": "37796009", "description": "Migraine"},
        {"code": "363346000", "description": "Malignant neoplastic disease"},
        {"code": "254837009", "description": "Breast carcinoma"},
        {"code": "363406005", "description": "Colon carcinoma"},
        {"code": "254637007", "description": "Non-small cell lung cancer"},
        {"code": "93870000", "description": "Prostate carcinoma"},
        {"code": "363354003", "description": "Cervical carcinoma"},
        {"code": "109989006", "description": "Multiple myeloma"},
        {"code": "91857003", "description": "Acute myeloid leukemia"},
        {"code": "128613002", "description": "End stage renal disease"},
        {"code": "14669001", "description": "Acute kidney injury"},
        {"code": "302215000", "description": "Hemodialysis session"},
        {"code": "238324003", "description": "Peritoneal dialysis procedure"},
        {"code": "52613005", "description": "Pre-eclampsia"},
        {"code": "23045005", "description": "Eclampsia"},
        {"code": "47346000", "description": "Postpartum hemorrhage"},
        {"code": "48194001", "description": "Neonatal jaundice"},
        {"code": "276536005", "description": "Neonatal respiratory distress syndrome"},
        {"code": "18099001", "description": "Cholecystectomy"},
        {"code": "80146002", "description": "Appendectomy"},
        {"code": "81723002", "description": "Coronary artery bypass graft"},
        {"code": "52734007", "description": "Total hip replacement"},
        {"code": "410620009", "description": "Rotavirus gastroenteritis"},
        {"code": "186431008", "description": "Measles"},
        {"code": "14189004", "description": "Pertussis"},
        {"code": "397428000", "description": "Severe acute respiratory syndrome"},
        {"code": "840539006", "description": "COVID-19"},
    ]


def get_core_loinc_seed_dataset() -> list[dict]:
    """
    Curated core dataset of essential LOINC lab observations and vitals.
    Used to seed the local database for offline fast searching.
    """
    return [
        {"code": "8302-2", "description": "Body height"},
        {"code": "29463-7", "description": "Body weight"},
        {"code": "8310-5", "description": "Body temperature"},
        {"code": "8867-4", "description": "Heart rate"},
        {"code": "8480-6", "description": "Systolic blood pressure"},
        {"code": "8462-4", "description": "Diastolic blood pressure"},
        {"code": "2708-6", "description": "Oxygen saturation in Arterial blood by Pulse oximetry"},
        {"code": "9279-1", "description": "Respiratory rate"},
        {"code": "718-7", "description": "Hemoglobin [Mass/volume] in Blood"},
        {"code": "4544-3", "description": "Hematocrit [Volume Fraction] in Blood"},
        {"code": "6690-2", "description": "Leukocytes [#/volume] in Blood by Automated count"},
        {"code": "777-3", "description": "Platelets [#/volume] in Blood by Automated count"},
        {"code": "1558-6", "description": "Fasting glucose [Mass/volume] in Serum or Plasma"},
        {"code": "2345-7", "description": "Glucose [Mass/volume] in Serum or Plasma"},
        {"code": "4548-4", "description": "Hemoglobin A1c/Hemoglobin.total in Blood"},
        {"code": "2160-0", "description": "Creatinine [Mass/volume] in Serum or Plasma"},
        {"code": "3094-0", "description": "Urea nitrogen [Mass/volume] in Serum or Plasma"},
        {"code": "1742-6", "description": "Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma"},
        {"code": "1920-8", "description": "Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma"},
        {"code": "1975-2", "description": "Bilirubin.total [Mass/volume] in Serum or Plasma"},
        {"code": "2093-3", "description": "Cholesterol [Mass/volume] in Serum or Plasma"},
        {"code": "2571-8", "description": "Triglyceride [Mass/volume] in Serum or Plasma"},
        {"code": "2085-9", "description": "Cholesterol in HDL [Mass/volume] in Serum or Plasma"},
        {"code": "13457-7", "description": "Cholesterol in LDL [Mass/volume] in Serum or Plasma"},
        {"code": "2951-2", "description": "Sodium [Moles/volume] in Serum or Plasma"},
        {"code": "2823-3", "description": "Potassium [Moles/volume] in Serum or Plasma"},
        {"code": "2075-0", "description": "Chloride [Moles/volume] in Serum or Plasma"},
        {"code": "1963-8", "description": "Bicarbonate [Moles/volume] in Serum or Plasma"},
        {"code": "17861-6", "description": "Calcium [Mass/volume] in Serum or Plasma"},
        {"code": "2777-1", "description": "Phosphate [Mass/volume] in Serum or Plasma"},
        {"code": "2601-3", "description": "Magnesium [Mass/volume] in Serum or Plasma"},
        {"code": "1988-5", "description": "C reactive protein [Mass/volume] in Serum or Plasma"},
        {"code": "6598-7", "description": "Troponin T.cardiac [Mass/volume] in Serum or Plasma"},
        {"code": "34571-0", "description": "Plasmodium sp identification in Blood by Light microscopy"},
        {"code": "43012-4", "description": "HIV 1+2 Ab screening rapid test in Blood"},
        {"code": "24357-6", "description": "Urinalysis automated dipstick panel - Urine"},
        {"code": "24467-3", "description": "CD4 cells [#/volume] in Blood"},
        {"code": "25835-0", "description": "HIV 1 RNA [#/volume] (viral load) in Plasma by NAA with probe"},
        {"code": "600-7", "description": "Bacteria identified in Blood by Culture"},
        {"code": "88206-8", "description": "Mycobacterium tuberculosis DNA [Presence] in Sputum by NAA with probe"},
        {"code": "14804-9", "description": "Lactate [Moles/volume] in Blood"},
        {"code": "33914-3", "description": "Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum or Plasma (CKD-EPI)"},
        {"code": "20505-4", "description": "Bilirubin.direct [Mass/volume] in Serum or Plasma"},
        {"code": "1751-7", "description": "Albumin [Mass/volume] in Serum or Plasma"},
        {"code": "2885-2", "description": "Protein [Mass/volume] in Serum or Plasma"},
        {"code": "2888-6", "description": "Protein [Mass/volume] in Urine"},
        {"code": "14933-6", "description": "Prothrombin time (PT)"},
        {"code": "6301-6", "description": "INR in Blood by Coagulation assay"},
        {"code": "3173-2", "description": "aPTT in Blood by Coagulation assay"},
        {"code": "48065-7", "description": "Fibrin D-dimer FEU [Mass/volume] in Platelet poor plasma"},
    ]
