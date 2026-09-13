"""
departments/medicine/terminology_server.py
──────────────────────────────────────────
HL7 FHIR R4 Terminology Engine & Clinical Search Server.

Provides:
  - FHIR R4 $lookup operation (CodeSystem/$lookup)
  - FHIR R4 $validate-code operation (CodeSystem/$validate-code)
  - High-speed indexed terminology autocomplete search across ICD-10, SNOMED CT, and LOINC.
"""

import logging
from typing import Any

from departments.models.terminology import ICD10Code, LoincCode, SnomedCode
from extensions import db

logger = logging.getLogger(__name__)

# Canonical CodeSystem URIs per FHIR R4 specification
SYSTEM_URIS = {
    "icd10": "http://hl7.org/fhir/sid/icd-10",
    "snomed": "http://snomed.info/sct",
    "loinc": "http://loinc.org",
}

# System aliases for flexible matching
SYSTEM_ALIASES = {
    "http://hl7.org/fhir/sid/icd-10": "icd10",
    "icd-10": "icd10",
    "icd10": "icd10",
    "icd": "icd10",
    "http://snomed.info/sct": "snomed",
    "snomed": "snomed",
    "snomed-ct": "snomed",
    "snomedct": "snomed",
    "http://loinc.org": "loinc",
    "loinc": "loinc",
}

# Common fallback dictionary for development/testing when DB is unpopulated
FALLBACK_CODES: dict[str, dict[str, dict[str, Any]]] = {
    "icd10": {
        "J00": {"description": "Acute nasopharyngitis [common cold]", "chapter": "Respiratory"},
        "I10": {"description": "Essential (primary) hypertension", "chapter": "Cardiovascular"},
        "R50.9": {"description": "Fever, unspecified", "chapter": "General"},
        "J06.9": {"description": "Acute upper respiratory infection, unspecified", "chapter": "Respiratory"},
        "E11.9": {"description": "Type 2 diabetes mellitus without complications", "chapter": "Endocrine"},
    },
    "snomed": {
        "404684003": {"description": "Clinical finding"},
        "22298006": {"description": "Myocardial infarction"},
        "38341003": {"description": "Hypertensive disorder"},
        "73211009": {"description": "Diabetes mellitus"},
        "195662009": {"description": "Acute viral pharyngitis"},
    },
    "loinc": {
        "8302-2": {"description": "Body height"},
        "8867-4": {"description": "Heart rate"},
        "8480-6": {"description": "Systolic blood pressure"},
        "8462-4": {"description": "Diastolic blood pressure"},
        "8310-5": {"description": "Body temperature"},
        "59408-5": {"description": "Oxygen saturation in Arterial blood by Pulse oximetry"},
    },
}


class FHIRTerminologyServer:
    """
    HL7 FHIR R4 Terminology Engine providing $lookup, $validate-code, and
    unified multi-dictionary autocomplete search.
    """

    @classmethod
    def normalize_system(cls, system: str | None) -> str:
        """Normalize system string/URI to standard key ('icd10', 'snomed', 'loinc')."""
        if not system:
            return "icd10"
        key = system.strip().lower()
        return SYSTEM_ALIASES.get(key, "icd10")

    @classmethod
    def get_canonical_uri(cls, system_key: str) -> str:
        """Return canonical FHIR URI for system key."""
        return SYSTEM_URIS.get(system_key, "http://hl7.org/fhir/sid/icd-10")

    @classmethod
    def lookup_code(cls, system: str, code: str) -> dict[str, Any]:
        """
        Execute FHIR R4 $lookup operation.

        Returns FHIR R4 Parameters resource or error dict.
        """
        sys_key = cls.normalize_system(system)
        canonical_uri = cls.get_canonical_uri(sys_key)
        code_clean = (code or "").strip()

        found_desc: str | None = None
        chapter: str | None = None

        if sys_key == "icd10":
            row = ICD10Code.query.filter_by(code=code_clean).first()
            if row:
                found_desc = row.description
                chapter = row.chapter
        elif sys_key == "snomed":
            row = SnomedCode.query.filter_by(code=code_clean).first()
            if row:
                found_desc = row.description
        elif sys_key == "loinc":
            row = LoincCode.query.filter_by(code=code_clean).first()
            if row:
                found_desc = row.description

        # Check fallback dictionary if not found in DB
        if not found_desc and code_clean in FALLBACK_CODES.get(sys_key, {}):
            fb = FALLBACK_CODES[sys_key][code_clean]
            found_desc = fb["description"]
            chapter = fb.get("chapter")

        if not found_desc:
            return {
                "resourceType": "Parameters",
                "error": True,
                "status": 404,
                "parameter": [
                    {"name": "result", "valueBoolean": False},
                    {"name": "message", "valueString": f"Code '{code_clean}' not found in system '{canonical_uri}'"},
                ],
            }

        name_display = {"icd10": "ICD-10", "snomed": "SNOMED CT", "loinc": "LOINC"}.get(sys_key, "Terminology")
        parameters = [
            {"name": "name", "valueString": name_display},
            {"name": "system", "valueUri": canonical_uri},
            {"name": "code", "valueCode": code_clean},
            {"name": "display", "valueString": found_desc},
            {"name": "abstract", "valueBoolean": False},
        ]
        if chapter:
            parameters.append({"name": "property", "valueString": f"chapter: {chapter}"})

        return {
            "resourceType": "Parameters",
            "parameter": parameters,
        }

    @classmethod
    def validate_code(
        cls, system: str, code: str, display: str | None = None
    ) -> dict[str, Any]:
        """
        Execute FHIR R4 $validate-code operation.

        Validates if code exists and optionally if display text matches.
        """
        sys_key = cls.normalize_system(system)
        canonical_uri = cls.get_canonical_uri(sys_key)
        code_clean = (code or "").strip()

        lookup_res = cls.lookup_code(system, code_clean)
        if lookup_res.get("error"):
            return {
                "resourceType": "Parameters",
                "parameter": [
                    {"name": "result", "valueBoolean": False},
                    {
                        "name": "message",
                        "valueString": f"Code '{code_clean}' is NOT valid in system '{canonical_uri}'",
                    },
                ],
            }

        # Extract official display
        official_display = ""
        for p in lookup_res.get("parameter", []):
            if p.get("name") == "display":
                official_display = p.get("valueString", "")

        is_valid = True
        msg = f"Code '{code_clean}' is valid in system '{canonical_uri}'"

        if display:
            disp_clean = display.strip().lower()
            if disp_clean not in official_display.lower():
                is_valid = False
                msg = (
                    f"Code '{code_clean}' exists, but display '{display}' does not match "
                    f"official display '{official_display}'"
                )

        return {
            "resourceType": "Parameters",
            "parameter": [
                {"name": "result", "valueBoolean": is_valid},
                {"name": "message", "valueString": msg},
                {"name": "display", "valueString": official_display},
            ],
        }

    @classmethod
    def search_terms(
        cls, query: str | None, system: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """
        Autocomplete multi-dictionary lookup across ICD-10, SNOMED, and LOINC.
        """
        q = (query or "").strip().lower()
        sys_key = cls.normalize_system(system) if system and system.lower() != "all" else "all"

        results: list[dict[str, Any]] = []

        def _search_icd10():
            nonlocal results
            try:
                if not q:
                    rows = ICD10Code.query.limit(limit).all()
                else:
                    rows = (
                        ICD10Code.query.filter(
                            db.or_(
                                ICD10Code.code.ilike(f"%{q}%"),
                                ICD10Code.description.ilike(f"%{q}%"),
                            )
                        )
                        .limit(limit)
                        .all()
                    )
                for r in rows:
                    results.append({
                        "code": r.code,
                        "description": r.description,
                        "system": SYSTEM_URIS["icd10"],
                        "system_name": "ICD-10",
                        "category": r.chapter or "General",
                    })
            except Exception as e:  # noqa: BLE001
                logger.warning("ICD-10 DB search error: %s", e)

        def _search_snomed():
            nonlocal results
            try:
                if not q:
                    rows = SnomedCode.query.limit(limit).all()
                else:
                    rows = (
                        SnomedCode.query.filter(
                            db.or_(
                                SnomedCode.code.ilike(f"%{q}%"),
                                SnomedCode.description.ilike(f"%{q}%"),
                            )
                        )
                        .limit(limit)
                        .all()
                    )
                for r in rows:
                    results.append({
                        "code": r.code,
                        "description": r.description,
                        "system": SYSTEM_URIS["snomed"],
                        "system_name": "SNOMED CT",
                        "category": "Clinical Finding",
                    })
            except Exception as e:  # noqa: BLE001
                logger.warning("SNOMED DB search error: %s", e)

        def _search_loinc():
            nonlocal results
            try:
                if not q:
                    rows = LoincCode.query.limit(limit).all()
                else:
                    rows = (
                        LoincCode.query.filter(
                            db.or_(
                                LoincCode.code.ilike(f"%{q}%"),
                                LoincCode.description.ilike(f"%{q}%"),
                            )
                        )
                        .limit(limit)
                        .all()
                    )
                for r in rows:
                    results.append({
                        "code": r.code,
                        "description": r.description,
                        "system": SYSTEM_URIS["loinc"],
                        "system_name": "LOINC",
                        "category": "Laboratory/Observation",
                    })
            except Exception as e:  # noqa: BLE001
                logger.warning("LOINC DB search error: %s", e)

        if sys_key == "icd10":
            _search_icd10()
        elif sys_key == "snomed":
            _search_snomed()
        elif sys_key == "loinc":
            _search_loinc()
        else:
            _search_icd10()
            _search_snomed()
            _search_loinc()

        # If DB search returned no results, fallback to common dictionary
        if not results:
            systems_to_check = [sys_key] if sys_key != "all" else ["icd10", "snomed", "loinc"]
            for skey in systems_to_check:
                fb_dict = FALLBACK_CODES.get(skey, {})
                for code_str, info in fb_dict.items():
                    desc = info["description"]
                    cat = info.get("chapter", "General")
                    if not q or q in code_str.lower() or q in desc.lower():
                        results.append({
                            "code": code_str,
                            "description": desc,
                            "system": SYSTEM_URIS[skey],
                            "system_name": {"icd10": "ICD-10", "snomed": "SNOMED CT", "loinc": "LOINC"}[skey],
                            "category": cat,
                        })

        return results[:limit]
