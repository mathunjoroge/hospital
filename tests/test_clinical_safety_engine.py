"""
tests/test_clinical_safety_engine.py
──────────────────────────────────────
Unit tests for departments/clinical_safety/engine.py.

Prior coverage: 34% (73/111 lines uncovered).
These tests exercise:
  - AlertSeverity / AlertType constants
  - SafetyAlert.to_dict() and overridable logic
  - SafetyCheckResult properties (has_critical, can_proceed, has_alerts)
  - ClinicalSafetyEngine.check_prescription() — no allergies, allergy match
    (critical & mild severity), case-insensitive matching, unrelated drug
  - ClinicalSafetyEngine.log_override() — DB write + return value

Model schema notes (required for correct fixtures):
  - Drug.generic_name  (not 'name' or 'drug_name')
  - Drug.category_id   (FK to DrugCategory, nullable=False)
  - PatientAllergy.patient_id is FK to patients.patient_id (string business key)
  - Patient.patient_id is the string business key (e.g. 'P-001')
"""

import datetime

from departments.clinical_safety.engine import (
    AlertSeverity,
    AlertType,
    ClinicalSafetyEngine,
    SafetyAlert,
    SafetyCheckResult,
)
from departments.clinical_safety.models import SafetyAlertOverride
from extensions import db

# ── Fixtures / factory helpers ────────────────────────────────────────────────

def _make_category(name: str = "Antibiotic"):
    """Create a DrugCategory row and return it flushed."""
    from departments.models.pharmacy import DrugCategory
    cat = DrugCategory(name=name)
    db.session.add(cat)
    db.session.flush()
    return cat


def _make_drug(generic_name: str, category_id: int):
    """Create a Drug row with all required columns."""
    from departments.models.pharmacy import Drug
    drug = Drug(
        generic_name=generic_name,
        brand_name=f"{generic_name} Brand",
        category_id=category_id,
        dosage_form="tablet",
        strength="500mg",
        buying_price=5.0,
        selling_price=10.0,
        quantity_in_stock=100,
    )
    db.session.add(drug)
    db.session.flush()
    return drug


def _make_patient(pid: str, name: str):
    """Create a Patient row with all required columns."""
    from departments.models.records import Patient
    p = Patient(
        patient_id=pid,
        name=name,
        place_of_residence="Nairobi",
        sex="Male",
        date_of_birth=datetime.date(1990, 1, 1),
        marital_status="Single",
        next_of_kin="Kin Name",
        relationship_with_next_of_kin="Parent",
        next_of_kin_contact="0700000000",
        national_id=pid.replace("-", ""),
        contact="0700000001",
    )
    db.session.add(p)
    db.session.flush()
    return p


def _make_allergy(patient_pid: str, allergen: str, severity: str = "SEVERE"):
    """
    Create a PatientAllergy row.
    patient_pid must be the string business key (patients.patient_id),
    not the integer pk — that is the FK the table uses.
    """
    from departments.models.records import PatientAllergy
    a = PatientAllergy(
        patient_id=patient_pid,   # FK to patients.patient_id (string)
        allergen=allergen,
        category="DRUG",
        severity=severity,
    )
    db.session.add(a)
    db.session.flush()
    return a


# ── AlertSeverity / AlertType constants ──────────────────────────────────────

class TestAlertConstants:
    def test_severity_values(self):
        assert AlertSeverity.CRITICAL == "CRITICAL"
        assert AlertSeverity.HIGH == "HIGH"
        assert AlertSeverity.MODERATE == "MODERATE"
        assert AlertSeverity.LOW == "LOW"

    def test_alert_type_values(self):
        assert AlertType.ALLERGY == "ALLERGY"
        assert AlertType.DRUG_INTERACTION == "DRUG_INTERACTION"
        assert AlertType.DOSE_LIMIT == "DOSE_LIMIT"
        assert AlertType.CONTRAINDICATION == "CONTRAINDICATION"
        assert AlertType.DUPLICATE_THERAPY == "DUPLICATE_THERAPY"


# ── SafetyAlert ───────────────────────────────────────────────────────────────

class TestSafetyAlert:
    def test_critical_is_not_overridable(self):
        alert = SafetyAlert(
            alert_type=AlertType.ALLERGY,
            severity=AlertSeverity.CRITICAL,
            message="Test critical",
            patient_id=1,
        )
        assert alert.overridable is False

    def test_high_is_overridable(self):
        alert = SafetyAlert(
            alert_type=AlertType.DRUG_INTERACTION,
            severity=AlertSeverity.HIGH,
            message="Test high",
            patient_id=1,
        )
        assert alert.overridable is True

    def test_moderate_is_overridable(self):
        alert = SafetyAlert(
            alert_type=AlertType.DUPLICATE_THERAPY,
            severity=AlertSeverity.MODERATE,
            message="Test moderate",
            patient_id=1,
        )
        assert alert.overridable is True

    def test_to_dict_contains_all_keys(self):
        alert = SafetyAlert(
            alert_type=AlertType.ALLERGY,
            severity=AlertSeverity.CRITICAL,
            message="Allergy to penicillin",
            patient_id=42,
            drug_id=7,
            allergen="penicillin",
            recommendation="Use macrolide instead",
        )
        d = alert.to_dict()
        assert d["alert_type"] == AlertType.ALLERGY
        assert d["severity"] == AlertSeverity.CRITICAL
        assert d["message"] == "Allergy to penicillin"
        assert d["patient_id"] == 42
        assert d["drug_id"] == 7
        assert d["allergen"] == "penicillin"
        assert d["recommendation"] == "Use macrolide instead"
        assert d["overridable"] is False

    def test_to_dict_optional_fields_none(self):
        alert = SafetyAlert(
            alert_type=AlertType.DOSE_LIMIT,
            severity=AlertSeverity.LOW,
            message="Low dose note",
            patient_id=1,
        )
        d = alert.to_dict()
        assert d["drug_id"] is None
        assert d["allergen"] is None
        assert d["recommendation"] is None


# ── SafetyCheckResult ─────────────────────────────────────────────────────────

class TestSafetyCheckResult:
    def test_empty_result_can_proceed(self):
        result = SafetyCheckResult()
        assert result.has_alerts is False
        assert result.has_critical is False
        assert result.can_proceed is True
        assert result.alerts == []

    def test_add_moderate_alert_does_not_block(self):
        result = SafetyCheckResult()
        result.add_alert(SafetyAlert(AlertType.DUPLICATE_THERAPY, AlertSeverity.MODERATE, "dup", 1))
        assert result.has_alerts is True
        assert result.has_critical is False
        assert result.can_proceed is True

    def test_add_critical_alert_blocks_proceed(self):
        result = SafetyCheckResult()
        result.add_alert(SafetyAlert(AlertType.ALLERGY, AlertSeverity.CRITICAL, "allergy", 1))
        assert result.has_critical is True
        assert result.can_proceed is False

    def test_to_dict_structure(self):
        result = SafetyCheckResult()
        result.add_alert(SafetyAlert(AlertType.ALLERGY, AlertSeverity.HIGH, "msg", 1))
        d = result.to_dict()
        assert d["has_alerts"] is True
        assert d["has_critical"] is False
        assert d["can_proceed"] is True
        assert d["total_alerts"] == 1
        assert len(d["alerts"]) == 1
        assert "checked_at" in d

    def test_mixed_severities_critical_wins(self):
        result = SafetyCheckResult()
        result.add_alert(SafetyAlert(AlertType.DUPLICATE_THERAPY, AlertSeverity.MODERATE, "dup", 1))
        result.add_alert(SafetyAlert(AlertType.ALLERGY, AlertSeverity.CRITICAL, "allergy", 1))
        assert result.has_critical is True
        assert result.can_proceed is False
        assert result.to_dict()["total_alerts"] == 2


# ── ClinicalSafetyEngine.check_prescription() ────────────────────────────────

class TestClinicalSafetyEngineCheckPrescription:
    def test_no_drugs_returns_empty(self, app):
        """Empty drug list → no alerts, no DB queries that could fail."""
        with app.app_context():
            db.create_all()
            result = ClinicalSafetyEngine().check_prescription(patient_id=999, drug_ids=[])
            assert result.has_alerts is False

    def test_no_allergies_no_alert(self, app):
        """Patient with no allergy records — allergy check returns nothing."""
        with app.app_context():
            db.create_all()
            cat = _make_category("Antibiotic")
            drug = _make_drug("Amoxicillin", cat.id)
            db.session.commit()

            result = ClinicalSafetyEngine().check_prescription(
                patient_id=9001, drug_ids=[drug.id]
            )
            allergy_alerts = [a for a in result.alerts if a.alert_type == AlertType.ALLERGY]
            assert allergy_alerts == []

    def test_allergy_match_severe_is_critical(self, app):
        """SEVERE allergy to the prescribed drug → CRITICAL alert, blocks prescription."""
        with app.app_context():
            db.create_all()
            cat = _make_category("Antibiotic-2")
            drug = _make_drug("Penicillin", cat.id)
            patient = _make_patient("P-CSE-001", "Allergy Patient")
            # PatientAllergy.patient_id FK is the string business key
            _make_allergy(patient.patient_id, "Penicillin", severity="SEVERE")
            db.session.commit()

            # Engine's check_prescription takes the integer pk as patient_id
            result = ClinicalSafetyEngine().check_prescription(
                patient_id=patient.id, drug_ids=[drug.id]
            )

            assert result.has_alerts is True
            assert result.has_critical is True
            assert result.can_proceed is False
            allergy_alerts = [a for a in result.alerts if a.alert_type == AlertType.ALLERGY]
            assert len(allergy_alerts) == 1
            assert "Penicillin" in allergy_alerts[0].message
            assert allergy_alerts[0].overridable is False

    def test_allergy_match_mild_is_moderate(self, app):
        """MILD allergy → MODERATE alert (overridable, prescription can still proceed)."""
        with app.app_context():
            db.create_all()
            cat = _make_category("NSAID")
            drug = _make_drug("Ibuprofen", cat.id)
            patient = _make_patient("P-CSE-002", "Mild Allergy Patient")
            _make_allergy(patient.patient_id, "Ibuprofen", severity="mild")
            db.session.commit()

            result = ClinicalSafetyEngine().check_prescription(
                patient_id=patient.id, drug_ids=[drug.id]
            )

            assert result.has_alerts is True
            assert result.has_critical is False
            assert result.can_proceed is True
            allergy_alerts = [a for a in result.alerts if a.alert_type == AlertType.ALLERGY]
            assert allergy_alerts[0].severity == AlertSeverity.MODERATE
            assert allergy_alerts[0].overridable is True

    def test_allergy_match_low_is_moderate(self, app):
        """LOW severity also maps to MODERATE (same code path as mild)."""
        with app.app_context():
            db.create_all()
            cat = _make_category("Analgesic")
            drug = _make_drug("Codeine", cat.id)
            patient = _make_patient("P-CSE-003", "Low Allergy Patient")
            _make_allergy(patient.patient_id, "Codeine", severity="low")
            db.session.commit()

            result = ClinicalSafetyEngine().check_prescription(
                patient_id=patient.id, drug_ids=[drug.id]
            )
            allergy_alerts = [a for a in result.alerts if a.alert_type == AlertType.ALLERGY]
            assert allergy_alerts[0].severity == AlertSeverity.MODERATE

    def test_allergy_case_insensitive_match(self, app):
        """Allergen matching must be case-insensitive (WARFARIN vs warfarin)."""
        with app.app_context():
            db.create_all()
            cat = _make_category("Anticoagulant")
            drug = _make_drug("WARFARIN", cat.id)
            patient = _make_patient("P-CSE-004", "Case Test Patient")
            _make_allergy(patient.patient_id, "warfarin", severity="SEVERE")
            db.session.commit()

            result = ClinicalSafetyEngine().check_prescription(
                patient_id=patient.id, drug_ids=[drug.id]
            )
            assert result.has_critical is True

    def test_unrelated_drug_no_allergy_alert(self, app):
        """Drug patient is NOT allergic to → zero allergy alerts."""
        with app.app_context():
            db.create_all()
            cat = _make_category("Antipyretic")
            safe_drug = _make_drug("Paracetamol", cat.id)
            patient = _make_patient("P-CSE-005", "Safe Patient")
            _make_allergy(patient.patient_id, "Penicillin", severity="SEVERE")
            db.session.commit()

            result = ClinicalSafetyEngine().check_prescription(
                patient_id=patient.id, drug_ids=[safe_drug.id]
            )
            allergy_alerts = [a for a in result.alerts if a.alert_type == AlertType.ALLERGY]
            assert allergy_alerts == []

    def test_multiple_drugs_only_allergenic_triggers_alert(self, app):
        """When prescribing two drugs, only the one matching an allergy triggers an alert."""
        with app.app_context():
            db.create_all()
            cat = _make_category("Mixed")
            safe_drug = _make_drug("Metformin", cat.id)
            allergenic_drug = _make_drug("Sulfonamide", cat.id)
            patient = _make_patient("P-CSE-006", "Multi Drug Patient")
            _make_allergy(patient.patient_id, "Sulfonamide", severity="SEVERE")
            db.session.commit()

            result = ClinicalSafetyEngine().check_prescription(
                patient_id=patient.id, drug_ids=[safe_drug.id, allergenic_drug.id]
            )
            allergy_alerts = [a for a in result.alerts if a.alert_type == AlertType.ALLERGY]
            assert len(allergy_alerts) == 1
            assert allergy_alerts[0].allergen.lower() == "sulfonamide"


# ── ClinicalSafetyEngine.log_override() ──────────────────────────────────────

class TestLogOverride:
    def test_log_override_creates_db_record(self, app):
        """log_override() persists a SafetyAlertOverride and returns it."""
        with app.app_context():
            db.create_all()
            override = ClinicalSafetyEngine().log_override(
                patient_id=77,
                clinician_id=5,
                alert_type=AlertType.ALLERGY,
                alert_message="Allergy to Penicillin overridden",
                justification="No alternative available in current formulary",
            )
            assert override.patient_id == 77
            assert override.clinician_id == 5
            assert override.alert_type == AlertType.ALLERGY
            assert "Penicillin" in override.alert_message
            assert "formulary" in override.justification

            from_db = db.session.get(SafetyAlertOverride, override.id)
            assert from_db is not None
            assert from_db.clinician_id == 5

    def test_log_override_drug_interaction(self, app):
        """Override is recorded for DRUG_INTERACTION alert type too."""
        with app.app_context():
            db.create_all()
            override = ClinicalSafetyEngine().log_override(
                patient_id=10,
                clinician_id=2,
                alert_type=AlertType.DRUG_INTERACTION,
                alert_message="Warfarin + Aspirin interaction",
                justification="Benefits outweigh risks — cardiology consult obtained",
            )
            assert override.alert_type == AlertType.DRUG_INTERACTION
            assert override.overridden_at is not None

    def test_log_override_multiple_patients_isolated(self, app):
        """Overrides for different patients are stored independently."""
        with app.app_context():
            db.create_all()
            engine = ClinicalSafetyEngine()
            ov1 = engine.log_override(1, 10, AlertType.ALLERGY, "msg1", "reason1")
            ov2 = engine.log_override(2, 11, AlertType.DOSE_LIMIT, "msg2", "reason2")
            assert ov1.id != ov2.id
            assert ov1.patient_id == 1
            assert ov2.patient_id == 2
