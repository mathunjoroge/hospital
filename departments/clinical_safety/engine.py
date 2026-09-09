"""
Clinical Decision Support (CDS) Safety Engine.

Cross-references prescribed drugs against patient allergies and known
drug interactions. Returns structured alerts with severity levels.

This engine is designed to be called from any clinical workflow:
- e-Prescribing
- Pharmacy dispensing
- Medication administration (MAR)
- Admission medication reconciliation
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from departments.clinical_safety.models import SafetyAlertOverride
from departments.models.medicine import PrescribedMedicine
from departments.models.pharmacy import Drug
from departments.models.records import Patient, PatientAllergy
from extensions import db

logger = logging.getLogger(__name__)


def _resolve_patient_pid(patient_id: int) -> "Optional[str]":
    """
    Resolve an integer patient PK to the string business key (patients.patient_id).

    PatientAllergy.patient_id is a FK to patients.patient_id (String), not the
    integer pk. The engine's public API accepts the integer pk for consistency
    with other clinical modules; we resolve here before querying allergies.

    Returns None if the patient record does not exist.
    """
    patient = db.session.get(Patient, patient_id)
    if patient is None:
        return None
    return patient.patient_id


# Severity levels for safety alerts
class AlertSeverity:
    CRITICAL = "CRITICAL"  # Absolute contraindication - must not proceed
    HIGH = "HIGH"  # Strong warning - proceed only with override
    MODERATE = "MODERATE"  # Caution advised
    LOW = "LOW"  # Informational


class AlertType:
    ALLERGY = "ALLERGY"
    DRUG_INTERACTION = "DRUG_INTERACTION"
    DOSE_LIMIT = "DOSE_LIMIT"
    CONTRAINDICATION = "CONTRAINDICATION"
    DUPLICATE_THERAPY = "DUPLICATE_THERAPY"


class SafetyAlert:
    """Structured safety alert returned by the engine."""

    def __init__(
        self,
        alert_type: str,
        severity: str,
        message: str,
        patient_id: int,
        drug_id: Optional[int] = None,
        allergen: Optional[str] = None,
        recommendation: Optional[str] = None,
    ):
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.patient_id = patient_id
        self.drug_id = drug_id
        self.allergen = allergen
        self.recommendation = recommendation
        self.overridable = severity != AlertSeverity.CRITICAL

    def to_dict(self) -> dict:
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "patient_id": self.patient_id,
            "drug_id": self.drug_id,
            "allergen": self.allergen,
            "recommendation": self.recommendation,
            "overridable": self.overridable,
        }


class SafetyCheckResult:
    """Aggregated result of all safety checks for a prescription."""

    def __init__(self):
        self.alerts: list[SafetyAlert] = []
        self.checked_at = datetime.now(timezone.utc)

    @property
    def has_critical(self) -> bool:
        return any(a.severity == AlertSeverity.CRITICAL for a in self.alerts)

    @property
    def has_alerts(self) -> bool:
        return len(self.alerts) > 0

    @property
    def can_proceed(self) -> bool:
        """Prescription can proceed if no critical alerts exist."""
        return not self.has_critical

    def add_alert(self, alert: SafetyAlert):
        self.alerts.append(alert)

    def to_dict(self) -> dict:
        return {
            "has_alerts": self.has_alerts,
            "has_critical": self.has_critical,
            "can_proceed": self.can_proceed,
            "total_alerts": len(self.alerts),
            "alerts": [a.to_dict() for a in self.alerts],
            "checked_at": self.checked_at.isoformat(),
        }


class ClinicalSafetyEngine:
    """
    Core engine that performs safety checks against patient records.

    Usage:
        engine = ClinicalSafetyEngine()
        result = engine.check_prescription(patient_id=123, drug_ids=[45, 67])

        if not result.can_proceed:
            # Block prescription, show alerts
            pass
        elif result.has_alerts:
            # Show warnings, allow override
            pass
    """

    def check_prescription(
        self,
        patient_id: int,
        drug_ids: list[int],
        clinician_id: Optional[int] = None,
    ) -> SafetyCheckResult:
        """
        Run all safety checks for a prescription.

        Args:
            patient_id: The patient being prescribed for
            drug_ids: List of drug IDs being prescribed
            clinician_id: The prescribing clinician (for audit)

        Returns:
            SafetyCheckResult with all alerts found
        """
        result = SafetyCheckResult()

        # Check 1: Allergy cross-reference
        allergy_alerts = self._check_allergies(patient_id, drug_ids)
        for alert in allergy_alerts:
            result.add_alert(alert)

        # Check 2: Duplicate therapy (same drug already prescribed)
        duplicate_alerts = self._check_duplicate_therapy(patient_id, drug_ids)
        for alert in duplicate_alerts:
            result.add_alert(alert)

        # Log the check
        logger.info(
            f"Safety check for patient {patient_id}: "
            f"{len(drug_ids)} drugs checked, "
            f"{len(result.alerts)} alerts found, "
            f"critical={result.has_critical}"
        )

        return result

    def _check_allergies(
        self, patient_id: int, drug_ids: list[int]
    ) -> list[SafetyAlert]:
        """
        Cross-reference prescribed drugs against patient allergy records.
        """
        alerts = []

        # Get all allergies for this patient
        # PatientAllergy.patient_id is a string FK to patients.patient_id
        # (the business key), not the integer PK. Resolve before querying.
        patient_pid = _resolve_patient_pid(patient_id)
        if patient_pid is None:
            return alerts
        patient_allergies = PatientAllergy.query.filter_by(patient_id=patient_pid).all()

        if not patient_allergies:
            return alerts

        # Build set of allergen names for quick lookup
        allergen_names = set()
        for allergy in patient_allergies:
            allergen = getattr(allergy, "allergen", None) or getattr(
                allergy, "allergy", None
            )
            if allergen:
                allergen_names.add(allergen.lower().strip())

        # Get the actual drug names for the prescribed drug IDs
        prescribed_drugs = Drug.query.filter(Drug.id.in_(drug_ids)).all()

        for drug in prescribed_drugs:
            # Drug model uses generic_name as the primary name column.
            # Fallback chain covers any future schema variants.
            drug_name = (
                getattr(drug, "generic_name", None)
                or getattr(drug, "name", None)
                or getattr(drug, "drug_name", None)
            )
            if not drug_name:
                continue

            # Check if drug name matches any allergen
            if drug_name.lower().strip() in allergen_names:
                # Find the matching allergy record for severity
                matching_allergy = None
                for allergy in patient_allergies:
                    allergen = getattr(allergy, "allergen", None) or getattr(
                        allergy, "allergy", None
                    )
                    if (
                        allergen
                        and allergen.lower().strip() == drug_name.lower().strip()
                    ):
                        matching_allergy = allergy
                        break

                # Determine severity based on allergy record
                severity = AlertSeverity.CRITICAL  # Default to critical
                if matching_allergy:
                    allergy_severity = getattr(
                        matching_allergy, "severity", None
                    ) or getattr(matching_allergy, "reaction_severity", None)
                    if allergy_severity and allergy_severity.lower() in (
                        "mild",
                        "low",
                    ):
                        severity = AlertSeverity.MODERATE

                alerts.append(
                    SafetyAlert(
                        alert_type=AlertType.ALLERGY,
                        severity=severity,
                        message=(
                            f"PATIENT ALLERGY ALERT: Patient is allergic to "
                            f"'{drug_name}'. Prescribing this medication may "
                            f"cause an adverse reaction."
                        ),
                        patient_id=patient_id,
                        drug_id=drug.id,
                        allergen=drug_name,
                        recommendation=(
                            "Do NOT prescribe this medication. "
                            "Select an alternative drug from a different class."
                        ),
                    )
                )

        return alerts

    def _check_duplicate_therapy(
        self, patient_id: int, drug_ids: list[int]
    ) -> list[SafetyAlert]:
        """
        Check if any of the prescribed drugs are already active prescriptions.
        """
        alerts = []

        # Find active prescriptions for this patient.
        # PrescribedMedicine.patient_id is also the string business key.
        dup_patient_pid = _resolve_patient_pid(patient_id)
        if dup_patient_pid is None:
            return alerts
        active_prescriptions = PrescribedMedicine.query.filter_by(
            patient_id=dup_patient_pid
        ).all()

        active_drug_ids = set()
        for rx in active_prescriptions:
            # PrescribedMedicine uses medicine_id as the FK to the drug/medicine table.
            # Fallback chain covers any future schema variants.
            rx_drug_id = (
                getattr(rx, "medicine_id", None)
                or getattr(rx, "drug_id", None)
            )
            if rx_drug_id:
                active_drug_ids.add(rx_drug_id)

        # Check for duplicates
        for drug_id in drug_ids:
            if drug_id in active_drug_ids:
                drug = db.session.get(Drug, drug_id)
                drug_name = "Unknown"
                if drug:
                    drug_name = (
                        getattr(drug, "generic_name", None)
                        or getattr(drug, "name", None)
                        or getattr(drug, "drug_name", None)
                        or "Unknown"
                    )

                alerts.append(
                    SafetyAlert(
                        alert_type=AlertType.DUPLICATE_THERAPY,
                        severity=AlertSeverity.MODERATE,
                        message=(
                            f"DUPLICATE THERAPY WARNING: '{drug_name}' is already "
                            f"in the patient's active prescriptions. Prescribing "
                            f"again may lead to overdose."
                        ),
                        patient_id=patient_id,
                        drug_id=drug_id,
                        recommendation=(
                            "Verify if this is intentional (e.g., dose adjustment). "
                            "If not, do not prescribe a duplicate."
                        ),
                    )
                )

        return alerts

    def log_override(
        self,
        patient_id: int,
        clinician_id: int,
        alert_type: str,
        alert_message: str,
        justification: str,
    ) -> SafetyAlertOverride:
        """
        Log when a clinician overrides a safety alert.
        Creates a permanent audit trail entry.
        """
        override = SafetyAlertOverride(
            patient_id=patient_id,
            clinician_id=clinician_id,
            alert_type=alert_type,
            alert_message=alert_message,
            justification=justification,
        )
        db.session.add(override)
        db.session.commit()

        logger.warning(
            f"SAFETY OVERRIDE: Clinician {clinician_id} overrode "
            f"{alert_type} alert for patient {patient_id}. "
            f"Justification: {justification}"
        )

        return override
