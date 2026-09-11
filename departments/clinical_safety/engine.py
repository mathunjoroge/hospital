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

from departments.clinical_safety.models import SafetyAlertOverride
from departments.models.medicine import PrescribedMedicine
from departments.models.nursing import NursingNote
from departments.models.pharmacy import Drug
from departments.models.records import Patient, PatientAllergy
from extensions import db

# Allergy groups for cross-reactivity checking (duplicate from prescribe.py to avoid circular import)

# NOTE: Must be kept in sync with departments/medicine/prescribe.py

ALLERGY_GROUPS = {

    "penicillin": [

        "amoxicillin",

        "ampicillin",

        "penicillin",

        "augmentin",

        "piperacillin",

        "amoxil",

    ],

    "sulfa": ["bactrim", "cotrimoxazole", "sulfamethoxazole", "septrin"],

    "nsaid": [

        "ibuprofen",

        "diclofenac",

        "naproxen",

        "aspirin",

        "indomethacin",

        "brufen",

    ],

    "macrolide": ["azithromycin", "erythromycin", "clarithromycin"],

}



# Drug-Drug Interaction Warning Rules (pairs -> severity, warning message)

# NOTE: Must be kept in sync with departments/medicine/prescribe.py

KNOWN_INTERACTIONS = [

    (

        {"warfarin", "aspirin"},

        "CRITICAL",

        "High risk of major gastrointestinal hemorrhage and severe bleeding.",

    ),

    (

        {"warfarin", "ibuprofen"},

        "HIGH",

        "Increased risk of bleeding and gastric mucosal ulceration.",

    ),

    (

        {"lisinopril", "spironolactone"},

        "HIGH",

        "Severe hyperkalemia risk; requires close serum potassium monitoring.",

    ),

    (

        {"ciprofloxacin", "antacid"},

        "MEDIUM",

        "Chelation reduces ciprofloxacin bioavailability and therapeutic efficacy.",

    ),

    (

        {"metformin", "contrast"},

        "HIGH",

        "Risk of contrast-induced acute renal failure and metformin lactic acidosis.",

    ),




    ({"simvastatin", "clarithromycin"}, "HIGH", "Risk of rhabdomyolysis"),

    ({"simvastatin", "erythromycin"}, "HIGH", "Risk of rhabdomyolysis"),

    ({"atorvastatin", "clarithromycin"}, "HIGH", "Risk of rhabdomyolysis"),

    ({"metformin", "contrast dye"}, "HIGH", "Risk of lactic acidosis"),

    ({"ace inhibitors", "nsaids"}, "MODERATE", "May reduce kidney function"),

    ({"diuretics", "lithium"}, "MODERATE", "Risk of lithium toxicity"),

    ({"ssris", "nsaids"}, "MODERATE", "Increased risk of bleeding"),

]


logger = logging.getLogger(__name__)


def _resolve_patient_pid(patient_id: int) -> "str | None":
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
        drug_id: int | None = None,
        allergen: str | None = None,
        recommendation: str | None = None,
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
        clinician_id: int | None = None,
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

    def check_by_names(self, patient_id: str, medication_names: list[str]) -> dict:
        """
        Check medication names against patient allergies and drug-drug interactions.
        This method mirrors the original check_drug_safety logic from prescribe.py
        to maintain backward compatibility.

        Args:
            patient_id: The patient identifier (string)
            medication_names: List of medication names to check

        Returns:
            dict with keys: has_warnings (bool), critical_block (bool), alerts (list)
        """
        alerts = []
        new_meds_lower = [m.lower().strip() for m in medication_names if m]

        # 1a. Fetch structured patient allergies from PatientAllergy model
        structured_allergies = PatientAllergy.query.filter_by(patient_id=patient_id).all()
        structured_allergen_names = [
            a.allergen.lower().strip() for a in structured_allergies if a.allergen
        ]

        # 1b. Fetch patient allergy history from NursingNotes free text
        notes = NursingNote.query.filter_by(patient_id=patient_id).all()
        documented_allergies = list(structured_allergen_names)
        for n in notes:
            if n.allergies:
                documented_allergies.extend(
                    [a.strip().lower() for a in n.allergies.split(",")]
                )

        # Check allergy cross-reactivity and direct matches
        for drug in new_meds_lower:
            # Direct allergen match from PatientAllergy registry or free-text
            for allergen in documented_allergies:
                if allergen in drug or drug in allergen:
                    alerts.append(
                        {
                            "type": "ALLERGY_WARNING",
                            "severity": "CRITICAL",
                            "drug": drug,
                            "message": f"PATIENT ALLERGY ALERT: Patient has documented allergy to '{allergen.title()}'! Drug '{drug.title()}' is contraindicated.",
                        }
                    )
                    break
            else:
                # Check group cross-reactivity
                for group_name, drug_list in ALLERGY_GROUPS.items():
                    if any(d in drug for d in drug_list):  # noqa: SIM102
                        if any(
                            group_name in allergy or any(d in allergy for d in drug_list)
                            for allergy in documented_allergies
                        ):
                            alerts.append(
                                {
                                    "type": "ALLERGY_WARNING",
                                    "severity": "CRITICAL",
                                    "drug": drug,
                                    "message": f"PATIENT ALLERGY ALERT: Patient is allergic to {group_name.upper()} group! Drug '{drug.title()}' is contraindicated.",
                                }
                            )
                            break

        # 2. Check Drug-Drug Interactions (DDI)
        # Fetch current active prescribed meds for patient
        active_prescriptions = PrescribedMedicine.query.filter_by(
            patient_id=patient_id
        ).all()
        current_meds_lower = [
            p.medicine.generic_name.lower().strip()
            for p in active_prescriptions
            if p.medicine and p.medicine.generic_name
        ]
        all_meds = set(new_meds_lower + current_meds_lower)

        for drug_set, severity, msg in KNOWN_INTERACTIONS:
            matched = [d for d in drug_set if any(d in med for med in all_meds)]
            if len(matched) >= 2:
                alerts.append(
                    {
                        "type": "DRUG_INTERACTION",
                        "severity": severity,
                        "drugs": matched,
                        "message": f"DRUG INTERACTION WARNING [{severity}]: {' + '.join([m.title() for m in matched])} — {msg}",
                    }
                )

        return {
            "has_warnings": len(alerts) > 0,
            "critical_block": any(a["severity"] == "CRITICAL" for a in alerts),
            "alerts": alerts,
        }


