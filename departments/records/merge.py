from datetime import datetime, timezone

from departments.models.admin import Log
from departments.models.records import Patient, PatientMerge
from extensions import db


def find_duplicate_candidates(patient):
    """
    Search for potential duplicate patient records based on:
    - Matching date of birth and normalized phone number (HIGH confidence)
    - Matching date of birth and name components (MEDIUM confidence)
    """
    if not patient or not patient.is_active:
        return []

    query = Patient.query.filter(Patient.id != patient.id, Patient.is_active.is_(True))

    candidates = []
    normalized_phone = "".join(filter(str.isdigit, patient.contact or ""))
    dob_matches = query.filter(Patient.date_of_birth == patient.date_of_birth).all()

    for cand in dob_matches:
        cand_phone = "".join(filter(str.isdigit, cand.contact or ""))

        # Phone + DOB match
        if normalized_phone and cand_phone and normalized_phone[-9:] == cand_phone[-9:]:
            candidates.append(
                {
                    "patient": cand,
                    "confidence": "HIGH",
                    "reason": "Matching date of birth and phone number",
                }
            )
            continue

        # Name + DOB match
        if patient.name and cand.name:
            p_name = patient.name.lower().split()
            c_name = cand.name.lower().split()
            common_names = set(p_name).intersection(set(c_name))
            if len(common_names) >= 1:
                candidates.append(
                    {
                        "patient": cand,
                        "confidence": "MEDIUM",
                        "reason": f"Matching date of birth and name: {', '.join(common_names)}",
                    }
                )

    return candidates


def merge_patient_records(source_patient_id, target_patient_id, user_id, notes=None):
    """
    Merge source patient into target patient:
    - Soft-delete source_patient (is_active = False, deleted_at = now)
    - Record PatientMerge audit log
    """
    source = Patient.query.filter_by(patient_id=source_patient_id).first()
    if source is None:
        raise ValueError(f"Source patient {source_patient_id} not found.")
    target = Patient.query.filter_by(patient_id=target_patient_id).first()
    if target is None:
        raise ValueError(f"Target patient {target_patient_id} not found.")

    if not source.is_active:
        raise ValueError(
            f"Source patient {source_patient_id} is already inactive or merged."
        )

    merge_log = PatientMerge(
        source_patient_id=source_patient_id,
        target_patient_id=target_patient_id,
        merged_by=user_id,
        merged_at=datetime.now(timezone.utc),
        notes=notes,
    )
    db.session.add(merge_log)

    source.is_active = False
    source.deleted_at = datetime.now(timezone.utc)

    db.session.add(
        Log(
            level="INFO",
            message=f"Patient {source_patient_id} merged into {target_patient_id} by user ID {user_id}",
            user_id=user_id,
            source="records",
        )
    )

    db.session.commit()
    return target
