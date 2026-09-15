"""Phase 3: department queue reader built on Encounter.stage.

Replaces direct PatientWaitingList.seen queries in department index routes.
PatientWaitingList writes continue (dual-write) for one release as a
rollback safety net; Phase 4 removes them.
"""
from sqlalchemy import case

from departments.models.encounter import Encounter

# Which Encounter.stage values count as 'queued' for each department.
DEPARTMENT_STAGES = {
    "billing": ["REGISTERED_UNPAID", "AWAITING_FINAL_BILLING", "AWAITING_BILLING"],
    "billing_registration": ["REGISTERED_UNPAID"],
    "billing_settlement": ["AWAITING_FINAL_BILLING", "AWAITING_BILLING"],
    "nursing": ["WAITING_TRIAGE", "REGISTERED", "REGISTERED_UNPAID"],
    "medicine": ["WAITING_DOCTOR", "IN_CONSULTATION"],
    "medicine_results": ["WAITING_DOCTOR_RESULTS"],
    "laboratory": ["AWAITING_LAB", "AWAITING_RESULTS"],
    "imaging": ["AWAITING_IMAGING"],
    "pharmacy": ["AWAITING_PHARMACY"],
    "ward": ["ADMITTED", "PRE_OP", "POST_OP"],
    "inpatient": ["ADMITTED", "PRE_OP", "POST_OP"],
}


def queue_for(department: str, provider_id: str = None):
    """Return ACTIVE encounters currently queued for the given department.

    Returns a list of Encounter objects (each with a `.patient` joined
    relationship) so templates iterating over the list can use the same
    `entry.patient.name` accessors they used on PatientWaitingList rows.
    """
    stages = DEPARTMENT_STAGES.get(department)
    if not stages:
        return []
    query = Encounter.query.filter(
        Encounter.status == "ACTIVE",
        Encounter.stage.in_(stages),
    )
    if provider_id and provider_id != "all":
        query = query.filter(Encounter.provider_id == str(provider_id))

    return query.order_by(
        case(
            (Encounter.esi_level.in_([1, 2]), 0),  # Emergent first
            (Encounter.esi_level.isnot(None), 1),  # Triaged (3-5) next
            else_=2,                               # Untriaged last
        ).asc(),
        Encounter.started_at.asc(),
    ).all()


def count_for(department: str, provider_id: str = None) -> int:
    stages = DEPARTMENT_STAGES.get(department)
    if not stages:
        return 0
    query = Encounter.query.filter(
        Encounter.status == "ACTIVE",
        Encounter.stage.in_(stages),
    )
    if provider_id and provider_id != "all":
        query = query.filter(Encounter.provider_id == str(provider_id))
    return query.count()
