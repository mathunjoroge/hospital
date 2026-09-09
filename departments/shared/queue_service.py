"""Phase 3: department queue reader built on Encounter.stage.

Replaces direct PatientWaitingList.seen queries in department index routes.
PatientWaitingList writes continue (dual-write) for one release as a
rollback safety net; Phase 4 removes them.
"""
from sqlalchemy import case

from departments.models.encounter import Encounter

# Which Encounter.stage values count as 'queued' for each department.
DEPARTMENT_STAGES = {
    "nursing": ["REGISTERED"],  # waiting for triage/vitals
    "medicine": ["WAITING_DOCTOR", "IN_CONSULTATION", "AWAITING_RESULTS"],
    "laboratory": ["AWAITING_RESULTS"],
    "pharmacy": ["AWAITING_PHARMACY"],
    "billing": ["AWAITING_BILLING"],
}


def queue_for(department: str):
    """Return ACTIVE encounters currently queued for the given department.

    Returns a list of Encounter objects (each with a `.patient` joined
    relationship) so templates iterating over the list can use the same
    `entry.patient.name` accessors they used on PatientWaitingList rows.
    """
    stages = DEPARTMENT_STAGES.get(department)
    if not stages:
        return []
    return (
        Encounter.query.filter(
            Encounter.status == "ACTIVE",
            Encounter.stage.in_(stages),
        )
        .order_by(
            case(
                (Encounter.esi_level.in_([1, 2]), 0),  # Emergent first
                (Encounter.esi_level.isnot(None), 1),  # Triaged (3-5) next
                else_=2,                               # Untriaged last
            ).asc(),
            Encounter.started_at.asc(),
        )
        .all()
    )


def count_for(department: str) -> int:
    stages = DEPARTMENT_STAGES.get(department)
    if not stages:
        return 0
    return Encounter.query.filter(
        Encounter.status == "ACTIVE",
        Encounter.stage.in_(stages),
    ).count()
