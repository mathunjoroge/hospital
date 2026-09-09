import logging
import uuid
from datetime import datetime, timezone

from extensions import db

logger = logging.getLogger(__name__)

# Late import used only in the relationship; declared here so templates
# iterating over Encounters can access `encounter.patient` directly.


class Encounter(db.Model):
    """
    Represents a single clinical interaction/visit (OPD, IPD, Telehealth, Emergency).
    Acts as the central hub linking clinical orders and financial charges to a specific visit.
    """

    __tablename__ = "encounters"

    id = db.Column(db.Integer, primary_key=True)
    encounter_id = db.Column(
        db.String(36), unique=True, nullable=False, index=True, default=lambda: str(uuid.uuid4())
    )

    # The patient involved in this visit
    patient_id = db.Column(
        db.String(20), db.ForeignKey("patients.patient_id"), nullable=False, index=True
    )

    # Links to the scheduling systems (Nullable to allow walk-ins/emergencies without prior booking)
    appointment_id = db.Column(
        db.String(36), db.ForeignKey("appointments.id"), nullable=True, index=True
    )
    clinic_booking_id = db.Column(
        db.Integer, db.ForeignKey("clinic_bookings.id"), nullable=True, index=True
    )

    # OPD, IPD, EMERGENCY, TELEHEALTH
    encounter_type = db.Column(db.String(20), nullable=False, default="OPD")
    # ACTIVE, DISCHARGED, CANCELLED, ABORTED
    status = db.Column(db.String(20), nullable=False, default="ACTIVE")

    # The attending clinician for this encounter
    provider_id = db.Column(db.String(50), nullable=True, index=True)
    chief_complaint = db.Column(db.Text, nullable=True)

    started_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    ended_at = db.Column(db.DateTime(timezone=True), nullable=True)

    patient = db.relationship("Patient", foreign_keys="Encounter.patient_id",
                              primaryjoin="Encounter.patient_id == Patient.patient_id",
                              lazy="joined", viewonly=True)


    # Clinical work relationships (for visit scoping)
    requested_labs = db.relationship("RequestedLab", backref="encounter", lazy="dynamic")
    requested_images = db.relationship("RequestedImage", backref="encounter", lazy="dynamic")
    prescribed_medicines = db.relationship("PrescribedMedicine", backref="encounter", lazy="dynamic")
    @property
    def seen(self):
        """Backward compat: maps stage to legacy QueueStatus integer for templates."""
        from departments.shared.queue_constants import QueueStatus
        stage_map = {
            "REGISTERED": QueueStatus.WAITING_TRIAGE,
            "WAITING_DOCTOR": QueueStatus.VITALS_DONE,
            "IN_CONSULTATION": QueueStatus.IN_CONSULTATION,
            "AWAITING_RESULTS": QueueStatus.AWAITING_RESULTS,
            "AWAITING_PHARMACY": QueueStatus.AWAITING_PHARMACY,
            "AWAITING_BILLING": QueueStatus.AWAITING_BILLING,
            "DISCHARGED": QueueStatus.DISCHARGED,
        }
        return stage_map.get(self.stage, 0)

    @property
    def last_updated(self):
        """Backward compat for templates expecting .last_updated."""
        return self.started_at

    created_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


    # Visit lifecycle stage (Phase 2). Independent of `status`, which
    # billing/sync.py relies on to scope invoices to a visit.
    stage = db.Column(db.String(30), nullable=True, index=True)
    esi_level = db.Column(db.Integer, nullable=True)  # 1-5 ESI acuity

    ALLOWED_STAGE_TRANSITIONS = {
        None: {"REGISTERED", "WAITING_DOCTOR", "IN_CONSULTATION"},
        "REGISTERED": {"WAITING_DOCTOR", "IN_CONSULTATION", "CANCELLED"},
        "WAITING_DOCTOR": {"IN_CONSULTATION", "CANCELLED"},
        "IN_CONSULTATION": {
            "IN_CONSULTATION",
            "AWAITING_RESULTS",
            "AWAITING_PHARMACY",
            "AWAITING_BILLING",
        },
        "AWAITING_RESULTS": {
            "IN_CONSULTATION",
            "AWAITING_RESULTS",
            "AWAITING_PHARMACY",
            "AWAITING_BILLING",
        },
        "AWAITING_PHARMACY": {"IN_CONSULTATION", "AWAITING_BILLING"},
        "AWAITING_BILLING": {"DISCHARGED"},
        "DISCHARGED": set(),
        "CANCELLED": set(),
    }

    def set_stage(self, new_stage: str) -> bool:
        """Advance the visit stage; refuses illegal transitions."""
        allowed = self.ALLOWED_STAGE_TRANSITIONS.get(
            self.stage, self.ALLOWED_STAGE_TRANSITIONS.get(None, set())
        )
        if new_stage not in allowed:
            logger.warning(
                "ENCOUNTER STAGE TRANSITION REFUSED: %s -> %s (encounter %s)",
                self.stage,
                new_stage,
                self.id,
            )
            return False
        self.stage = new_stage
        return True

    def close(self):
        """Marks the encounter as completed/discharged."""
        self.status = "DISCHARGED"
        self.ended_at = datetime.now(timezone.utc)
        self.stage = "DISCHARGED"
