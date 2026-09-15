"""
Schedule and Queue Management Engine.

Handles appointment booking, slot availability (preventing double-booking),
and live waiting room queue management.
"""

import logging
from datetime import datetime, timedelta, timezone

from departments.models.encounter import Encounter
from extensions import db

from .models import Appointment

logger = logging.getLogger(__name__)


class ScheduleEngine:
    """
    Core engine for managing provider schedules and patient queues.

    Usage:
        engine = ScheduleEngine()
        is_free = engine.check_availability(provider_id=1, start_time=dt)
        if is_free:
            appt = engine.book_appointment(patient_id=5, provider_id=1, start_time=dt)
    """

    def check_availability(
        self, provider_id: str | int, start_time: datetime, duration_minutes: int = 30
    ) -> bool:
        """
        Checks if a provider has a free slot at the requested time.
        Prevents double-booking by checking for overlapping active appointments.
        """
        end_time = start_time + timedelta(minutes=duration_minutes)

        # Look for any appointment that overlaps with the requested window
        overlapping = Appointment.query.filter(
            Appointment.provider_id == str(provider_id),
            Appointment.status.notin_(["CANCELLED", "NO_SHOW"]),
            Appointment.scheduled_start < end_time,
            Appointment.scheduled_end > start_time,
        ).first()

        return overlapping is None

    def book_appointment(
        self,
        patient_id: str | int,
        provider_id: str | int,
        start_time: datetime,
        duration_minutes: int = 30,
        appointment_type: str = "CONSULTATION",
        reason: str | None = None,
    ) -> Appointment | None:
        """
        Books an appointment if the slot is available.
        Returns the Appointment object or None if double-booking is attempted.
        """
        if not self.check_availability(provider_id, start_time, duration_minutes):
            logger.warning(
                "BOOKING BLOCKED: Double-booking attempt for provider %s at %s",
                provider_id,
                start_time.isoformat(),
            )
            return None

        end_time = start_time + timedelta(minutes=duration_minutes)

        appt = Appointment(
            patient_id=str(patient_id),
            provider_id=str(provider_id),
            scheduled_start=start_time,
            scheduled_end=end_time,
            appointment_type=appointment_type,
            reason_for_visit=reason,
            status="SCHEDULED",
        )
        db.session.add(appt)
        db.session.commit()

        logger.info(
            "APPOINTMENT BOOKED: Patient %s with Provider %s at %s",
            patient_id,
            provider_id,
            start_time.isoformat(),
        )
        return appt

    def create_walk_in(
        self,
        patient_id: str | int,
        provider_id: str | int = "1",
        reason: str = "Walk-in Registration",
    ) -> Appointment:
        """
        Creates a walk-in appointment pre-checked into the live queue.
        Automatically creates an ACTIVE Encounter.
        """
        now = datetime.now(timezone.utc)
        appt = Appointment(
            patient_id=str(patient_id),
            provider_id=str(provider_id),
            scheduled_start=now,
            scheduled_end=now + timedelta(minutes=30),
            appointment_type="WALK_IN",
            reason_for_visit=reason,
            status="CHECKED_IN",
        )
        db.session.add(appt)

        # Attach to existing active Encounter or auto-create one for walk-in
        existing_enc = (
            Encounter.query.filter_by(patient_id=str(patient_id), status="ACTIVE")
            .order_by(Encounter.started_at.desc())
            .first()
        )
        if existing_enc:
            existing_enc.appointment_id = appt.id
            if not existing_enc.provider_id:
                existing_enc.provider_id = str(provider_id)
        else:
            enc = Encounter(
                patient_id=str(patient_id),
                appointment_id=appt.id,
                provider_id=str(provider_id),
                encounter_type="OPD",
                status="ACTIVE",
                stage="REGISTERED",
            )
            db.session.add(enc)

        db.session.commit()
        logger.info(
            "WALK-IN APPOINTMENT & ENCOUNTER CREATED: Patient %s with Provider %s",
            patient_id,
            provider_id,
        )
        return appt

    def check_in(self, appointment_id: str) -> Appointment | None:
        """
        Moves a patient from SCHEDULED to CHECKED_IN, updating the queue.
        Automatically creates an ACTIVE Encounter for the visit.
        """
        appt = db.session.get(Appointment, appointment_id)
        if not appt:
            return None

        if appt.status != "SCHEDULED":
            logger.warning(
                "CHECK-IN BLOCKED: Appointment %s is already %s",
                appointment_id,
                appt.status,
            )
            return None

        appt.check_in()

        # Auto-create Encounter for this visit
        existing_enc = Encounter.query.filter_by(appointment_id=appt.id).first()
        if not existing_enc:
            enc_type = "OPD"
            if appt.appointment_type in ["IPD", "EMERGENCY", "TELEHEALTH"]:
                enc_type = appt.appointment_type

            enc = Encounter(
                patient_id=appt.patient_id,
                appointment_id=appt.id,
                provider_id=appt.provider_id,
                encounter_type=enc_type,
                status="ACTIVE",
                stage="REGISTERED",
            )
            db.session.add(enc)
            logger.info("ENCOUNTER CREATED for Appointment %s", appointment_id)

        db.session.commit()
        logger.info("PATIENT CHECKED IN: Appointment %s", appointment_id)
        return appt

    def mark_triage_complete(self, patient_id: str | int) -> Appointment | None:
        """
        Bridge from the legacy nursing queue: vitals recorded, patient ready.

        Advances the patient's most recent CHECKED_IN appointment to READY so
        the live-queue dashboard can distinguish 'with nurse' from 'ready for
        doctor'. Idempotent: returns None when nothing is CHECKED_IN.
        """
        appt = (
            Appointment.query.filter_by(patient_id=str(patient_id), status="CHECKED_IN")
            .order_by(Appointment.updated_at.desc())
            .first()
        )
        if not appt:
            return None
        appt.mark_ready()
        enc = (
            Encounter.query.filter_by(patient_id=str(patient_id), status="ACTIVE")
            .order_by(Encounter.started_at.desc())
            .first()
        )
        if enc:
            enc.set_stage("WAITING_DOCTOR")
        db.session.commit()
        logger.info("TRIAGE COMPLETE: Appointment %s is READY", appt.id)
        return appt

    def call_in(self, appointment_id: str) -> Appointment | None:
        """
        Moves a patient from CHECKED_IN to IN_PROGRESS.
        """
        appt = db.session.get(Appointment, appointment_id)
        if not appt:
            return None

        if appt.status not in ("CHECKED_IN", "READY"):
            logger.warning(
                "CALL-IN BLOCKED: Appointment %s is in status %s (expected CHECKED_IN/READY)",
                appointment_id,
                appt.status,
            )
            return None

        appt.start_consultation()
        from departments.models.encounter import Encounter

        enc = (
            Encounter.query.filter_by(appointment_id=appt.id).first()
            or Encounter.query.filter_by(patient_id=appt.patient_id, status="ACTIVE").first()
        )
        if enc:
            enc.set_stage("IN_CONSULTATION")
        db.session.commit()
        logger.info("PATIENT CALLED IN: Appointment %s", appointment_id)
        return appt

    def mark_no_show(self, appointment_id: str) -> Appointment | None:
        """
        Marks a patient as a no-show, freeing up the provider's schedule.
        """
        appt = db.session.get(Appointment, appointment_id)
        if not appt:
            return None

        appt.mark_no_show()
        from departments.models.encounter import Encounter
        enc = Encounter.query.filter(
            (Encounter.appointment_id == appt.id)
            | ((Encounter.patient_id == appt.patient_id) & (Encounter.status == "ACTIVE"))
        ).first()
        if enc:
            enc.status = "CANCELLED"
            enc.stage = "CANCELLED"
            enc.ended_at = datetime.now(timezone.utc)
        db.session.commit()
        logger.info("NO-SHOW RECORDED: Appointment %s", appointment_id)
        return appt

    def get_provider_schedule(
        self, provider_id: str | int, date: datetime
    ) -> list[Appointment]:
        """
        Retrieves all appointments for a provider on a specific date.
        """
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        return (
            Appointment.query.filter(
                Appointment.provider_id == str(provider_id),
                Appointment.scheduled_start >= start_of_day,
                Appointment.scheduled_start < end_of_day,
            )
            .order_by(Appointment.scheduled_start.asc())
            .all()
        )

    def get_live_queue(self, provider_id: str | int | None = None) -> list[dict]:
        """
        Retrieves the current waiting room queue for a provider (or all providers if None or 'all').
        Uses the unified patient flow queue system to show patients across all stages.
        Returns formatted dict data for the dashboard.
        """
        from departments.shared import queue_service  # Fixed import

        # Get all encounters in the patient flow pipeline
        # For a specific provider, filter by provider_id in medicine/nursing queues
        if provider_id and str(provider_id).lower() != "all":
            # Get queue for medicine and nursing for this provider
            medicine_queue = queue_service.queue_for("medicine", str(provider_id))
            nursing_queue = queue_service.queue_for("nursing", str(provider_id))
            # Combine and sort by start time
            all_queue = medicine_queue + nursing_queue
            all_queue.sort(key=lambda e: e.started_at or datetime.min.replace(tzinfo=None))
        else:
            # Get all queues across the pipeline
            billing_registration = queue_service.queue_for("billing_registration")
            nursing = queue_service.queue_for("nursing")
            medicine = queue_service.queue_for("medicine")
            medicine_results = queue_service.queue_for("medicine_results")
            laboratory = queue_service.queue_for("laboratory")
            imaging = queue_service.queue_for("imaging")
            pharmacy = queue_service.queue_for("pharmacy")
            billing_settlement = queue_service.queue_for("billing_settlement")

            # Combine all queues, deduplicating by patient_id
            seen_patients = set()
            all_queue = []
            for q in [billing_registration, nursing, medicine, medicine_results, laboratory, imaging, pharmacy, billing_settlement]:
                for entry in q:
                    if entry.patient_id not in seen_patients:
                        seen_patients.add(entry.patient_id)
                        all_queue.append(entry)

            all_queue.sort(key=lambda e: e.started_at or datetime.min.replace(tzinfo=None))

        # Format the queue for dashboard display
        from departments.models.records import Patient

        now_utc = datetime.now(timezone.utc)
        patient_ids = [a.patient_id for a in all_queue]
        patients = {}
        if patient_ids:
            patient_records = Patient.query.filter(Patient.patient_id.in_(patient_ids)).all()
            patients = {p.patient_id: p.name for p in patient_records}

        formatted_queue = []
        for idx, entry in enumerate(all_queue, 1):
            # Determine stage display and color
            stage = entry.stage or "UNKNOWN"
            wait_mins = 0
            if entry.updated_at:
                dt = entry.updated_at
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                wait_mins = max(0, int((now_utc - dt).total_seconds() / 60))

            # Map stage to display text and color
            stage_info = {
                "REGISTERED_UNPAID": {"text": "Reg Unpaid", "color": "amber"},
                "WAITING_TRIAGE": {"text": "Triage", "color": "amber"},
                "WAITING_DOCTOR": {"text": "Doctor Queue", "color": "blue"},
                "IN_CONSULTATION": {"text": "In Consult", "color": "green"},
                "AWAITING_LAB": {"text": "Lab Order", "color": "teal"},
                "AWAITING_IMAGING": {"text": "Imaging Order", "color": "teal"},
                "AWAITING_RESULTS": {"text": "Awaiting Results", "color": "teal"},
                "WAITING_DOCTOR_RESULTS": {"text": "Doctor Review", "color": "purple"},
                "AWAITING_PHARMACY": {"text": "Pharmacy", "color": "orange"},
                "AWAITING_FINAL_BILLING": {"text": "Billing", "color": "amber"},
                "AWAITING_BILLING": {"text": "Billing", "color": "amber"},
                "DISCHARGED": {"text": "Discharged", "color": "gray"},
                "CANCELLED": {"text": "Cancelled", "color": "gray"},
            }.get(stage, {"text": stage or "Unknown", "color": "gray"})

            formatted_queue.append(
                {
                    "id": getattr(entry, "appointment_id", None) or entry.id,
                    "appointment_id": getattr(entry, "appointment_id", None),
                    "idx": idx,
                    "patient_id": entry.patient_id,
                    "patient_name": patients.get(entry.patient_id, f"Patient {entry.patient_id}"),
                    "stage": stage,
                    "stage_display": stage_info["text"],
                    "stage_color": f"bg-{stage_info['color']}-100 text-{stage_info['color']}-800",
                    "started_at": entry.started_at.strftime("%H:%M") if entry.started_at else "--:--",
                    "wait_time_mins": wait_mins,
                    "type": entry.encounter_type or "OPD",
                }
            )

        return formatted_queue

