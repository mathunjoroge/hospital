"""
Schedule and Queue Management Engine.

Handles appointment booking, slot availability (preventing double-booking),
and live waiting room queue management.
"""

import logging
from datetime import datetime, timedelta

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
        self, provider_id: int, start_time: datetime, duration_minutes: int = 30
    ) -> bool:
        """
        Checks if a provider has a free slot at the requested time.
        Prevents double-booking by checking for overlapping active appointments.
        """
        end_time = start_time + timedelta(minutes=duration_minutes)

        # Look for any appointment that overlaps with the requested window
        overlapping = Appointment.query.filter(
            Appointment.provider_id == provider_id,
            Appointment.status.notin_(["CANCELLED", "NO_SHOW"]),
            Appointment.scheduled_start < end_time,
            Appointment.scheduled_end > start_time,
        ).first()

        return overlapping is None

    def book_appointment(
        self,
        patient_id: int,
        provider_id: int,
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
            patient_id=patient_id,
            provider_id=provider_id,
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

    def check_in(self, appointment_id: str) -> Appointment | None:
        """
        Moves a patient from SCHEDULED to CHECKED_IN, updating the queue.
        """
        appt = Appointment.query.get(appointment_id)
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
        db.session.commit()
        logger.info("PATIENT CHECKED IN: Appointment %s", appointment_id)
        return appt

    def mark_no_show(self, appointment_id: str) -> Appointment | None:
        """
        Marks a patient as a no-show, freeing up the provider's schedule.
        """
        appt = Appointment.query.get(appointment_id)
        if not appt:
            return None

        appt.mark_no_show()
        db.session.commit()
        logger.info("NO-SHOW RECORDED: Appointment %s", appointment_id)
        return appt

    def get_provider_schedule(
        self, provider_id: int, date: datetime
    ) -> list[Appointment]:
        """
        Retrieves all appointments for a provider on a specific date.
        """
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        return (
            Appointment.query.filter(
                Appointment.provider_id == provider_id,
                Appointment.scheduled_start >= start_of_day,
                Appointment.scheduled_start < end_of_day,
            )
            .order_by(Appointment.scheduled_start.asc())
            .all()
        )

    def get_live_queue(self, provider_id: int) -> list[Appointment]:
        """
        Retrieves the current waiting room queue for a provider.
        Ordered by check-in time (approximated by updated_at timestamp).
        """
        return (
            Appointment.query.filter(
                Appointment.provider_id == provider_id,
                Appointment.status == "CHECKED_IN",
            )
            .order_by(Appointment.updated_at.asc())
            .all()
        )
