"""
departments/theatre/theatre_engine.py
───────────────────────────────────────
Operating Theatre (OT) & Perioperative / Anesthesia Engine.

Capabilities:
  - Real-time intraoperative vitals time-series streaming
  - Intraoperative fluid balance calculation (Intake vs. Loss)
  - Clinical safety gates for WHO 3-Stage Surgical Checklist & Instrument Reconciliation
  - PACU Aldrete Recovery Score (>= 9) discharge readiness evaluation
  - ASA Physical Status Classification & Operative Mortality Risk Scoring
  - Intraoperative Anesthesia Timeline Event Matrix (Pre-induction -> PACU Transfer)
  - Operating Room (OR) Scheduling, Room Allocation & Conflict Detection
  - OR Utilization & Surgical Safety Dashboard Metrics
"""

import logging
from datetime import datetime, timedelta, timezone

from departments.models.medicine import TheatreList
from departments.models.theatre import (
    AnaestheticRecord,
    PostOpNote,
    SurgicalInstrumentCount,
    WhoSurgicalChecklist,
)
from extensions import db

logger = logging.getLogger(__name__)

# ASA Physical Status Reference Dictionary
ASA_REFERENCE = {
    "ASA I": {
        "title": "Normal Healthy Patient",
        "description": "Healthy, non-smoking, no or minimal alcohol use.",
        "mortality_pct": 0.05,
        "risk_level": "LOW",
    },
    "ASA II": {
        "title": "Mild Systemic Disease",
        "description": "Mild diseases without substantive functional limitations (e.g., controlled HTN, DM, mild lung disease).",
        "mortality_pct": 0.2,
        "risk_level": "MILD",
    },
    "ASA III": {
        "title": "Severe Systemic Disease",
        "description": "Substantive functional limitations (e.g., poorly controlled DM/HTN, COPD, morbid obesity, active hepatitis).",
        "mortality_pct": 1.8,
        "risk_level": "MODERATE",
    },
    "ASA IV": {
        "title": "Severe Systemic Disease - Constant Threat to Life",
        "description": "Recent MI, CVA, TIA, CAD/stents, ongoing cardiac ischemia or severe valve dysfunction, sepsis.",
        "mortality_pct": 7.8,
        "risk_level": "HIGH",
    },
    "ASA V": {
        "title": "Moribund Patient",
        "description": "Not expected to survive without operation (e.g., ruptured abdominal/thoracic aneurysm, massive trauma, intracranial bleed).",
        "mortality_pct": 9.4,
        "risk_level": "CRITICAL",
    },
    "ASA VI": {
        "title": "Brain-Dead Organ Donor",
        "description": "Declared brain-dead patient whose organs are being harvested for donor purposes.",
        "mortality_pct": 100.0,
        "risk_level": "DONOR",
    },
}


class TheatreOperationsEngine:
    """Core domain logic engine for theatre operations, perioperative safety, and anesthesia timeline."""

    @staticmethod
    def evaluate_asa_score(asa_status: str, is_emergency: bool = False) -> dict:
        """
        Evaluate ASA Physical Status classification & compute perioperative risk grade.
        Emergency cases (is_emergency=True or '-E' suffix) double the baseline risk percentage.
        """
        clean_status = (asa_status or "ASA I").strip().upper()
        if clean_status.endswith("-E") or clean_status.endswith(" E"):
            is_emergency = True
            clean_status = clean_status.replace("-E", "").replace(" E", "").strip()

        ref = ASA_REFERENCE.get(clean_status, ASA_REFERENCE["ASA I"])
        base_mortality = ref["mortality_pct"]

        emergency_multiplier = 2.0 if is_emergency else 1.0
        estimated_mortality_pct = round(base_mortality * emergency_multiplier, 2)

        risk_level = ref["risk_level"]
        if is_emergency and risk_level not in ("CRITICAL", "DONOR"):
            risk_level = f"{risk_level}_EMERGENCY"

        return {
            "asa_code": f"{clean_status}{'E' if is_emergency else ''}",
            "base_asa": clean_status,
            "title": ref["title"],
            "description": ref["description"],
            "is_emergency": is_emergency,
            "base_mortality_pct": base_mortality,
            "estimated_mortality_pct": estimated_mortality_pct,
            "risk_level": risk_level,
        }

    @staticmethod
    def record_timeline_event(
        entry_id: int,
        event_type: str,
        notes: str | None = None,
        recorded_by: str | None = None,
    ) -> dict:
        """
        Record a timestamped anesthesia timeline event (e.g., INDUCTION, INTUBATION, INCISION).
        """
        entry = db.session.get(TheatreList, entry_id)
        if not entry:
            raise ValueError(f"Theatre list entry {entry_id} not found.")

        record = AnaestheticRecord.query.filter_by(theatre_entry_id=entry_id).first()
        if not record:
            record = AnaestheticRecord(
                theatre_entry_id=entry.id,
                patient_id=entry.patient_id,
                encounter_id=entry.encounter_id,
            )
            db.session.add(record)

        valid_events = [
            "PRE_INDUCTION",
            "INDUCTION",
            "INTUBATION",
            "INCISION",
            "MAINTENANCE",
            "EMERGENCE",
            "EXTUBATION",
            "PACU_TRANSFER",
        ]

        event_clean = (event_type or "").upper().strip()
        if event_clean not in valid_events:
            event_clean = "MAINTENANCE"

        now_iso = datetime.now(timezone.utc).isoformat()
        event_obj = {
            "timestamp": now_iso,
            "event_type": event_clean,
            "notes": notes or "",
            "recorded_by": recorded_by or "Clinician",
        }

        events = list(record.timeline_events)
        events.append(event_obj)
        record.timeline_events = events

        # Update anesthesia start/end times if applicable
        if event_clean == "INDUCTION" and not record.anaesthesia_start_time:
            record.anaesthesia_start_time = datetime.now(timezone.utc)
        elif event_clean == "PACU_TRANSFER":
            record.anaesthesia_end_time = datetime.now(timezone.utc)

        db.session.commit()
        return event_obj

    @staticmethod
    def get_anesthesia_timeline_matrix(entry_id: int) -> dict:
        """
        Retrieve complete timeline event matrix aligned with vital sign snapshots.
        """
        record = AnaestheticRecord.query.filter_by(theatre_entry_id=entry_id).first()
        if not record:
            return {
                "theatre_entry_id": entry_id,
                "asa_assessment": TheatreOperationsEngine.evaluate_asa_score("ASA I"),
                "events": [],
                "vitals_count": 0,
                "duration_minutes": 0,
            }

        asa_eval = TheatreOperationsEngine.evaluate_asa_score(
            record.asa_status, record.is_emergency
        )

        events = record.timeline_events
        vitals = record.vitals_series

        duration_min = 0
        if record.anaesthesia_start_time and record.anaesthesia_end_time:
            diff = record.anaesthesia_end_time - record.anaesthesia_start_time
            duration_min = int(diff.total_seconds() / 60)

        return {
            "theatre_entry_id": entry_id,
            "asa_assessment": asa_eval,
            "events": events,
            "vitals_series": vitals,
            "vitals_count": len(vitals),
            "duration_minutes": duration_min,
            "technique": record.technique,
            "airway_management": record.airway_management,
        }

    @staticmethod
    def check_room_schedule_conflict(
        or_room: str,
        start_time: datetime,
        duration_minutes: int = 120,
        exclude_entry_id: int | None = None,
    ) -> tuple[bool, str | None]:
        """
        Check whether an OR room schedule has a time collision with an existing booking.
        """
        if not start_time:
            return False, None

        # Normalize start_time to naive UTC for safe comparison
        if start_time.tzinfo is not None:
            start_time = start_time.astimezone(timezone.utc).replace(tzinfo=None)

        end_time = start_time + timedelta(minutes=duration_minutes)

        existing_entries = TheatreList.query.filter(
            TheatreList.or_room == or_room,
            TheatreList.status == 0,  # Scheduled / In-Progress
            TheatreList.scheduled_start_time.isnot(None),
        ).all()

        for entry in existing_entries:
            if exclude_entry_id and entry.id == exclude_entry_id:
                continue

            entry_start = entry.scheduled_start_time
            if entry_start.tzinfo is not None:
                entry_start = entry_start.astimezone(timezone.utc).replace(tzinfo=None)

            entry_duration = entry.estimated_duration_minutes or 120
            entry_end = entry_start + timedelta(minutes=entry_duration)

            # Check overlap: max(start1, start2) < min(end1, end2)
            if max(start_time, entry_start) < min(end_time, entry_end):
                patient_name = entry.patient.name if entry.patient else entry.patient_id
                conflict_msg = (
                    f"Schedule collision in {or_room}: Case #{entry.id} ({patient_name}) "
                    f"is scheduled from {entry_start.strftime('%H:%M')} to {entry_end.strftime('%H:%M')}."
                )
                return True, conflict_msg

        return False, None

    @staticmethod
    def calculate_or_utilization_metrics() -> dict:
        """
        Compute Operating Theatre utilization, room occupancy rates, ASA distribution, and turnover time.
        """
        rooms = ["OR 1", "OR 2", "OR 3", "Cardiac OR", "Emergency OR"]
        room_stats = {}

        total_cases = TheatreList.query.count()
        completed_cases = TheatreList.query.filter_by(status=1).count()
        scheduled_cases = TheatreList.query.filter_by(status=0).count()

        asa_counts = {
            "ASA I": 0,
            "ASA II": 0,
            "ASA III": 0,
            "ASA IV": 0,
            "ASA V": 0,
            "ASA VI": 0,
        }

        records = AnaestheticRecord.query.all()
        for r in records:
            base_asa = r.asa_status.replace("-E", "").strip()
            if base_asa in asa_counts:
                asa_counts[base_asa] += 1

        for r in rooms:
            cases_in_room = TheatreList.query.filter_by(or_room=r).all()
            in_progress = [c for c in cases_in_room if c.status == 0]
            done = [c for c in cases_in_room if c.status == 1]
            room_stats[r] = {
                "total": len(cases_in_room),
                "active": len(in_progress),
                "completed": len(done),
                "occupancy_status": "OCCUPIED" if len(in_progress) > 0 else "AVAILABLE",
            }

        # Overall OR Occupancy Percentage
        active_rooms = sum(
            1 for r in room_stats.values() if r["occupancy_status"] == "OCCUPIED"
        )
        or_utilization_pct = round((active_rooms / len(rooms)) * 100, 1)

        return {
            "total_cases": total_cases,
            "scheduled_cases": scheduled_cases,
            "completed_cases": completed_cases,
            "or_utilization_pct": or_utilization_pct,
            "room_stats": room_stats,
            "asa_counts": asa_counts,
            "available_rooms": len(rooms) - active_rooms,
        }

    @staticmethod
    def record_intraop_vitals(
        entry_id: int,
        hr: int | None = None,
        bp_systolic: int | None = None,
        bp_diastolic: int | None = None,
        spo2: int | None = None,
        etco2: int | None = None,
        agent_concentration: float | None = None,
    ) -> AnaestheticRecord:
        """Stream a time-stamped intraoperative vital sign snapshot to AnaestheticRecord."""
        entry = db.session.get(TheatreList, entry_id)
        if not entry:
            raise ValueError(f"Theatre list entry {entry_id} not found.")

        record = AnaestheticRecord.query.filter_by(theatre_entry_id=entry_id).first()
        if not record:
            record = AnaestheticRecord(
                theatre_entry_id=entry.id,
                patient_id=entry.patient_id,
                encounter_id=entry.encounter_id,
            )
            db.session.add(record)

        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hr": hr,
            "bp_sys": bp_systolic,
            "bp_dia": bp_diastolic,
            "spo2": spo2,
            "etco2": etco2,
            "agent_conc": agent_concentration,
        }

        current_series = list(record.vitals_series)
        current_series.append(snapshot)
        record.vitals_series = current_series

        db.session.commit()
        return record

    @staticmethod
    def calculate_fluid_balance(entry_id: int) -> dict[str, float]:
        """Compute intraoperative fluid intake vs. loss summary."""
        record = AnaestheticRecord.query.filter_by(theatre_entry_id=entry_id).first()
        if not record:
            return {
                "crystalloids_ml": 0,
                "colloids_ml": 0,
                "blood_products_ml": 0,
                "estimated_blood_loss_ml": 0,
                "urine_output_ml": 0,
                "total_intake_ml": 0,
                "total_loss_ml": 0,
                "net_balance_ml": 0,
            }

        crystalloids = record.crystalloids_ml or 0
        colloids = record.colloids_ml or 0
        blood_products = record.blood_products_ml or 0
        ebl = record.estimated_blood_loss_ml or 0
        urine = record.urine_output_ml or 0

        total_intake = crystalloids + colloids + blood_products
        total_loss = ebl + urine
        net_balance = total_intake - total_loss

        return {
            "crystalloids_ml": crystalloids,
            "colloids_ml": colloids,
            "blood_products_ml": blood_products,
            "estimated_blood_loss_ml": ebl,
            "urine_output_ml": urine,
            "total_intake_ml": total_intake,
            "total_loss_ml": total_loss,
            "net_balance_ml": net_balance,
        }

    @staticmethod
    def evaluate_who_checklist_gate(
        entry_id: int, target_stage: str
    ) -> tuple[bool, str]:
        """Verify whether WHO Surgical Safety Checklist gate is passed for target stage transition."""
        checklist = WhoSurgicalChecklist.query.filter_by(
            theatre_entry_id=entry_id
        ).first()
        stage_clean = (target_stage or "").upper()

        if stage_clean == "INTRA_OP":
            if not checklist or not checklist.sign_in_completed:
                return (
                    False,
                    "WHO Checklist Sign In must be completed before entering INTRA_OP stage.",
                )
            return True, "WHO Sign In gate passed."

        elif stage_clean == "POST_OP":
            if not checklist or not checklist.sign_out_completed:
                return (
                    False,
                    "WHO Checklist Sign Out must be completed before entering POST_OP stage.",
                )

            # Verify surgical instrument count reconciliation
            counts = SurgicalInstrumentCount.query.filter_by(
                theatre_entry_id=entry_id
            ).first()
            if not counts or not counts.count_reconciled:
                return (
                    False,
                    "Surgical instrument and sponge count must be reconciled cleanly before POST_OP transfer.",
                )

            return True, "WHO Sign Out and Instrument Reconciliation gate passed."

        return True, "No specific gate required for stage."

    @staticmethod
    def evaluate_pacu_discharge_readiness(entry_id: int) -> dict:
        """Evaluate PACU Aldrete Recovery Score (0-10) discharge threshold (>= 9)."""
        note = PostOpNote.query.filter_by(theatre_entry_id=entry_id).first()
        if not note:
            return {
                "score": 0,
                "is_fit_for_discharge": False,
                "discharge_to": "PACU",
                "breakdown": {
                    "activity": 0,
                    "respiration": 0,
                    "circulation": 0,
                    "consciousness": 0,
                    "spo2": 0,
                },
            }

        score = note.total_aldrete_score
        fit = note.is_fit_for_pacu_discharge()

        return {
            "score": score,
            "is_fit_for_discharge": fit,
            "discharge_to": note.discharge_to or "PACU",
            "breakdown": {
                "activity": note.aldrete_activity or 0,
                "respiration": note.aldrete_respiration or 0,
                "circulation": note.aldrete_circulation or 0,
                "consciousness": note.aldrete_consciousness or 0,
                "spo2": note.aldrete_spo2 or 0,
            },
        }

    @staticmethod
    def get_or_dashboard_metrics() -> dict:
        """Compute real-time Operating Theatre utilization, active cases, and safety metrics."""
        entries = (
            TheatreList.query.order_by(TheatreList.created_at.desc()).limit(50).all()
        )

        active_cases = []
        who_completed_count = 0
        discrepancy_count = 0
        completed_cases_count = 0

        for e in entries:
            chk = WhoSurgicalChecklist.query.filter_by(theatre_entry_id=e.id).first()
            cnt = SurgicalInstrumentCount.query.filter_by(theatre_entry_id=e.id).first()
            post = PostOpNote.query.filter_by(theatre_entry_id=e.id).first()
            ana = AnaestheticRecord.query.filter_by(theatre_entry_id=e.id).first()

            is_chk_complete = chk.is_fully_completed() if chk else False
            if is_chk_complete:
                who_completed_count += 1

            is_discrepancy = False
            if cnt and not cnt.count_reconciled:
                is_discrepancy = True
                discrepancy_count += 1

            if e.status == 1:
                completed_cases_count += 1

            asa_eval = TheatreOperationsEngine.evaluate_asa_score(
                ana.asa_status if ana else "ASA I",
                ana.is_emergency if ana else False,
            )

            active_cases.append(
                {
                    "entry_id": e.id,
                    "patient_id": e.patient_id,
                    "patient_name": e.patient.name if e.patient else "Unknown",
                    "procedure_name": e.procedure.name
                    if e.procedure
                    else "Surgical Procedure",
                    "or_room": e.or_room or "OR 1",
                    "scheduled_time": e.scheduled_start_time.isoformat()
                    if e.scheduled_start_time
                    else (e.created_at.isoformat() if e.created_at else None),
                    "duration_min": e.estimated_duration_minutes or 120,
                    "status": "Completed" if e.status == 1 else "Scheduled/In-Progress",
                    "who_completed": is_chk_complete,
                    "sign_in": chk.sign_in_completed if chk else False,
                    "time_out": chk.time_out_completed if chk else False,
                    "sign_out": chk.sign_out_completed if chk else False,
                    "count_discrepancy": is_discrepancy,
                    "aldrete_score": post.total_aldrete_score if post else None,
                    "fit_for_pacu_discharge": post.is_fit_for_pacu_discharge()
                    if post
                    else False,
                    "asa_status": asa_eval["asa_code"],
                    "asa_risk": asa_eval["risk_level"],
                    "asa_mortality_pct": asa_eval["estimated_mortality_pct"],
                }
            )

        total_entries = len(entries)
        who_compliance_pct = (
            round((who_completed_count / total_entries * 100), 1)
            if total_entries > 0
            else 100.0
        )
        util_metrics = TheatreOperationsEngine.calculate_or_utilization_metrics()

        return {
            "total_cases": total_entries,
            "active_cases_count": total_entries - completed_cases_count,
            "completed_cases_count": completed_cases_count,
            "who_compliance_pct": who_compliance_pct,
            "discrepancy_count": discrepancy_count,
            "or_utilization_pct": util_metrics["or_utilization_pct"],
            "room_stats": util_metrics["room_stats"],
            "asa_counts": util_metrics["asa_counts"],
            "or_cases": active_cases,
        }
