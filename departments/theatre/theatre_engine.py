"""
departments/theatre/theatre_engine.py
───────────────────────────────────────
Operating Theatre (OT) & Surgical Operations Engine.

Capabilities:
  - Real-time intraoperative vitals time-series streaming
  - Intraoperative fluid balance calculation (Intake vs. Loss)
  - Clinical safety gates for WHO 3-Stage Surgical Checklist & Instrument Reconciliation
  - PACU Aldrete Recovery Score (>= 9) discharge readiness evaluation
  - OR utilization & surgical safety metrics aggregation
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from departments.models.medicine import TheatreList
from departments.models.theatre import (
    AnaestheticRecord,
    PostOpNote,
    SurgicalInstrumentCount,
    WhoSurgicalChecklist,
)
from extensions import db

logger = logging.getLogger(__name__)


class TheatreOperationsEngine:
    """Core domain logic engine for theatre operations and surgical safety."""

    @staticmethod
    def record_intraop_vitals(
        entry_id: int,
        hr: Optional[int] = None,
        bp_systolic: Optional[int] = None,
        bp_diastolic: Optional[int] = None,
        spo2: Optional[int] = None,
        etco2: Optional[int] = None,
        agent_concentration: Optional[float] = None,
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
    def calculate_fluid_balance(entry_id: int) -> Dict[str, float]:
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
    def evaluate_who_checklist_gate(entry_id: int, target_stage: str) -> Tuple[bool, str]:
        """Verify whether WHO Surgical Safety Checklist gate is passed for target stage transition."""
        checklist = WhoSurgicalChecklist.query.filter_by(theatre_entry_id=entry_id).first()
        stage_clean = (target_stage or "").upper()

        if stage_clean == "INTRA_OP":
            if not checklist or not checklist.sign_in_completed:
                return False, "WHO Checklist Sign In must be completed before entering INTRA_OP stage."
            return True, "WHO Sign In gate passed."

        elif stage_clean == "POST_OP":
            if not checklist or not checklist.sign_out_completed:
                return False, "WHO Checklist Sign Out must be completed before entering POST_OP stage."

            # Verify surgical instrument count reconciliation
            counts = SurgicalInstrumentCount.query.filter_by(theatre_entry_id=entry_id).first()
            if not counts or not counts.count_reconciled:
                return False, "Surgical instrument and sponge count must be reconciled cleanly before POST_OP transfer."

            return True, "WHO Sign Out and Instrument Reconciliation gate passed."

        return True, "No specific gate required for stage."

    @staticmethod
    def evaluate_pacu_discharge_readiness(entry_id: int) -> Dict:
        """Evaluate PACU Aldrete Recovery Score (0-10) discharge threshold (>= 9)."""
        note = PostOpNote.query.filter_by(theatre_entry_id=entry_id).first()
        if not note:
            return {
                "score": 0,
                "is_fit_for_discharge": False,
                "discharge_to": "PACU",
                "breakdown": {"activity": 0, "respiration": 0, "circulation": 0, "consciousness": 0, "spo2": 0},
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
    def get_or_dashboard_metrics() -> Dict:
        """Compute real-time Operating Theatre utilization, active cases, and safety metrics."""
        entries = TheatreList.query.order_by(TheatreList.created_at.desc()).limit(50).all()

        active_cases = []
        who_completed_count = 0
        discrepancy_count = 0
        completed_cases_count = 0

        for e in entries:
            chk = WhoSurgicalChecklist.query.filter_by(theatre_entry_id=e.id).first()
            cnt = SurgicalInstrumentCount.query.filter_by(theatre_entry_id=e.id).first()
            post = PostOpNote.query.filter_by(theatre_entry_id=e.id).first()

            is_chk_complete = chk.is_fully_completed() if chk else False
            if is_chk_complete:
                who_completed_count += 1

            is_discrepancy = False
            if cnt and not cnt.count_reconciled:
                is_discrepancy = True
                discrepancy_count += 1

            if e.status == 1:
                completed_cases_count += 1

            active_cases.append({
                "entry_id": e.id,
                "patient_id": e.patient_id,
                "patient_name": e.patient.name if e.patient else "Unknown",
                "procedure_name": e.procedure.name if e.procedure else "Surgical Procedure",
                "scheduled_time": e.created_at.isoformat() if e.created_at else None,
                "status": "Completed" if e.status == 1 else "Scheduled/In-Progress",
                "who_completed": is_chk_complete,
                "sign_in": chk.sign_in_completed if chk else False,
                "time_out": chk.time_out_completed if chk else False,
                "sign_out": chk.sign_out_completed if chk else False,
                "count_discrepancy": is_discrepancy,
                "aldrete_score": post.total_aldrete_score if post else None,
                "fit_for_pacu_discharge": post.is_fit_for_pacu_discharge() if post else False,
            })

        total_entries = len(entries)
        who_compliance_pct = round((who_completed_count / total_entries * 100), 1) if total_entries > 0 else 100.0

        return {
            "total_cases": total_entries,
            "active_cases_count": total_entries - completed_cases_count,
            "completed_cases_count": completed_cases_count,
            "who_compliance_pct": who_compliance_pct,
            "discrepancy_count": discrepancy_count,
            "or_cases": active_cases,
        }
