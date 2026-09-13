"""
departments/medicine/adt_engine.py
───────────────────────────────────
Inpatient Bed Management & HL7 v2 ADT Event Flow Engine (Gap #12 Roadmap).
Handles ADT^A01 (Admit), ADT^A02 (Transfer), ADT^A03 (Discharge), ADT^A08 (Update),
bed cleaning/turnaround state machine (AVAILABLE -> OCCUPIED -> DIRTY -> CLEANING -> AVAILABLE),
and real-time ward bed matrix aggregations.
"""

from datetime import datetime, timezone
import logging
from typing import Any

from extensions import db
from departments.audit import log_audit_event
from departments.models.encounter import Encounter
from departments.models.medicine import (
    AdmittedPatient,
    ADTLog,
    Bed,
    Ward,
    WardBedHistory,
    WardRoom,
)
from departments.models.records import Patient

logger = logging.getLogger(__name__)

VALID_BED_STATUSES = {"AVAILABLE", "OCCUPIED", "DIRTY", "CLEANING", "MAINTENANCE"}


class ADTEngine:
    """Operations engine for Inpatient Bed Management and HL7 v2 ADT Events."""

    @staticmethod
    def format_hl7_adt_payload(
        event_type: str,
        patient_id: str,
        patient_name: str,
        ward_name: str,
        bed_number: str,
        timestamp: datetime,
    ) -> str:
        """Construct standard HL7 v2 ADT message segment string."""
        ts_str = timestamp.strftime("%Y%m%d%H%M%S")
        msh = f"MSH|^~\\&|HIMS|HOSPITAL|RECEIVER|FACILITY|{ts_str}||ADT^{event_type}|{patient_id}_{ts_str}|P|2.5"
        pid = f"PID|1||{patient_id}^^^HOSPITAL||{patient_name}|||||||||||||"
        pv1 = f"PV1|1|I|{ward_name}^{bed_number}^^HOSPITAL||||||||||||||||{patient_id}|||||||||||||||||||||||||{ts_str}"
        return f"{msh}\r{pid}\r{pv1}"

    @classmethod
    def admit_patient_a01(
        cls,
        patient_id: str,
        ward_id: int,
        room_id: int,
        bed_id: int,
        admission_criteria: str,
        user_id: int | None = None,
    ) -> tuple[AdmittedPatient, ADTLog]:
        """
        Execute ADT^A01 Patient Inpatient Admission.
        Allocates bed, sets bed status to OCCUPIED, updates ward occupancy count,
        creates IPD Encounter and HL7 ADTLog.
        """
        patient = Patient.query.filter_by(patient_id=patient_id).first()
        if not patient:
            raise ValueError(f"Patient ID {patient_id} not found.")

        ward = db.session.get(Ward, ward_id)
        if not ward:
            raise ValueError(f"Ward ID {ward_id} not found.")

        room = db.session.query(WardRoom).filter_by(id=room_id, ward_id=ward_id).first()
        if not room:
            raise ValueError(f"Room ID {room_id} not found in Ward {ward.name}.")

        bed = db.session.query(Bed).filter_by(id=bed_id, room_id=room_id).first()
        if not bed:
            raise ValueError(f"Bed ID {bed_id} not found in Room {room.room_number}.")

        if bed.status != "AVAILABLE" or bed.occupied:
            raise ValueError(f"Bed {bed.bed_number} is not available for admission (Current status: {bed.status}).")

        now = datetime.now(timezone.utc)

        # 1. Create AdmittedPatient record
        admission = AdmittedPatient(
            patient_id=patient_id,
            ward_id=ward_id,
            room_id=room_id,
            bed_id=bed_id,
            admission_criteria=admission_criteria or "Inpatient admission",
            admitted_by=user_id or 1,
            admitted_on=now,
        )
        db.session.add(admission)

        # 2. Update Bed state
        bed.occupied = True
        bed.status = "OCCUPIED"

        # 3. Recalculate Ward Occupancy
        ward.occupied_beds = db.session.query(Bed).join(WardRoom).filter(
            WardRoom.ward_id == ward_id, Bed.occupied.is_(True)
        ).count() + 1

        # 4. Record WardBedHistory
        history = WardBedHistory(
            ward_id=ward_id,
            patient_id=patient_id,
            action="Admit",
            timestamp=now,
        )
        db.session.add(history)

        # 5. Create IPD Encounter
        encounter = Encounter(
            patient_id=patient_id,
            encounter_type="IPD",
            status="ACTIVE",
            stage="ADMITTED",
            started_at=now,
        )
        db.session.add(encounter)

        # 6. Generate HL7 ADT^A01 Log
        hl7_msg = cls.format_hl7_adt_payload(
            event_type="A01",
            patient_id=patient_id,
            patient_name=patient.name,
            ward_name=ward.name,
            bed_number=bed.bed_number,
            timestamp=now,
        )

        adt_log = ADTLog(
            event_type="A01",
            patient_id=patient_id,
            admission=admission,
            to_ward_id=ward_id,
            to_bed_id=bed_id,
            user_id=user_id,
            hl7_message=hl7_msg,
            created_at=now,
        )
        db.session.add(adt_log)

        log_audit_event(
            db.session,
            level="INFO",
            message=f"ADT^A01 Admit: Patient {patient_id} admitted to Ward {ward.name}, Bed {bed.bed_number}",
            user_id=user_id,
            source="adt_engine",
        )

        db.session.commit()
        return admission, adt_log

    @classmethod
    def transfer_patient_a02(
        cls,
        admission_id: int,
        new_ward_id: int,
        new_room_id: int,
        new_bed_id: int,
        user_id: int | None = None,
        reason: str | None = None,
    ) -> tuple[AdmittedPatient, ADTLog]:
        """
        Execute ADT^A02 Patient Transfer.
        Moves patient from old ward/bed to new ward/bed, sets previous bed to DIRTY,
        assigns new bed as OCCUPIED, updates ward occupancy counts, and records ADTLog.
        """
        admission = db.session.get(AdmittedPatient, admission_id)
        if not admission or admission.discharged_on is not None:
            raise ValueError(f"Active admission ID {admission_id} not found.")

        new_ward = db.session.get(Ward, new_ward_id)
        if not new_ward:
            raise ValueError(f"Destination Ward ID {new_ward_id} not found.")

        new_room = db.session.query(WardRoom).filter_by(id=new_room_id, ward_id=new_ward_id).first()
        if not new_room:
            raise ValueError(f"Destination Room ID {new_room_id} not found in Ward {new_ward.name}.")

        new_bed = db.session.query(Bed).filter_by(id=new_bed_id, room_id=new_room_id).first()
        if not new_bed:
            raise ValueError(f"Destination Bed ID {new_bed_id} not found.")

        if new_bed.status != "AVAILABLE" or new_bed.occupied:
            raise ValueError(f"Destination Bed {new_bed.bed_number} is not available for transfer.")

        old_ward_id = admission.ward_id
        old_bed_id = admission.bed_id
        old_bed = db.session.get(Bed, old_bed_id) if old_bed_id else None

        now = datetime.now(timezone.utc)

        # 1. Release previous bed -> set status DIRTY for housekeeping
        if old_bed:
            old_bed.occupied = False
            old_bed.status = "DIRTY"

        # 2. Occupy new bed
        new_bed.occupied = True
        new_bed.status = "OCCUPIED"

        # 3. Update Admission details
        admission.ward_id = new_ward_id
        admission.room_id = new_room_id
        admission.bed_id = new_bed_id

        # 4. Recalculate Ward Occupancies
        if old_ward_id:
            old_ward = db.session.get(Ward, old_ward_id)
            if old_ward:
                old_ward.occupied_beds = db.session.query(Bed).join(WardRoom).filter(
                    WardRoom.ward_id == old_ward_id, Bed.occupied.is_(True)
                ).count()

        new_ward.occupied_beds = db.session.query(Bed).join(WardRoom).filter(
            WardRoom.ward_id == new_ward_id, Bed.occupied.is_(True)
        ).count()

        # 5. History & ADT Log
        history = WardBedHistory(
            ward_id=new_ward_id,
            patient_id=admission.patient_id,
            action="Transfer",
            timestamp=now,
        )
        db.session.add(history)

        patient = admission.patient
        hl7_msg = cls.format_hl7_adt_payload(
            event_type="A02",
            patient_id=admission.patient_id,
            patient_name=patient.name if patient else "Patient",
            ward_name=new_ward.name,
            bed_number=new_bed.bed_number,
            timestamp=now,
        )

        adt_log = ADTLog(
            event_type="A02",
            patient_id=admission.patient_id,
            admission_id=admission.id,
            from_ward_id=old_ward_id,
            from_bed_id=old_bed_id,
            to_ward_id=new_ward_id,
            to_bed_id=new_bed_id,
            user_id=user_id,
            hl7_message=hl7_msg,
            created_at=now,
        )
        db.session.add(adt_log)

        log_audit_event(
            db.session,
            level="INFO",
            message=f"ADT^A02 Transfer: Patient {admission.patient_id} transferred to Ward {new_ward.name}, Bed {new_bed.bed_number}",
            user_id=user_id,
            source="adt_engine",
        )

        db.session.commit()
        return admission, adt_log

    @classmethod
    def discharge_patient_a03(
        cls,
        admission_id: int,
        discharge_summary: str | None = None,
        user_id: int | None = None,
    ) -> tuple[AdmittedPatient, ADTLog]:
        """
        Execute ADT^A03 Patient Discharge.
        Marks admission discharged, frees bed and transitions status to DIRTY,
        updates ward occupancy counts, closes IPD Encounter, and records ADTLog.
        """
        admission = db.session.get(AdmittedPatient, admission_id)
        if not admission or admission.discharged_on is not None:
            raise ValueError(f"Active admission ID {admission_id} not found.")

        now = datetime.now(timezone.utc)
        admission.discharged_on = now
        if discharge_summary:
            admission.discharge_summary = discharge_summary

        # 1. Release bed -> mark DIRTY for housekeeping
        bed = db.session.get(Bed, admission.bed_id) if admission.bed_id else None
        if bed:
            bed.occupied = False
            bed.status = "DIRTY"

        # 2. Recalculate Ward Occupancy
        ward = db.session.get(Ward, admission.ward_id)
        if ward:
            ward.occupied_beds = db.session.query(Bed).join(WardRoom).filter(
                WardRoom.ward_id == ward.id, Bed.occupied.is_(True)
            ).count()

        # 3. History
        history = WardBedHistory(
            ward_id=admission.ward_id,
            patient_id=admission.patient_id,
            action="Discharge",
            timestamp=now,
        )
        db.session.add(history)

        # 4. Close Active IPD Encounter
        encounter = Encounter.query.filter_by(
            patient_id=admission.patient_id, encounter_type="IPD", status="ACTIVE"
        ).first()
        if encounter:
            encounter.status = "COMPLETED"
            encounter.stage = "DISCHARGED"
            encounter.ended_at = now

        # 5. Generate HL7 ADT^A03 Log
        patient = admission.patient
        hl7_msg = cls.format_hl7_adt_payload(
            event_type="A03",
            patient_id=admission.patient_id,
            patient_name=patient.name if patient else "Patient",
            ward_name=ward.name if ward else "Ward",
            bed_number=bed.bed_number if bed else "Bed",
            timestamp=now,
        )

        adt_log = ADTLog(
            event_type="A03",
            patient_id=admission.patient_id,
            admission_id=admission.id,
            from_ward_id=admission.ward_id,
            from_bed_id=admission.bed_id,
            user_id=user_id,
            hl7_message=hl7_msg,
            created_at=now,
        )
        db.session.add(adt_log)

        log_audit_event(
            db.session,
            level="INFO",
            message=f"ADT^A03 Discharge: Patient {admission.patient_id} discharged from Ward {ward.name if ward else admission.ward_id}",
            user_id=user_id,
            source="adt_engine",
        )

        db.session.commit()
        return admission, adt_log

    @classmethod
    def update_bed_status(
        cls,
        bed_id: int,
        new_status: str,
        user_id: int | None = None,
        notes: str | None = None,
    ) -> Bed:
        """
        Manage bed turnaround state transitions.
        Statuses: AVAILABLE, OCCUPIED, DIRTY, CLEANING, MAINTENANCE.
        """
        new_status_upper = new_status.upper()
        if new_status_upper not in VALID_BED_STATUSES:
            raise ValueError(f"Invalid bed status '{new_status}'. Must be one of {VALID_BED_STATUSES}.")

        bed = db.session.get(Bed, bed_id)
        if not bed:
            raise ValueError(f"Bed ID {bed_id} not found.")

        # If marking AVAILABLE, ensure occupied flag is False
        if new_status_upper == "AVAILABLE":
            bed.occupied = False

        old_status = bed.status
        bed.status = new_status_upper

        log_audit_event(
            db.session,
            level="INFO",
            message=f"Bed Turnaround: Bed {bed.bed_number} (ID #{bed.id}) status changed from {old_status} -> {new_status_upper}",
            user_id=user_id,
            source="adt_engine",
        )

        db.session.commit()
        return bed

    @classmethod
    def get_ward_bed_matrix(cls) -> dict[str, Any]:
        """
        Aggregate live ward bed statuses, occupancy percentages, and bed turnaround metrics.
        """
        wards = Ward.query.all()
        ward_matrix = []

        total_beds_all = 0
        occupied_beds_all = 0
        dirty_beds_all = 0
        cleaning_beds_all = 0
        available_beds_all = 0
        maintenance_beds_all = 0

        for ward in wards:
            rooms_data = []
            ward_beds = Bed.query.join(WardRoom).filter(WardRoom.ward_id == ward.id).all()

            w_total = len(ward_beds)
            w_occupied = sum(1 for b in ward_beds if b.status == "OCCUPIED" or b.occupied)
            w_dirty = sum(1 for b in ward_beds if b.status == "DIRTY")
            w_cleaning = sum(1 for b in ward_beds if b.status == "CLEANING")
            w_available = sum(1 for b in ward_beds if b.status == "AVAILABLE" and not b.occupied)
            w_maint = sum(1 for b in ward_beds if b.status == "MAINTENANCE")

            total_beds_all += w_total
            occupied_beds_all += w_occupied
            dirty_beds_all += w_dirty
            cleaning_beds_all += w_cleaning
            available_beds_all += w_available
            maintenance_beds_all += w_maint

            for room in ward.rooms:
                beds_in_room = []
                for b in room.beds:
                    active_adm = AdmittedPatient.query.filter_by(
                        bed_id=b.id, discharged_on=None
                    ).first()
                    patient_info = None
                    if active_adm and active_adm.patient:
                        patient_info = {
                            "patient_id": active_adm.patient_id,
                            "patient_name": active_adm.patient.name,
                            "admitted_on": active_adm.admitted_on.strftime("%Y-%m-%d %H:%M") if active_adm.admitted_on else "",
                            "admission_id": active_adm.id,
                        }

                    beds_in_room.append({
                        "bed_id": b.id,
                        "bed_number": b.bed_number,
                        "status": b.status,
                        "occupied": b.occupied,
                        "patient": patient_info,
                    })

                rooms_data.append({
                    "room_id": room.id,
                    "room_number": room.room_number,
                    "beds": beds_in_room,
                })

            occupancy_pct = round((w_occupied / w_total * 100), 1) if w_total > 0 else 0.0

            ward_matrix.append({
                "ward_id": ward.id,
                "ward_name": ward.name,
                "sex": ward.sex,
                "daily_charge": float(ward.daily_charge) if ward.daily_charge else 0.0,
                "total_beds": w_total,
                "occupied_beds": w_occupied,
                "dirty_beds": w_dirty,
                "cleaning_beds": w_cleaning,
                "available_beds": w_available,
                "maintenance_beds": w_maint,
                "occupancy_pct": occupancy_pct,
                "rooms": rooms_data,
            })

        overall_occupancy = round((occupied_beds_all / total_beds_all * 100), 1) if total_beds_all > 0 else 0.0

        return {
            "total_beds": total_beds_all,
            "occupied_beds": occupied_beds_all,
            "available_beds": available_beds_all,
            "dirty_beds": dirty_beds_all,
            "cleaning_beds": cleaning_beds_all,
            "maintenance_beds": maintenance_beds_all,
            "overall_occupancy_pct": overall_occupancy,
            "wards": ward_matrix,
        }
