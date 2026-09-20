"""
departments/models/theatre.py
──────────────────────────────
Database models for Johns Hopkins–Grade Theatre & Surgical Module:
1. WhoSurgicalChecklist — 3-stage WHO Surgical Safety Checklist (Sign In, Time Out, Sign Out)
2. AnaestheticRecord   — Pre-op ASA status, airway grade, intra-op agents, vitals time-series & fluid balance
3. PostOpNote          — Operative note, surgical team, findings, pathology specimens, PACU Aldrete score
4. SurgicalInstrumentCount — Instrument/sponge/needle reconciliation before cavity & skin closure
"""

import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

from extensions import db


class WhoSurgicalChecklist(db.Model):
    """
    WHO Surgical Safety Checklist (3-stage safety verification).
    - Sign In: Before induction of anaesthesia
    - Time Out: Before skin incision
    - Sign Out: Before patient leaves operating room
    """

    __tablename__ = "who_surgical_checklists"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    theatre_entry_id = db.Column(
        db.Integer, db.ForeignKey("theatre_list.id"), nullable=False, index=True
    )
    patient_id = db.Column(
        db.String(20), db.ForeignKey("patients.patient_id"), nullable=False, index=True
    )
    encounter_id = db.Column(
        db.Integer, db.ForeignKey("encounters.id"), nullable=True, index=True
    )

    # --- Stage 1: Sign In (Pre-induction) ---
    sign_in_completed = db.Column(db.Boolean, default=False, nullable=False)
    patient_identity_confirmed = db.Column(db.Boolean, default=False)
    site_marked = db.Column(db.Boolean, default=False)
    anaesthesia_safety_check_completed = db.Column(db.Boolean, default=False)
    pulse_oximeter_functioning = db.Column(db.Boolean, default=False)
    known_allergy = db.Column(db.Boolean, default=False)
    allergy_details = db.Column(db.String(255), nullable=True)
    difficult_airway_risk = db.Column(db.Boolean, default=False)
    airway_equipment_available = db.Column(db.Boolean, default=False)
    blood_loss_risk_over_500ml = db.Column(db.Boolean, default=False)
    iv_access_and_fluids_planned = db.Column(db.Boolean, default=False)
    sign_in_completed_by = db.Column(
        db.Integer, db.ForeignKey("users.id"), nullable=True
    )
    sign_in_completed_at = db.Column(db.DateTime, nullable=True)

    # --- Stage 2: Time Out (Pre-incision) ---
    time_out_completed = db.Column(db.Boolean, default=False, nullable=False)
    team_members_introduced = db.Column(db.Boolean, default=False)
    verbal_confirm_patient_site_procedure = db.Column(db.Boolean, default=False)
    critical_steps_surgeon_reviewed = db.Column(db.Boolean, default=False)
    critical_steps_anaesthetist_reviewed = db.Column(db.Boolean, default=False)
    critical_steps_nursing_reviewed = db.Column(db.Boolean, default=False)
    antibiotic_prophylaxis_given = db.Column(db.Boolean, default=False)
    antibiotic_given_within_60min = db.Column(db.Boolean, default=False)
    essential_imaging_displayed = db.Column(db.Boolean, default=False)
    time_out_completed_by = db.Column(
        db.Integer, db.ForeignKey("users.id"), nullable=True
    )
    time_out_completed_at = db.Column(db.DateTime, nullable=True)

    # --- Stage 3: Sign Out (Pre-transfer out of OR) ---
    sign_out_completed = db.Column(db.Boolean, default=False, nullable=False)
    procedure_name_recorded = db.Column(db.String(255), nullable=True)
    instrument_sponge_needle_counts_correct = db.Column(db.Boolean, default=False)
    specimens_labelled = db.Column(db.Boolean, default=False)
    equipment_problems_addressed = db.Column(db.Boolean, default=False)
    key_recovery_concerns_reviewed = db.Column(db.Text, nullable=True)
    sign_out_completed_by = db.Column(
        db.Integer, db.ForeignKey("users.id"), nullable=True
    )
    sign_out_completed_at = db.Column(db.DateTime, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    theatre_entry = db.relationship(
        "TheatreList", backref=db.backref("who_checklist", uselist=False)
    )
    patient = db.relationship("Patient", backref="who_checklists")
    encounter = db.relationship("Encounter", foreign_keys=[encounter_id])

    def is_fully_completed(self) -> bool:
        return (
            self.sign_in_completed
            and self.time_out_completed
            and self.sign_out_completed
        )

    def __repr__(self):
        return f"<WhoSurgicalChecklist entry={self.theatre_entry_id} in={self.sign_in_completed} out={self.time_out_completed} final={self.sign_out_completed}>"


class AnaestheticRecord(db.Model):
    """
    Anaesthetic Intraoperative Record.
    Tracks ASA classification, airway grade, anaesthetic techniques, agents,
    vital sign time series (JSON), fluid/blood product balance, and recovery status.
    """

    __tablename__ = "anaesthetic_records"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    theatre_entry_id = db.Column(
        db.Integer, db.ForeignKey("theatre_list.id"), nullable=False, index=True
    )
    patient_id = db.Column(
        db.String(20), db.ForeignKey("patients.patient_id"), nullable=False, index=True
    )
    encounter_id = db.Column(
        db.Integer, db.ForeignKey("encounters.id"), nullable=True, index=True
    )
    anaesthetist_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    # ASA Classification: ASA I, ASA II, ASA III, ASA IV, ASA V, ASA VI, + Emergency E
    asa_status = db.Column(db.String(10), nullable=False, default="ASA I")
    is_emergency = db.Column(db.Boolean, default=False, nullable=False)

    # Airway Assessment: Mallampati I, II, III, IV
    mallampati_class = db.Column(db.String(10), nullable=True)
    airway_management = db.Column(
        db.String(100), nullable=True
    )  # ETT, LMA, Mask, Tracheostomy
    ett_size = db.Column(db.String(20), nullable=True)  # e.g., "7.5 cuffed"

    # Technique: General, Spinal, Epidural, Regional, Local, Sedation
    technique = db.Column(db.String(100), nullable=False, default="General")

    # JSON stored fields
    agents_administered_json = db.Column(
        db.Text, nullable=True, default="[]"
    )  # list of dicts
    vitals_series_json = db.Column(
        db.Text, nullable=True, default="[]"
    )  # list of dicts (time, hr, bp_sys, bp_dia, spo2, etco2)
    timeline_events_json = db.Column(
        db.Text, nullable=True, default="[]"
    )  # list of dicts (timestamp, event_type, notes, recorded_by)

    # Fluid & Blood balance
    estimated_blood_loss_ml = db.Column(db.Integer, default=0, nullable=False)
    crystalloids_ml = db.Column(db.Integer, default=0, nullable=False)
    colloids_ml = db.Column(db.Integer, default=0, nullable=False)
    blood_products_ml = db.Column(db.Integer, default=0, nullable=False)
    urine_output_ml = db.Column(db.Integer, default=0, nullable=False)

    complications_notes = db.Column(db.Text, nullable=True)
    anaesthesia_start_time = db.Column(db.DateTime, nullable=True)
    anaesthesia_end_time = db.Column(db.DateTime, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    theatre_entry = db.relationship(
        "TheatreList", backref=db.backref("anaesthetic_record", uselist=False)
    )
    patient = db.relationship("Patient", backref="anaesthetic_records")
    anaesthetist = db.relationship("User", foreign_keys=[anaesthetist_id])

    @property
    def agents_administered(self) -> list:
        try:
            return json.loads(self.agents_administered_json or "[]")
        except Exception:
            logger.exception("Failed to decode agents_administered_json for AnaestheticRecord id=%s", self.id)
            return []

    @agents_administered.setter
    def agents_administered(self, value: list):
        self.agents_administered_json = json.dumps(value or [])

    @property
    def vitals_series(self) -> list:
        try:
            return json.loads(self.vitals_series_json or "[]")
        except Exception:
            logger.exception("Failed to decode vitals_series_json for AnaestheticRecord id=%s", self.id)
            return []

    @vitals_series.setter
    def vitals_series(self, value: list):
        self.vitals_series_json = json.dumps(value or [])

    @property
    def timeline_events(self) -> list:
        try:
            return json.loads(self.timeline_events_json or "[]")
        except Exception:
            logger.exception("Failed to decode timeline_events_json for AnaestheticRecord id=%s", self.id)
            return []

    @timeline_events.setter
    def timeline_events(self, value: list):
        self.timeline_events_json = json.dumps(value or [])

    def __repr__(self):
        return f"<AnaestheticRecord entry={self.theatre_entry_id} asa={self.asa_status} tech={self.technique}>"


class PostOpNote(db.Model):
    """
    Surgical Operative & Post-Op Note.
    Captures surgical team, pre & post-operative diagnoses, procedure description,
    specimens collected, implants, drains, PACU Aldrete recovery score (0-10), and orders.
    """

    __tablename__ = "post_op_notes"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    theatre_entry_id = db.Column(
        db.Integer, db.ForeignKey("theatre_list.id"), nullable=False, index=True
    )
    patient_id = db.Column(
        db.String(20), db.ForeignKey("patients.patient_id"), nullable=False, index=True
    )
    encounter_id = db.Column(
        db.Integer, db.ForeignKey("encounters.id"), nullable=True, index=True
    )
    surgeon_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    assistant_surgeon = db.Column(db.String(100), nullable=True)
    scrub_nurse = db.Column(db.String(100), nullable=True)
    circulating_nurse = db.Column(db.String(100), nullable=True)

    preop_diagnosis = db.Column(db.String(255), nullable=False)
    postop_diagnosis = db.Column(db.String(255), nullable=False)
    procedure_performed = db.Column(db.Text, nullable=False)
    surgical_findings = db.Column(db.Text, nullable=False)
    specimens_taken = db.Column(db.Text, nullable=True)
    implants_inserted = db.Column(db.Text, nullable=True)  # Lot / Serial numbers
    drains_tubes_placed = db.Column(db.Text, nullable=True)

    # PACU Aldrete Recovery Score (0-10)
    # Motor (0-2), Respiration (0-2), Circulation (0-2), Consciousness (0-2), O2 Sat (0-2)
    aldrete_activity = db.Column(db.Integer, default=2, nullable=False)
    aldrete_respiration = db.Column(db.Integer, default=2, nullable=False)
    aldrete_circulation = db.Column(db.Integer, default=2, nullable=False)
    aldrete_consciousness = db.Column(db.Integer, default=2, nullable=False)
    aldrete_spo2 = db.Column(db.Integer, default=2, nullable=False)

    postop_instructions = db.Column(db.Text, nullable=True)
    discharge_to = db.Column(
        db.String(100), default="PACU", nullable=False
    )  # PACU, ICU, Ward, Home

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    theatre_entry = db.relationship(
        "TheatreList", backref=db.backref("post_op_note", uselist=False)
    )
    patient = db.relationship("Patient", backref="post_op_notes")
    surgeon = db.relationship("User", foreign_keys=[surgeon_id])

    @property
    def total_aldrete_score(self) -> int:
        return (
            (self.aldrete_activity or 0)
            + (self.aldrete_respiration or 0)
            + (self.aldrete_circulation or 0)
            + (self.aldrete_consciousness or 0)
            + (self.aldrete_spo2 or 0)
        )

    def is_fit_for_pacu_discharge(self) -> bool:
        """Aldrete score >= 9 is standard clinical threshold for PACU discharge."""
        return self.total_aldrete_score >= 9

    def __repr__(self):
        return f"<PostOpNote entry={self.theatre_entry_id} diagnosis={self.postop_diagnosis} aldrete={self.total_aldrete_score}>"


class SurgicalInstrumentCount(db.Model):
    """
    Surgical Instrument, Sponge, and Needle Reconciliation Record.
    Blocks OR exit and flags discrepancy if counts do not reconcile.
    """

    __tablename__ = "surgical_instrument_counts"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    theatre_entry_id = db.Column(
        db.Integer, db.ForeignKey("theatre_list.id"), nullable=False, index=True
    )
    encounter_id = db.Column(
        db.Integer, db.ForeignKey("encounters.id"), nullable=True, index=True
    )
    scrub_nurse_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    circulating_nurse_id = db.Column(
        db.Integer, db.ForeignKey("users.id"), nullable=True
    )

    tray_name_or_barcode = db.Column(
        db.String(100), nullable=False, default="Standard Surgical Tray"
    )

    # Counts
    sponges_initial = db.Column(db.Integer, default=0, nullable=False)
    sponges_added = db.Column(db.Integer, default=0, nullable=False)
    sponges_closing_cavity = db.Column(db.Integer, default=0, nullable=False)
    sponges_closing_skin = db.Column(db.Integer, default=0, nullable=False)

    needles_initial = db.Column(db.Integer, default=0, nullable=False)
    needles_added = db.Column(db.Integer, default=0, nullable=False)
    needles_closing_cavity = db.Column(db.Integer, default=0, nullable=False)
    needles_closing_skin = db.Column(db.Integer, default=0, nullable=False)

    instruments_initial = db.Column(db.Integer, default=0, nullable=False)
    instruments_added = db.Column(db.Integer, default=0, nullable=False)
    instruments_closing_cavity = db.Column(db.Integer, default=0, nullable=False)
    instruments_closing_skin = db.Column(db.Integer, default=0, nullable=False)

    count_reconciled = db.Column(db.Boolean, default=False, nullable=False)
    xray_ordered_for_discrepancy = db.Column(db.Boolean, default=False, nullable=False)
    notes = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    theatre_entry = db.relationship(
        "TheatreList", backref=db.backref("instrument_count", uselist=False)
    )
    scrub_nurse = db.relationship("User", foreign_keys=[scrub_nurse_id])
    circulating_nurse = db.relationship("User", foreign_keys=[circulating_nurse_id])

    def calculate_reconciliation(self) -> dict:
        """Reconcile counts at skin closure — and at cavity closure when one was
        performed.

        A cavity-closure count is considered performed when ANY cavity figure is
        non-zero. If it was performed, it must reconcile too: a count that only
        matches at skin while the cavity figure is short means an item may still
        be inside the patient. When no cavity count exists (legacy rows / "no
        cavity" procedures), the cavity check is skipped.
        """
        total_sponges_in = (self.sponges_initial or 0) + (self.sponges_added or 0)
        total_needles_in = (self.needles_initial or 0) + (self.needles_added or 0)
        total_inst_in = (self.instruments_initial or 0) + (self.instruments_added or 0)

        sponges_skin = self.sponges_closing_skin or 0
        needles_skin = self.needles_closing_skin or 0
        inst_skin = self.instruments_closing_skin or 0

        sponge_ok = sponges_skin == total_sponges_in
        needle_ok = needles_skin == total_needles_in
        inst_ok = inst_skin == total_inst_in

        cavity_performed = any(
            (
                (self.sponges_closing_cavity or 0) != 0,
                (self.needles_closing_cavity or 0) != 0,
                (self.instruments_closing_cavity or 0) != 0,
            )
        )
        if cavity_performed:
            sponge_cavity_ok = (self.sponges_closing_cavity or 0) == total_sponges_in
            needle_cavity_ok = (self.needles_closing_cavity or 0) == total_needles_in
            inst_cavity_ok = (self.instruments_closing_cavity or 0) == total_inst_in
        else:
            sponge_cavity_ok = needle_cavity_ok = inst_cavity_ok = True

        reconciled = (
            sponge_ok
            and needle_ok
            and inst_ok
            and sponge_cavity_ok
            and needle_cavity_ok
            and inst_cavity_ok
        )
        self.count_reconciled = reconciled
        return {
            "reconciled": reconciled,
            "sponges_ok": sponge_ok,
            "needles_ok": needle_ok,
            "instruments_ok": inst_ok,
            "cavity_performed": cavity_performed,
            "sponges_cavity_ok": sponge_cavity_ok,
            "needles_cavity_ok": needle_cavity_ok,
            "instruments_cavity_ok": inst_cavity_ok,
            "sponge_diff": sponges_skin - total_sponges_in,
            "needle_diff": needles_skin - total_needles_in,
            "inst_diff": inst_skin - total_inst_in,
        }

    def __repr__(self):
        return f"<SurgicalInstrumentCount entry={self.theatre_entry_id} reconciled={self.count_reconciled}>"
