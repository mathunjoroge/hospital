"""
departments/models/renal.py
────────────────────────────
Data models for Renal / Dialysis Unit.
Tracks hemodialysis (HD) & CRRT session logs, vascular access records,
and renal unit configuration settings.
"""

from datetime import datetime

from extensions import db


class DialysisSession(db.Model):
    """
    Hemodialysis (HD) and Continuous Renal Replacement Therapy (CRRT) session log.
    """

    __tablename__ = "dialysis_sessions"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(20), nullable=False, index=True)
    nurse_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    modality = db.Column(db.String(20), nullable=False)  # "HD" or "CRRT"
    session_date = db.Column(db.Date, nullable=False)
    start_time = db.Column(db.DateTime, nullable=True)
    end_time = db.Column(db.DateTime, nullable=True)

    # Raw session parameters (NOT Kt/V calculation)
    blood_flow_rate = db.Column(db.Float, nullable=True)  # mL/min
    dialysate_flow_rate = db.Column(db.Float, nullable=True)  # mL/min
    ultrafiltration_volume = db.Column(db.Float, nullable=True)  # mL or L
    pre_weight = db.Column(db.Float, nullable=True)  # kg
    post_weight = db.Column(db.Float, nullable=True)  # kg

    status = db.Column(db.String(50), nullable=False, default="SCHEDULED")  # SCHEDULED, IN_PROGRESS, COMPLETED, TERMINATED_EARLY
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)

    nurse = db.relationship("User", backref="dialysis_sessions")


class VascularAccessRecord(db.Model):
    """
    Vascular access record and complication surveillance log.
    """

    __tablename__ = "vascular_access_records"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(20), nullable=False, index=True)

    access_type = db.Column(db.String(50), nullable=False)  # "AVF", "AVG", "Tunnelled Catheter", "Temporary Catheter"
    insertion_date = db.Column(db.Date, nullable=True)
    site_description = db.Column(db.String(100), nullable=True)  # e.g., "Left forearm"
    complication_notes = db.Column(db.Text, nullable=True)
    dialysis_session_id = db.Column(db.Integer, db.ForeignKey("dialysis_sessions.id"), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)

    dialysis_session = db.relationship("DialysisSession", backref="vascular_access_records")


class RenalUnitConfig(db.Model):
    """
    Renal unit scheduling and facility-level configuration.
    """

    __tablename__ = "renal_unit_configs"

    id = db.Column(db.Integer, primary_key=True)
    facility_id = db.Column(db.Integer, db.ForeignKey("facilities.id"), nullable=True, index=True)
    scheduling_mode = db.Column(db.String(20), nullable=False, default="MANUAL")  # MANUAL or SLOT_BASED
    chair_count = db.Column(db.Integer, nullable=True)
    shift_pattern = db.Column(db.String(255), nullable=True)  # Free text e.g., "Mon/Wed/Fri"
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
