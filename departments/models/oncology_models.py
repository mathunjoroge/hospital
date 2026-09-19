"""
departments/models/oncology_models.py
──────────────────────────────────────
Data models for Oncology & Chemotherapy Regimen Engine.
Stores chemotherapy orders, BSA calculations, protocol definitions, and cumulative lifetime toxicity tracking.
"""

from datetime import datetime

from extensions import db

# Order lifecycle. ADMINISTERED and CANCELLED are terminal.
CHEMO_STATUSES = ("ORDERED", "PREPARED", "ADMINISTERED", "CANCELLED")
CHEMO_ALLOWED_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "ORDERED": ("PREPARED", "CANCELLED"),
    "PREPARED": ("ADMINISTERED", "CANCELLED"),
    "ADMINISTERED": (),
    "CANCELLED": (),
}


class ChemotherapyRegimenOrder(db.Model):
    """
    Chemotherapy regimen prescription order for oncology patients.
    Tracks body surface area (BSA), protocol doses, and cumulative lifetime toxicity caps.

    Clinical history is never deleted: an erroneous order is moved to CANCELLED
    (with a reason) and is then excluded from cumulative lifetime-dose totals.
    """

    __tablename__ = "chemotherapy_regimen_orders"
    __table_args__ = (
        # At most one *active* (non-cancelled) order per patient/protocol/cycle.
        # Stops a double-click or retry from double-counting cumulative doses.
        db.Index(
            "uq_chemo_active_patient_protocol_cycle",
            "patient_id",
            "protocol_name",
            "cycle_number",
            unique=True,
            postgresql_where=db.text("status <> 'CANCELLED'"),
            sqlite_where=db.text("status <> 'CANCELLED'"),
        ),
    )

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(
        db.String(20),
        db.ForeignKey("patients.patient_id"),
        nullable=False,
        index=True,
    )
    physician_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    protocol_name = db.Column(
        db.String(50), nullable=False
    )  # e.g., FOLFOX6, AC-T, ABVD
    cancer_type = db.Column(
        db.String(100), nullable=True
    )  # e.g., Breast Cancer, Colorectal Cancer

    # Patient Biometrics & BSA
    weight_kg = db.Column(db.Float, nullable=False)
    height_cm = db.Column(db.Float, nullable=False)
    bsa_m2 = db.Column(db.Float, nullable=False)
    bsa_formula = db.Column(
        db.String(20), nullable=False, default="mosteller"
    )  # mosteller or dubois

    # Protocol Scheduling
    cycle_number = db.Column(db.Integer, nullable=False, default=1)
    total_cycles = db.Column(db.Integer, nullable=False, default=6)

    # Calculated Doses (JSON string of list of dicts)
    # [{"drug_name": "Doxorubicin", "dose_per_m2": 60.0, "unit": "mg/m2", "calculated_dose": 108.0, "cumulative_dose_mg": 216.0, "cap_exceeded": False}]
    calculated_doses_json = db.Column(db.Text, nullable=False)

    # Status & Warnings
    status = db.Column(
        db.String(20), nullable=False, default="ORDERED"
    )  # ORDERED, PREPARED, ADMINISTERED, CANCELLED
    has_toxicity_warning = db.Column(db.Boolean, nullable=False, default=False)
    toxicity_warning_details = db.Column(db.Text, nullable=True)
    # Prescriber's documented justification when a lifetime-cap warning was overridden.
    toxicity_override_reason = db.Column(db.Text, nullable=True)

    # Last status transition (who / when / why)
    status_reason = db.Column(db.Text, nullable=True)
    status_changed_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    status_changed_at = db.Column(db.DateTime, nullable=True)

    created_at = db.Column(
        db.DateTime, nullable=False, default=datetime.utcnow, index=True
    )

    physician = db.relationship(
        "User", foreign_keys=[physician_id], backref="chemotherapy_orders"
    )
