"""
departments/models/icu.py
─────────────────────────
Data models for ICU / HDU Flowsheet Workstation.
Tracks vital sign trends, mechanical ventilator parameters, GCS scores, and 24h I/O fluid balance.
"""

from datetime import datetime

from extensions import db


class ICUFlowsheetEntry(db.Model):
    """
    Hemodynamic, respiratory, ventilator, and neurological observations for ICU/HDU patients.
    """

    __tablename__ = "icu_flowsheet_entries"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(20), nullable=False, index=True)
    nurse_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    # Hemodynamics & Vitals
    heart_rate = db.Column(db.Integer, nullable=True)  # bpm
    bp_systolic = db.Column(db.Integer, nullable=True)  # mmHg
    bp_diastolic = db.Column(db.Integer, nullable=True)  # mmHg
    mean_arterial_pressure = db.Column(
        db.Float, nullable=True
    )  # mmHg (MAP = DP + 1/3(SP-DP))
    spo2 = db.Column(db.Integer, nullable=True)  # %
    temperature = db.Column(db.Float, nullable=True)  # Celsius
    central_venous_pressure = db.Column(db.Float, nullable=True)  # cmH2O (CVP)

    # Mechanical Ventilator Settings
    ventilator_mode = db.Column(
        db.String(50), nullable=True
    )  # e.g., AC/VC, SIMV, PSV, CPAP, BiPAP
    fio2 = db.Column(db.Float, nullable=True)  # % (21 - 100)
    peep = db.Column(db.Float, nullable=True)  # cmH2O
    tidal_volume = db.Column(db.Integer, nullable=True)  # mL
    peak_inspiratory_pressure = db.Column(db.Float, nullable=True)  # cmH2O (PIP)
    respiratory_rate = db.Column(db.Integer, nullable=True)  # breaths/min

    # Neurological (GCS & RASS) & Pain
    gcs_eye = db.Column(db.Integer, nullable=True)  # 1-4
    gcs_verbal = db.Column(db.Integer, nullable=True)  # 1-5
    gcs_motor = db.Column(db.Integer, nullable=True)  # 1-6
    gcs_total = db.Column(db.Integer, nullable=True)  # 3-15
    rass_score = db.Column(
        db.Integer, nullable=True
    )  # Richmond Agitation-Sedation Scale (-5 to +4)
    pain_score = db.Column(db.Integer, nullable=True)  # 0-10

    notes = db.Column(db.Text, nullable=True)
    timestamp = db.Column(
        db.DateTime, nullable=False, default=datetime.utcnow, index=True
    )

    nurse = db.relationship("User", backref="icu_flowsheet_entries")


class ICUFluidBalance(db.Model):
    """
    Input/Output (I/O) fluid balance tracking for ICU/HDU patients.
    """

    __tablename__ = "icu_fluid_balances"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(20), nullable=False, index=True)
    nurse_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    # Inputs (mL)
    iv_fluids_ml = db.Column(db.Float, nullable=False, default=0.0)
    blood_products_ml = db.Column(db.Float, nullable=False, default=0.0)
    enteral_oral_ml = db.Column(db.Float, nullable=False, default=0.0)
    iv_medications_ml = db.Column(db.Float, nullable=False, default=0.0)
    total_input_ml = db.Column(db.Float, nullable=False, default=0.0)

    # Outputs (mL)
    urine_output_ml = db.Column(db.Float, nullable=False, default=0.0)
    drain_output_ml = db.Column(db.Float, nullable=False, default=0.0)
    ng_emesis_ml = db.Column(db.Float, nullable=False, default=0.0)
    stool_ml = db.Column(db.Float, nullable=False, default=0.0)
    total_output_ml = db.Column(db.Float, nullable=False, default=0.0)

    # Net Balance (Inputs - Outputs)
    net_balance_ml = db.Column(db.Float, nullable=False, default=0.0)
    patient_weight_kg = db.Column(
        db.Float, nullable=True
    )  # Used for urine output rate (mL/kg/hr)

    notes = db.Column(db.String(255), nullable=True)
    timestamp = db.Column(
        db.DateTime, nullable=False, default=datetime.utcnow, index=True
    )

    nurse = db.relationship("User", backref="icu_fluid_balances")
