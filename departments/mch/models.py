import uuid
from datetime import datetime, timezone

from extensions import db


class AncVisit(db.Model):
    """Antenatal Care visit record — aligns with Kenya MoH 711 ANC register."""

    __tablename__ = "anc_visits"
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.String(20), index=True, nullable=False)
    visit_number = db.Column(db.Integer, nullable=False)
    gestation_weeks = db.Column(db.Integer, nullable=False)
    high_risk_factors = db.Column(db.Text, nullable=True)
    next_appointment_date = db.Column(db.Date, nullable=True)
    visit_date = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    # Clinical measurements
    blood_pressure_systolic = db.Column(db.Integer, nullable=True)   # mmHg
    blood_pressure_diastolic = db.Column(db.Integer, nullable=True)  # mmHg
    weight_kg = db.Column(db.Float, nullable=True)
    fundal_height_cm = db.Column(db.Float, nullable=True)
    foetal_heart_rate = db.Column(db.Integer, nullable=True)         # bpm
    haemoglobin_g_dl = db.Column(db.Float, nullable=True)
    urine_protein = db.Column(db.String(20), nullable=True)          # e.g. NEGATIVE / +1 / +2
    hiv_status = db.Column(db.String(20), nullable=True)             # NEGATIVE / POSITIVE / UNKNOWN
    # FK to the Encounter opened when this ANC visit starts
    encounter_id = db.Column(
        db.Integer, db.ForeignKey("encounters.id"), nullable=True, index=True
    )


class ImmunizationRecord(db.Model):
    """Individual vaccine dose administration — aligns with Kenya MoH 710 Child Health register."""

    __tablename__ = "immunization_records"
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    child_patient_id = db.Column(db.String(20), index=True, nullable=False)
    vaccine_name = db.Column(db.String(100), nullable=False)
    dose_number = db.Column(db.Integer, nullable=False)
    batch_number = db.Column(db.String(50), nullable=True)
    # Extended administration metadata
    site_of_injection = db.Column(db.String(50), nullable=True)      # e.g. LEFT_THIGH
    administered_by = db.Column(db.String(100), nullable=True)       # nurse/clinician name
    adverse_event_noted = db.Column(db.Text, nullable=True)          # free-text AEFI note
    # Link to cold-chain batch for traceability
    vaccine_batch_id = db.Column(
        db.String(36), db.ForeignKey("vaccine_batches.id"), nullable=True, index=True
    )
    administered_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    # FK to the ANC encounter under which this immunization was given
    encounter_id = db.Column(
        db.Integer, db.ForeignKey("encounters.id"), nullable=True, index=True
    )


# ── Cold-Chain Models ─────────────────────────────────────────────────────────


class VaccineBatch(db.Model):
    """
    Cold-chain vaccine batch — tracks stock levels, VVM stage, and breach flags.

    VVM (Vaccine Vial Monitor) stages:
      1 = Unused / OK
      2 = Approaching discard point (still usable)
      3 = Discard point reached (do not use)
      4 = Beyond discard point (quarantine)
    """

    __tablename__ = "vaccine_batches"
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    vaccine_name = db.Column(db.String(100), nullable=False, index=True)
    batch_number = db.Column(db.String(80), nullable=False)
    manufacturer = db.Column(db.String(150), nullable=True)
    supplied_by = db.Column(db.String(150), nullable=True)           # UNICEF / KEMSA / supplier
    quantity_vials = db.Column(db.Integer, nullable=False)           # original received qty
    doses_per_vial = db.Column(db.Integer, nullable=False, default=1)
    quantity_remaining_vials = db.Column(db.Integer, nullable=False) # current stock
    expiry_date = db.Column(db.Date, nullable=False, index=True)
    storage_location = db.Column(db.String(100), nullable=False, index=True)  # e.g. FRIDGE_A
    received_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    # Cold-chain integrity
    vvm_stage = db.Column(db.Integer, nullable=False, default=1)
    is_cold_chain_breach = db.Column(db.Boolean, nullable=False, default=False)

    __table_args__ = (
        db.UniqueConstraint("vaccine_name", "batch_number", name="uq_vaccine_batch"),
    )


class VaccineTemperatureLog(db.Model):
    """
    Temperature telemetry for cold-chain storage locations.

    Readings may come from manual checks or automated IoT sensors.
    A breach is recorded whenever temperature leaves the [2 °C, 8 °C] safe band.
    """

    __tablename__ = "vaccine_temperature_logs"
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    storage_location = db.Column(db.String(100), nullable=False, index=True)
    temperature_celsius = db.Column(db.Float, nullable=False)
    is_breach = db.Column(db.Boolean, nullable=False, default=False)
    breach_type = db.Column(db.String(20), nullable=True)  # FREEZE | TOO_COLD | TOO_HOT
    sensor_id = db.Column(db.String(80), nullable=True)
    logged_by = db.Column(db.String(100), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    recorded_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
