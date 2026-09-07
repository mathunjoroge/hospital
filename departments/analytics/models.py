import uuid

from extensions import db


class DailyKpiSnapshot(db.Model):
    """
    Stores aggregated daily operational metrics for executive dashboards.
    Essential for hospital management and MoH reporting.
    """

    __tablename__ = "daily_kpi_snapshots"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    snapshot_date = db.Column(db.Date, nullable=False, unique=True, index=True)

    # Patient Flow
    total_outpatient_visits = db.Column(db.Integer, default=0)
    total_admissions = db.Column(db.Integer, default=0)
    total_discharges = db.Column(db.Integer, default=0)
    total_emergency_cases = db.Column(db.Integer, default=0)

    # Clinical Metrics
    total_anc_visits = db.Column(db.Integer, default=0)
    total_immunizations = db.Column(db.Integer, default=0)
    total_lab_tests_ordered = db.Column(db.Integer, default=0)

    # Financial Metrics
    total_revenue_collected = db.Column(db.Numeric(12, 2), default=0.0)

    created_at = db.Column(db.DateTime, server_default=db.func.current_timestamp())
