import uuid
from datetime import datetime, timezone

from extensions import db


class BackupJob(db.Model):
    """
    Tracks automated and manual database/system backup executions.
    Essential for validating RPO (Recovery Point Objective).
    """

    __tablename__ = "backup_jobs"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # DATABASE, FILESYSTEM, FULL_SYSTEM
    backup_type = db.Column(db.String(30), nullable=False)

    # PENDING, RUNNING, COMPLETED, FAILED
    status = db.Column(db.String(20), nullable=False, default="PENDING")

    # Size in megabytes
    size_mb = db.Column(db.Numeric(10, 2), nullable=True)

    # Duration in seconds
    duration_seconds = db.Column(db.Integer, nullable=True)

    # Storage destination (e.g., S3 bucket, NAS path)
    destination_path = db.Column(db.String(255), nullable=True)

    error_message = db.Column(db.Text, nullable=True)

    started_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    completed_at = db.Column(db.DateTime(timezone=True), nullable=True)


class RestoreTest(db.Model):
    """
    Tracks disaster recovery drills to validate RTO (Recovery Time Objective).
    Ensures backups are actually restorable in an emergency.
    """

    __tablename__ = "restore_tests"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # Reference to the backup job being tested
    backup_job_id = db.Column(
        db.String(36), db.ForeignKey("backup_jobs.id"), nullable=False, index=True
    )

    # PENDING, RUNNING, SUCCESS, FAILED
    status = db.Column(db.String(20), nullable=False, default="PENDING")

    # Time taken to restore (RTO validation)
    restore_duration_minutes = db.Column(db.Integer, nullable=True)

    performed_by = db.Column(db.Integer, nullable=False)
    notes = db.Column(db.Text, nullable=True)

    started_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    completed_at = db.Column(db.DateTime(timezone=True), nullable=True)


class SystemAlert(db.Model):
    """
    Tracks application and infrastructure alerts for observability.
    e.g., High API latency, low disk space, Celery queue backlog.
    """

    __tablename__ = "system_alerts"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # e.g., "DATABASE", "REDIS", "CELERY", "DISK_SPACE", "API_LATENCY"
    alert_source = db.Column(db.String(50), nullable=False)

    # INFO, WARNING, CRITICAL
    severity = db.Column(db.String(20), nullable=False, default="WARNING")

    alert_message = db.Column(db.Text, nullable=False)

    # ACTIVE, ACKNOWLEDGED, RESOLVED
    status = db.Column(db.String(20), nullable=False, default="ACTIVE")

    acknowledged_by = db.Column(db.Integer, nullable=True)
    resolution_notes = db.Column(db.Text, nullable=True)

    triggered_at = db.Column(
        db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    resolved_at = db.Column(db.DateTime(timezone=True), nullable=True)
