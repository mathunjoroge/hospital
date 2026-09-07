from datetime import datetime

from extensions import db


class OutboundNotificationLog(db.Model):
    """Log table tracking outbound notifications sent to patients or staff."""

    __tablename__ = "outbound_notification_logs"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(
        db.String(50),
        db.ForeignKey("patients.patient_id"),
        nullable=True,
        index=True,
    )
    recipient = db.Column(db.String(255), nullable=False, index=True)
    channel = db.Column(
        db.String(20), nullable=False, default="email"
    )  # email, sms, in_app
    event_type = db.Column(
        db.String(50), nullable=False, index=True
    )  # appointment_reminder, lab_result_ready, invoice_due, etc.
    subject = db.Column(db.String(255), nullable=True)
    body = db.Column(db.Text, nullable=False)
    status = db.Column(
        db.String(20), nullable=False, default="PENDING", index=True
    )  # PENDING, SENT, FAILED
    error_message = db.Column(db.Text, nullable=True)
    sent_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    patient = db.relationship("Patient", backref="outbound_notifications")

    def __repr__(self):
        return f"<OutboundNotificationLog id={self.id} event={self.event_type} status={self.status}>"
