import logging
from datetime import datetime

from flask import current_app
from flask_mail import Message

from departments.models.notification_log import OutboundNotificationLog
from extensions import db

logger = logging.getLogger(__name__)

# Standard notification event types
EVENT_APPOINTMENT_REMINDER = 'appointment_reminder'
EVENT_LAB_RESULT_READY = 'lab_result_ready'
EVENT_INVOICE_DUE = 'invoice_due'
EVENT_PAYMENT_RECEIVED = 'payment_received'
EVENT_CLAIM_STATUS_CHANGED = 'claim_status_changed'
EVENT_BREAK_GLASS = 'break_glass_invoked'     # Phase D: emergency override alert


class BaseChannel:
    """Base interface for notification channel drivers."""

    def send(self, log_entry: OutboundNotificationLog) -> bool:
        raise NotImplementedError


class EmailChannel(BaseChannel):
    """Outbound email delivery channel driver powered by Flask-Mail."""

    def send(self, log_entry: OutboundNotificationLog) -> bool:
        mail = current_app.extensions.get('mail') if current_app else None
        if not mail or (current_app and current_app.config.get('TESTING')):
            logger.info(
                f"[EmailChannel Mock] Sent email to {log_entry.recipient} - Subject: {log_entry.subject}"
            )
            return True

        sender = (
            current_app.config.get('MAIL_DEFAULT_SENDER') or 'noreply@hospital.org'
            if current_app
            else 'noreply@hospital.org'
        )
        msg = Message(
            subject=log_entry.subject or "Hospital Notification",
            recipients=[log_entry.recipient],
            body=log_entry.body,
            sender=sender,
        )
        try:
            mail.send(msg)
            logger.info(f"Email sent successfully to {log_entry.recipient}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email to {log_entry.recipient}: {e}")
            raise e


class SandboxSMSChannel(BaseChannel):
    """Sandbox SMS channel driver simulating SMS gateway dispatches."""

    def send(self, log_entry: OutboundNotificationLog) -> bool:
        logger.info(
            f"[SandboxSMSChannel] SMS to {log_entry.recipient}: {log_entry.body}"
        )
        return True


class InAppChannel(BaseChannel):
    """In-app internal notification channel driver."""

    def send(self, log_entry: OutboundNotificationLog) -> bool:

        # If patient or recipient matches a user ID or internal record
        logger.info(
            f"[InAppChannel] In-App Notification created for recipient: {log_entry.recipient}"
        )
        return True


# Channel registry
CHANNELS = {
    'email': EmailChannel(),
    'sms': SandboxSMSChannel(),
    'in_app': InAppChannel(),
}


class NotificationDispatcher:
    """Event-driven notification dispatcher."""

    @staticmethod
    def dispatch_event(
        event_type: str,
        recipient: str,
        subject: str,
        body: str,
        patient_id: str = None,
        channels: list = None,
    ) -> OutboundNotificationLog:
        """
        Dispatch a notification event to the specified recipient over selected channels.
        Defaults to ['email'] channel. Records audit log in OutboundNotificationLog.
        """
        if channels is None:
            channels = ['email']

        log_entries = []
        for ch_name in channels:
            log_entry = OutboundNotificationLog(
                patient_id=patient_id,
                recipient=recipient,
                channel=ch_name,
                event_type=event_type,
                subject=subject,
                body=body,
                status='PENDING',
            )
            db.session.add(log_entry)
            db.session.flush()

            driver = CHANNELS.get(ch_name, CHANNELS['email'])
            try:
                success = driver.send(log_entry)
                if success:
                    log_entry.status = 'SENT'
                    log_entry.sent_at = datetime.utcnow()
                else:
                    log_entry.status = 'FAILED'
                    log_entry.error_message = 'Driver returned failure'
            except Exception as exc:
                log_entry.status = 'FAILED'
                log_entry.error_message = str(exc)

            log_entries.append(log_entry)

        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error committing notification logs: {e}")

        return log_entries[0] if log_entries else None
