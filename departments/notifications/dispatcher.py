import logging
import os
from datetime import datetime

import requests
from flask import current_app
from flask_mail import Message

from departments.models.notification_log import OutboundNotificationLog
from extensions import db

logger = logging.getLogger(__name__)

# Standard notification event types
EVENT_APPOINTMENT_REMINDER = "appointment_reminder"
EVENT_LAB_RESULT_READY = "lab_result_ready"
EVENT_INVOICE_DUE = "invoice_due"
EVENT_PAYMENT_RECEIVED = "payment_received"
EVENT_CLAIM_STATUS_CHANGED = "claim_status_changed"
EVENT_BREAK_GLASS = "break_glass_invoked"  # Phase D: emergency override alert
EVENT_CREDENTIAL_EXPIRING = "staff_credential_expiring"
EVENT_CREDENTIAL_EXPIRED = "staff_credential_expired"
EVENT_PASSWORD_RESET = "password_reset_request"


class BaseChannel:
    """Base interface for notification channel drivers."""

    def send(self, log_entry: OutboundNotificationLog) -> bool:
        raise NotImplementedError


class EmailChannel(BaseChannel):
    """Outbound email delivery channel driver powered by Flask-Mail."""

    def send(self, log_entry: OutboundNotificationLog) -> bool:
        mail = current_app.extensions.get("mail") if current_app else None
        if not mail or (current_app and current_app.config.get("TESTING")):
            logger.info(
                f"[EmailChannel Mock] Sent email to {log_entry.recipient} - Subject: {log_entry.subject}"
            )
            return True

        sender = (
            current_app.config.get("MAIL_DEFAULT_SENDER") or "noreply@hospital.org"
            if current_app
            else "noreply@hospital.org"
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


class AfricasTalkingSMSChannel(BaseChannel):
    """
    Production/Sandbox SMS channel driver powered by Africa's Talking REST API.
    Reads credentials dynamically from AT_API_KEY, AT_USERNAME, AT_SENDER_ID.
    """

    def send(self, log_entry: OutboundNotificationLog) -> bool:
        username = (
            current_app.config.get("AT_USERNAME")
            or os.environ.get("AT_USERNAME")
            or "sandbox"
            if current_app
            else os.environ.get("AT_USERNAME", "sandbox")
        )
        api_key = (
            current_app.config.get("AT_API_KEY") or os.environ.get("AT_API_KEY")
            if current_app
            else os.environ.get("AT_API_KEY")
        )
        sender_id = (
            current_app.config.get("AT_SENDER_ID") or os.environ.get("AT_SENDER_ID")
            if current_app
            else os.environ.get("AT_SENDER_ID")
        )

        if not api_key:
            err_msg = "Africa's Talking API key (AT_API_KEY) is not configured."
            logger.error(err_msg)
            raise ValueError(err_msg)

        if username.lower() == "sandbox":
            url = "https://api.sandbox.africastalking.com/version1/messaging"
        else:
            url = "https://api.africastalking.com/version1/messaging"

        headers = {
            "ApiKey": api_key,
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        payload = {
            "username": username,
            "to": log_entry.recipient,
            "message": log_entry.body,
        }
        if sender_id:
            payload["from"] = sender_id

        try:
            response = requests.post(url, data=payload, headers=headers, timeout=10)
            if response.status_code not in (200, 201):
                err_msg = f"Africa's Talking API returned HTTP {response.status_code}: {response.text[:200]}"
                logger.error(err_msg)
                raise RuntimeError(err_msg)

            data = response.json()
            sms_data = data.get("SMSMessageData", {})
            recipients = sms_data.get("Recipients", [])

            if recipients:
                rec_status = recipients[0].get("status", "")
                rec_msg_id = recipients[0].get("messageId", "")
                status_code = recipients[0].get("statusCode")
                if rec_status.lower() in (
                    "success",
                    "sent",
                    "queued",
                ) or status_code in (100, 101, 102):
                    logger.info(
                        f"[AfricasTalkingSMSChannel] SMS sent to {log_entry.recipient} (msg_id: {rec_msg_id})"
                    )
                    return True
                else:
                    err_msg = f"Africa's Talking error status: {rec_status} (code: {status_code})"
                    logger.warning(err_msg)
                    raise RuntimeError(err_msg)

            logger.info(
                f"[AfricasTalkingSMSChannel] Response summary: {sms_data.get('Message')}"
            )
            return True
        except Exception as e:
            logger.error(
                f"[AfricasTalkingSMSChannel] Delivery error to {log_entry.recipient}: {e}"
            )
            raise e


class InAppChannel(BaseChannel):
    """In-app internal notification channel driver."""

    def send(self, log_entry: OutboundNotificationLog) -> bool:
        # If patient or recipient matches a user ID or internal record
        logger.info(
            f"[InAppChannel] In-App Notification created for recipient: {log_entry.recipient}"
        )
        return True


def get_sms_channel() -> BaseChannel:
    """Resolve active SMS driver based on SMS_CHANNEL environment or config setting."""
    sms_driver = (
        current_app.config.get("SMS_CHANNEL")
        or os.environ.get("SMS_CHANNEL")
        or "sandbox"
        if current_app
        else os.environ.get("SMS_CHANNEL", "sandbox")
    ).lower()

    if sms_driver == "africastalking":
        return AfricasTalkingSMSChannel()
    return SandboxSMSChannel()


# Channel registry
CHANNELS = {
    "email": EmailChannel(),
    "sms": SandboxSMSChannel(),
    "in_app": InAppChannel(),
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
            channels = ["email"]

        log_entries = []
        for ch_name in channels:
            log_entry = OutboundNotificationLog(
                patient_id=patient_id,
                recipient=recipient,
                channel=ch_name,
                event_type=event_type,
                subject=subject,
                body=body,
                status="PENDING",
            )
            db.session.add(log_entry)
            db.session.flush()

            driver = (
                get_sms_channel()
                if ch_name == "sms"
                else CHANNELS.get(ch_name, CHANNELS["email"])
            )
            try:
                success = driver.send(log_entry)
                if success:
                    log_entry.status = "SENT"
                    log_entry.sent_at = datetime.utcnow()
                else:
                    log_entry.status = "FAILED"
                    log_entry.error_message = "Driver returned failure"
            except Exception as exc:
                log_entry.status = "FAILED"
                log_entry.error_message = str(exc)

            log_entries.append(log_entry)

        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error committing notification logs: {e}")

        return log_entries[0] if log_entries else None
