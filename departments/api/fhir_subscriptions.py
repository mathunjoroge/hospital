"""
departments/api/fhir_subscriptions.py
──────────────────────────────────────
FHIR R4 REST-hook Subscription Webhook Engine.
Handles real-time webhook notification dispatching with HMAC-SHA256 payload signing.
"""

import hmac
import json
import logging
from datetime import datetime, timezone
from typing import Any

import requests

from departments.models.fhir_subscription import FHIRSubscription
from extensions import db

logger = logging.getLogger(__name__)


def generate_hmac_signature(secret_token: str, payload_bytes: bytes) -> str:
    """
    Generate HMAC-SHA256 signature hex digest for payload verification.
    """
    key = secret_token.encode("utf-8")
    sig = hmac.new(key, payload_bytes, digestmod="sha256").hexdigest()
    return f"sha256={sig}"


def create_subscription(
    criteria: str,
    endpoint_url: str,
    secret_token: str | None = None,
    reason: str | None = None,
) -> FHIRSubscription:
    """
    Register new active FHIR Subscription.
    """
    sub = FHIRSubscription(
        criteria=criteria.strip(),
        endpoint_url=endpoint_url.strip(),
        secret_token=secret_token.strip() if secret_token else None,
        reason=reason or "HIE Real-Time Interoperability Webhook",
        status="active",
    )
    db.session.add(sub)
    db.session.commit()
    return sub


def list_subscriptions() -> list[dict[str, Any]]:
    """
    Return all active FHIR Subscription resources.
    """
    subs = FHIRSubscription.query.filter_by(status="active").all()
    return [s.to_fhir() for s in subs]


def delete_subscription(subscription_id: str) -> bool:
    """
    Deactivate / delete a FHIR Subscription by ID.
    """
    sub = FHIRSubscription.query.filter_by(subscription_id=subscription_id).first()
    if not sub:
        return False
    sub.status = "off"
    db.session.commit()
    return True


def dispatch_subscription_event(resource_type: str, resource_data: dict[str, Any]) -> int:
    """
    Dispatch real-time FHIR notification webhooks for `resource_type` to registered subscribers.
    Returns the number of webhooks successfully notified.
    """
    active_subs = FHIRSubscription.query.filter_by(status="active").all()
    matching_subs = [
        s for s in active_subs
        if s.criteria.lower() in (resource_type.lower(), "*") or resource_type.lower() in s.criteria.lower()
    ]

    if not matching_subs:
        return 0

    notification_bundle = {
        "resourceType": "Bundle",
        "type": "history",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "entry": [{"resource": resource_data}],
    }

    payload_bytes = json.dumps(notification_bundle).encode("utf-8")
    success_count = 0

    for sub in matching_subs:
        headers = {
            "Content-Type": "application/fhir+json",
            "User-Agent": "HIMS-FHIR-Subscription-Engine/1.0",
        }
        if sub.secret_token:
            headers["X-FHIR-Signature"] = generate_hmac_signature(sub.secret_token, payload_bytes)
            headers["X-FHIR-Token"] = sub.secret_token

        try:
            resp = requests.post(sub.endpoint_url, data=payload_bytes, headers=headers, timeout=3.0)
            if resp.status_code in (200, 201, 202, 204):
                success_count += 1
                sub.last_triggered_at = datetime.now(timezone.utc).replace(tzinfo=None)
                sub.failure_count = 0
            else:
                sub.failure_count += 1
                if sub.failure_count >= 5:
                    sub.status = "error"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Subscription webhook dispatch failed for %s: %s", sub.endpoint_url, exc)
            sub.failure_count += 1
            if sub.failure_count >= 5:
                sub.status = "error"

    db.session.commit()
    return success_count
