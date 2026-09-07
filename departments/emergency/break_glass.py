"""
departments/emergency/break_glass.py
──────────────────────────────────────
Phase D — Break-glass Emergency Access Core Logic

Provides:
  - invoke_break_glass():  create a time-boxed override grant
  - check_break_glass():   verify an active grant exists for a user/patient
  - expire_break_glass():  expire stale grants (called by scheduler or manually)
  - break_glass_required(): decorator that allows access when a valid grant exists,
                            even if the user's normal RBAC role would deny it
"""

import logging
from datetime import datetime, timedelta, timezone
from functools import wraps

from flask import abort, g, request

from departments.api.audit import log_audit_event
from departments.models.break_glass import BreakGlassAccessLog
from departments.rbac import get_effective_user
from extensions import db

logger = logging.getLogger(__name__)

# Break-glass roles: who is allowed to invoke emergency override
BREAK_GLASS_ELIGIBLE_ROLES = {"doctor", "medicine", "nursing", "admin"}

# Default validity window for an override grant
DEFAULT_BREAK_GLASS_DURATION_HOURS = 4


def invoke_break_glass(
    reason: str,
    patient_id: str = None,
    resource_type: str = None,
    resource_id: str = None,
    duration_hours: int = DEFAULT_BREAK_GLASS_DURATION_HOURS,
    user=None,
) -> BreakGlassAccessLog:
    """
    Record a break-glass override invocation for the specified or currently authenticated user.
    """
    if not reason or not reason.strip():
        raise ValueError(
            "A mandatory clinical reason must be provided for break-glass access."
        )

    if not user:
        user = get_effective_user()
    if not user:
        raise PermissionError("No authenticated user found.")

    if user.role not in BREAK_GLASS_ELIGIBLE_ROLES:
        raise PermissionError(
            f"Role '{user.role}' is not authorised to invoke break-glass access. "
            f"Eligible roles: {sorted(BREAK_GLASS_ELIGIBLE_ROLES)}"
        )

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=duration_hours)

    ip_addr = None
    ua_string = None
    if request:
        forwarded = request.headers.get("X-Forwarded-For")
        ip_addr = forwarded.split(",")[0].strip() if forwarded else request.remote_addr
        ua_string = request.headers.get("User-Agent", "")[:255]

    log_entry = BreakGlassAccessLog(
        user_id=user.id,
        username=user.username,
        user_role=user.role,
        patient_id=patient_id,
        resource_type=resource_type,
        resource_id=resource_id,
        reason=reason.strip(),
        invoked_at=now,
        expires_at=expires_at,
        is_active=True,
        supervisor_notified=False,
        ip_address=ip_addr,
        user_agent=ua_string,
    )
    db.session.add(log_entry)
    db.session.commit()

    # Record a prominent audit trail entry, distinct from normal audit rows
    log_audit_event(
        action="BREAK_GLASS_OVERRIDE",
        resource_type=resource_type or "Patient",
        resource_id=patient_id or resource_id,
        details={
            "break_glass_id": log_entry.id,
            "reason": reason.strip(),
            "expires_at": expires_at.isoformat(),
            "patient_id": patient_id,
            "user_role": user.role,
        },
        user_id=user.id,
        username=user.username,
    )
    logger.warning(
        "BREAK-GLASS INVOKED | user=%s (%s) | patient=%s | reason=%s | expires=%s",
        user.username,
        user.role,
        patient_id,
        reason.strip(),
        expires_at.isoformat(),
    )

    # Notify supervisors (async-safe: errors are caught and logged)
    _notify_supervisors(log_entry)

    return log_entry


def check_break_glass(
    user_id: int, patient_id: str = None
) -> BreakGlassAccessLog | None:
    """
    Return the most-recent valid (active & not expired) break-glass grant for
    the given user, optionally scoped to a specific patient.

    Returns None if no valid grant exists.
    """
    now = datetime.now(timezone.utc)
    q = db.session.query(BreakGlassAccessLog).filter(
        BreakGlassAccessLog.user_id == user_id,
        BreakGlassAccessLog.is_active.is_(True),
        BreakGlassAccessLog.expires_at > now,
    )
    if patient_id:
        q = q.filter(
            (BreakGlassAccessLog.patient_id == patient_id)
            | (BreakGlassAccessLog.patient_id.is_(None))
        )
    return q.order_by(BreakGlassAccessLog.invoked_at.desc()).first()


def expire_stale_grants() -> int:
    """
    Mark all expired break-glass grants as inactive.
    Intended to be called by a periodic scheduler or maintenance task.

    Returns:
        Number of grants deactivated.
    """
    now = datetime.now(timezone.utc)
    stale = (
        db.session.query(BreakGlassAccessLog)
        .filter(
            BreakGlassAccessLog.is_active.is_(True),
            BreakGlassAccessLog.expires_at <= now,
        )
        .all()
    )
    count = len(stale)
    for grant in stale:
        grant.is_active = False
    if count:
        db.session.commit()
        logger.info("Expired %d stale break-glass grants.", count)
    return count


def break_glass_required(*normal_roles):
    """
    Decorator that allows access to a view either via normal RBAC roles OR via
    an active break-glass grant.

    Usage:
        @bp.route('/patient/<patient_id>/record')
        @login_required
        @break_glass_required('medicine', 'nursing')
        def view_sensitive_record(patient_id):
            ...

    The decorator checks:
    1. User has one of `normal_roles` → allow (normal path).
    2. User has no matching role but has a valid active break-glass grant → allow
       (override path). Sets `g.break_glass_active = True` so views can display
       a visible UI banner.
    3. Neither → 403.
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            user = get_effective_user()
            if not user:
                abort(403)

            # Normal RBAC path
            if user.role in normal_roles:
                g.break_glass_active = False
                return fn(*args, **kwargs)

            # Break-glass path
            patient_id = kwargs.get("patient_id")
            grant = check_break_glass(user.id, patient_id=patient_id)
            if grant:
                g.break_glass_active = True
                g.break_glass_grant = grant
                logger.info(
                    "BREAK-GLASS ACCESS | user=%s | grant_id=%d | patient=%s",
                    user.username,
                    grant.id,
                    patient_id,
                )
                return fn(*args, **kwargs)

            abort(403)

        return wrapper

    return decorator


def _notify_supervisors(log_entry: BreakGlassAccessLog):
    """
    Send an immediate notification to all admin users when break-glass is invoked.
    Uses the Phase B NotificationDispatcher; errors are caught so they never block
    the break-glass grant itself.
    """
    try:
        from departments.models.user import User
        from departments.notifications.dispatcher import (
            EVENT_BREAK_GLASS,
            NotificationDispatcher,
        )

        admins = User.query.filter_by(role="admin").all()
        for admin in admins:
            NotificationDispatcher.dispatch_event(
                event_type=EVENT_BREAK_GLASS,
                recipient=f"admin-{admin.id}@hospital.internal",
                subject="⚠️ BREAK-GLASS ACCESS INVOKED",
                body=(
                    f"URGENT — Emergency access override invoked.\n\n"
                    f"User:     {log_entry.username} ({log_entry.user_role})\n"
                    f"Patient:  {log_entry.patient_id or 'N/A'}\n"
                    f"Reason:   {log_entry.reason}\n"
                    f"Invoked:  {log_entry.invoked_at.isoformat()}\n"
                    f"Expires:  {log_entry.expires_at.isoformat()}\n\n"
                    "Review this event immediately in the admin audit trail."
                ),
                patient_id=log_entry.patient_id,
                channels=["email", "in_app"],
            )

        log_entry.supervisor_notified = True
        log_entry.supervisor_notified_at = datetime.now(timezone.utc)
        db.session.commit()

    except Exception as exc:
        logger.error(
            "Failed to notify supervisors of break-glass event (grant_id=%s): %s",
            log_entry.id,
            exc,
        )
