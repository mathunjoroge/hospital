"""
departments/api/audit.py
─────────────────────────
Audit trail module for recording user and system actions in HIMS.
Supports helper function `log_audit_event` and route decorator `@audited`.
"""

import json
import logging
from functools import wraps
from typing import Optional, Union

from flask import request
from flask_login import current_user

try:
    from extensions import db
except ImportError:
    from extensions import db

from departments.models.compliance import AuditLog

logger = logging.getLogger("HIMS.AuditTrail")


def get_client_ip() -> Optional[str]:
    """Extract client IP address, handling proxy headers."""
    if not request:
        return None
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr


def log_audit_event(
    action: str,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    details: Optional[Union[dict, str]] = None,
    user_id: Optional[int] = None,
    username: Optional[str] = None
) -> Optional[AuditLog]:
    """
    Record an append-only audit log entry in the database.

    Args:
        action: Short identifier of action (e.g. 'PATIENT_VIEW', 'BILL_PAY')
        resource_type: Target model/entity (e.g. 'Patient', 'Invoice')
        resource_id: Key/ID of target entity
        details: Additional context as dict or string
        user_id: Optional explicit user ID (defaults to current_user.id)
        username: Optional explicit username (defaults to current_user.username)

    Returns:
        The created AuditLog instance, or None if save failed.
    """
    try:
        # Resolve current user if available in request context
        if user_id is None or username is None:
            try:
                if current_user and current_user.is_authenticated:
                    if user_id is None:
                        user_id = getattr(current_user, 'id', None)
                    if username is None:
                        username = getattr(current_user, 'username', None) or getattr(current_user, 'email', None)
            except RuntimeError:
                pass  # Outside request context

        ip_addr = get_client_ip()
        user_agent = request.headers.get("User-Agent", "")[:250] if request else None

        formatted_details = None
        if details is not None:
            if isinstance(details, (dict, list)):
                formatted_details = json.dumps(details)
            else:
                formatted_details = str(details)[:2000]

        entry = AuditLog(
            user_id=user_id,
            username=username or "system",
            action=action,
            resource_type=resource_type,
            resource_id=str(resource_id) if resource_id is not None else None,
            ip_address=ip_addr,
            user_agent=user_agent,
            details=formatted_details
        )

        db.session.add(entry)
        db.session.commit()
        logger.info(f"AUDIT | action={action} | resource={resource_type}:{resource_id} | user={username} ({user_id}) | ip={ip_addr}")
        return entry
    except Exception as e:
        db.session.rollback()
        logger.error(f"Failed to record audit log: {e}", exc_info=True)
        return None


def audited(action: str, resource_type: Optional[str] = None):
    """
    Route decorator to automatically audit invocations of Flask routes.

    Usage:
        @app.route('/patient/<id>')
        @login_required
        @audited(action='PATIENT_VIEW', resource_type='Patient')
        def view_patient(id):
            ...
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Resolve resource_id from kwargs if available
            res_id = kwargs.get('patient_id') or kwargs.get('id') or kwargs.get('pk')
            response = f(*args, **kwargs)
            log_audit_event(
                action=action,
                resource_type=resource_type,
                resource_id=res_id,
                details={"method": request.method, "path": request.path} if request else None
            )
            return response
        return decorated_function
    return decorator
