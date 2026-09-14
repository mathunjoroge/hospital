from functools import wraps

from flask import abort, g, session
from flask_login import current_user

ROLE_PERMISSIONS = {
    "admin": ["*"],
    "doctor": ["read", "write", "prescribe"],
    "nurse": ["read", "write", "triage"],
    "pharmacist": ["read", "dispense"],
    "lab_tech": ["read", "laboratory"],
    "radiology": ["read", "imaging"],
    "api": ["read", "write"],
}



ROLE_ALIASES = {
    "doctor": {"doctor", "medicine", "clinical"},
    "medicine": {"doctor", "medicine", "clinical"},
    "clinical": {"doctor", "medicine", "clinical"},
    "nurse": {"nurse", "nursing"},
    "nursing": {"nurse", "nursing"},
}


def get_effective_user():
    """Return g.api_user if set via JWT, otherwise current_user if authenticated."""
    api_user = getattr(g, "api_user", None)
    if api_user:
        return api_user
    try:
        if current_user and current_user.is_authenticated:
            return current_user
    except (AttributeError, RuntimeError):
        pass
    return None


def get_effective_role():
    user = get_effective_user()
    if not user:
        return None
    role = (user.role or "").lower()
    if role == "admin" and "switched_user" in session:
        return (session.get("switched_user") or "").lower()
    return role


def roles_required(*roles):
    """
    Decorator to enforce role-based access control across department view functions.
    Supports both Flask-Login sessions and JWT Bearer tokens (via g.api_user).

    Admin users automatically have access to all routes (standard RBAC pattern).
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            user = get_effective_user()
            if not user:
                abort(403)

            user_role = (user.role or "").lower()
            effective_role = get_effective_role()

            # Admin always has access (standard RBAC pattern) unless role switch is active
            if user_role == "admin" and "switched_user" not in session:
                return fn(*args, **kwargs)

            # Build allowed roles set including role aliases
            allowed_roles = {r.lower() for r in roles}
            for r in list(allowed_roles):
                if r in ROLE_ALIASES:
                    allowed_roles.update(ROLE_ALIASES[r])

            # Check if effective role is in allowed roles
            if effective_role not in allowed_roles:
                abort(403)

            return fn(*args, **kwargs)

        return wrapper

    return decorator
