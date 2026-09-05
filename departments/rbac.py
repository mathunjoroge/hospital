from functools import wraps

from flask import abort, g, session
from flask_login import current_user


def get_effective_user():
    """Return g.api_user if set via JWT, otherwise current_user if authenticated."""
    api_user = getattr(g, 'api_user', None)
    if api_user:
        return api_user
    if current_user.is_authenticated:
        return current_user
    return None

def get_effective_role():
    user = get_effective_user()
    if not user:
        return None
    if user.role == 'admin' and 'switched_user' in session:
        return session['switched_user']
    return user.role

def roles_required(*roles):
    """
    Decorator to enforce role-based access control across department view functions.
    Supports both Flask-Login sessions and JWT Bearer tokens (via g.api_user).
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            user = get_effective_user()
            if not user:
                abort(403)
            user_role = user.role
            effective_role = get_effective_role()
            if user_role not in roles and effective_role not in roles:
                abort(403)
            return fn(*args, **kwargs)
        return wrapper
    return decorator

