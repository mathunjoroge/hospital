from functools import wraps
from flask_login import current_user
from flask import abort, session

def get_effective_role():
    if current_user.is_authenticated and current_user.role == 'admin' and 'switched_user' in session:
        return session['switched_user']
    return current_user.role if current_user.is_authenticated else None

def roles_required(*roles):
    """
    Decorator to enforce role-based access control across department view functions.
    Usage:
        @bp.route('/dashboard')
        @login_required
        @roles_required('admin', 'nursing')
        def dashboard():
            ...
    Returns HTTP 403 Forbidden if user is unauthenticated or role is not permitted.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated:
                abort(403)
            user_role = current_user.role
            effective_role = get_effective_role()
            if user_role not in roles and effective_role not in roles:
                abort(403)
            return fn(*args, **kwargs)
        return wrapper
    return decorator
