"""
departments/api/auth.py
────────────────────────
JWT token issuance + a shared `jwt_or_session` decorator.

Endpoints (registered on the `api` blueprint, prefix /api):
  POST /api/auth/token   — exchange username+password for a Bearer JWT
  GET  /api/auth/me      — inspect the current token / session identity
"""

import logging
from functools import wraps

from flask import g, jsonify, request
from flask_jwt_extended import (
    create_access_token,
    get_jwt_identity,
    verify_jwt_in_request,
)
from flask_login import current_user
from werkzeug.security import check_password_hash

from departments.models.user import User
from extensions import db, limiter

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Decorator: accept either a Bearer JWT OR an active session
# ─────────────────────────────────────────────────────────
def jwt_or_session_required(fn):
    """
    Decorator that allows access when the caller presents EITHER:
      • A valid Bearer JWT  (Authorization: Bearer <token>)
      • An active Flask-Login session  (browser / cookie)

    Sets g.api_user to the resolved User object.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        # 1. Try JWT first (only if Authorization header is present)
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                verify_jwt_in_request()
                user_id = get_jwt_identity()
                user = db.session.get(User, int(user_id))
                if not user:
                    return jsonify({"error": "Token user not found"}), 401
                g.api_user = user
                return fn(*args, **kwargs)
            except Exception as exc:
                logger.warning("JWT verification failed: %s", exc)
                return jsonify(
                    {"error": "Invalid or expired token", "detail": str(exc)}
                ), 401

        # 2. Fall back to Flask-Login session
        if current_user.is_authenticated:
            g.api_user = current_user
            return fn(*args, **kwargs)

        return jsonify(
            {"error": "Authentication required. Provide a Bearer token or log in."}
        ), 401

    return wrapper


# ─────────────────────────────────────────────────────────
# Route helpers  (imported by __init__.py)
# ─────────────────────────────────────────────────────────
from . import bp  # noqa: E402  (circular-safe — bp defined before routes)


@bp.route("/auth/token", methods=["POST"])
@limiter.limit("10 per minute")
def get_token():
    """
    Issue a 24-hour Bearer JWT.

    Request (JSON or form-encoded):
        username  — string
        password  — string

    Response 200:
        {
          "access_token": "<jwt>",
          "token_type":   "Bearer",
          "expires_in":   86400,
          "user": { "id": 1, "username": "superadmin", "role": "admin" }
        }
    """
    data = request.get_json(silent=True) or request.form
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()

    if not username or not password:
        return jsonify({"error": "username and password are required"}), 400

    try:
        user = User.query.filter_by(username=username).first()
    except Exception as exc:
        db.session.rollback()
        logger.error("DB error querying user for token: %s", exc)
        return jsonify({"error": "Database query error", "detail": str(exc)}), 500

    if not user or not check_password_hash(user.password, password):
        logger.warning("Failed API token request for username=%s", username)
        return jsonify({"error": "Invalid credentials"}), 401

    if not user.is_active:
        return jsonify({"error": "Account is disabled"}), 403

    token = create_access_token(identity=str(user.id))
    logger.info("API token issued for user_id=%s role=%s", user.id, user.role)

    return jsonify(
        {
            "access_token": token,
            "token_type": "Bearer",
            "expires_in": 86400,  # seconds (matches JWT_ACCESS_TOKEN_EXPIRES = 24 h)
            "user": {
                "id": user.id,
                "username": user.username,
                "role": user.role,
            },
        }
    ), 200


@bp.route("/auth/me", methods=["GET"])
@jwt_or_session_required
def whoami():
    """Return the identity of the currently authenticated caller."""
    u = g.api_user
    return jsonify(
        {
            "id": u.id,
            "username": u.username,
            "role": u.role,
        }
    ), 200
