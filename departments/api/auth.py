"""
departments/api/auth.py
────────────────────────
JWT token issuance + a shared `jwt_or_session` decorator.

Endpoints (registered on the `api` blueprint, prefix /api):
  POST /api/auth/token   — exchange username+password for a Bearer JWT
  GET  /api/auth/me      — inspect the current token / session identity
"""

import logging
from datetime import datetime, timezone
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
      • A valid OAuth2 access token (Authorization: Bearer <token>)
      • An active Flask-Login session  (browser / cookie)

    Sets g.api_user to the resolved User object.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        # 1. Try JWT or OAuth2 first (only if Authorization header is present)
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            # Try JWT first
            try:
                verify_jwt_in_request()
                user_id = get_jwt_identity()
                user = db.session.get(User, int(user_id))
                if not user:
                    return jsonify({"error": "Token user not found"}), 401
                g.api_user = user
                return fn(*args, **kwargs)
            except Exception:  # noqa: S110, BLE001                # JWT failed, try OAuth2
                pass

            # Try OAuth2 access token
            try:
                from departments.models.oauth2 import OAuth2Token

                token_str = auth_header.split(" ")[1]
                token = OAuth2Token.query.filter_by(
                    access_token=token_str, revoked=False
                ).first()
                if token and token.expires_at > datetime.now(timezone.utc).timestamp():
                    g.api_user = token.user
                    return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("OAuth2 verification failed: %s", exc)
                return jsonify(
                    {"error": "Invalid or expired token"}
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
from datetime import timedelta

import pyotp

from . import bp


@bp.route("/auth/token", methods=["POST"])
@limiter.limit("10 per minute")
def get_token():
    """
    Issue a Bearer JWT after credentials & MFA validation.

    Request (JSON or form-encoded):
        username   — string
        password   — string
        totp_code  — string (required for MFA-enrolled/staff roles)

    Response 200:
        {
          "access_token": "<jwt>",
          "token_type":   "Bearer",
          "expires_in":   3600,
          "user": { "id": 1, "username": "superadmin", "role": "admin" }
        }
    """
    data = request.get_json(silent=True) or request.form
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()
    totp_code = (data.get("totp_code") or "").strip()

    if not username or not password:
        return jsonify({"error": "username and password are required"}), 400

    try:
        user = User.query.filter_by(username=username).first()
    except Exception as exc:  # noqa: BLE001
        db.session.rollback()
        logger.error("DB error querying user for token: %s", exc)
        return jsonify({"error": "An internal database error occurred"}), 500

    if not user:
        logger.warning("Failed API token request for non-existent username=%s", username)
        return jsonify({"error": "Invalid credentials"}), 401

    if user.is_locked():
        logger.warning("API token request for locked username=%s", username)
        return jsonify({"error": "Account is locked. Try again later."}), 403

    if not check_password_hash(user.password, password):
        user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
        if user.failed_login_attempts >= 5:
            user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=15)
        db.session.commit()
        logger.warning(
            "Failed API token request for username=%s (attempt %d)",
            username,
            user.failed_login_attempts,
        )
        return jsonify({"error": "Invalid credentials"}), 401

    if not user.is_active:
        return jsonify({"error": "Account is disabled"}), 403

    # Reset failed attempts on success
    user.failed_login_attempts = 0
    user.locked_until = None
    db.session.commit()

    # MFA requirement for API tokens
    MFA_REQUIRED_ROLES = {
        "admin",
        "medicine",
        "imaging",
        "nursing",
        "pharmacy",
        "records",
        "billing",
    }
    if user.totp_secret or user.mfa_enabled or user.role in MFA_REQUIRED_ROLES:
        if not user.totp_secret:
            return jsonify(
                {
                    "error": "MFA enrollment required. Please log into the web portal to setup TOTP."
                }
            ), 403
        if not totp_code or not pyotp.TOTP(user.totp_secret).verify(totp_code):
            logger.warning("Invalid TOTP code on API token request for user_id=%s", user.id)
            return jsonify({"error": "Valid MFA TOTP code required"}), 401

    token = create_access_token(identity=str(user.id))
    logger.info("API token issued for user_id=%s role=%s", user.id, user.role)

    return jsonify(
        {
            "access_token": token,
            "token_type": "Bearer",
            "expires_in": 3600,
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
