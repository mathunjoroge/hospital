"""
departments/sso/routes.py
──────────────────────────
Flask HTTP routes for Enterprise LDAP / Active Directory & OIDC Single Sign-On (SSO).

Endpoints:
  GET  /auth/sso/login          – Initiate OIDC Login redirect to Identity Provider (Azure AD / Keycloak)
  GET  /auth/sso/callback       – Handle OIDC callback from IdP and authenticate user session
  POST /auth/sso/ldap           – Direct Active Directory / LDAP bind login API
  GET  /auth/sso/status         – SSO Configuration & Health check endpoint
"""
import logging

from flask import jsonify, redirect, request, session, url_for
from flask_login import login_user

from . import bp
from .sso_engine import SSOEngine, SSOError

logger = logging.getLogger(__name__)
_sso_engine = SSOEngine()


@bp.route("/login", methods=["GET"])
def sso_login():
    """
    GET /auth/sso/login
    Initiate OIDC authorization flow by redirecting to the enterprise IdP.
    """
    redirect_uri = url_for("sso.sso_callback", _external=True)
    state = request.args.get("state", "sso_auth_state")

    auth_url = _sso_engine.get_oidc_authorization_url(redirect_uri=redirect_uri, state=state)
    return redirect(auth_url)


@bp.route("/callback", methods=["GET"])
def sso_callback():
    """
    GET /auth/sso/callback?code=xxx&state=yyy
    Receive authorization code from enterprise OIDC IdP, provision/sync user, and log in.
    """
    code = request.args.get("code")

    if not code:
        return jsonify({"error": "Missing authorization code from Identity Provider."}), 400

    try:
        redirect_uri = url_for("sso.sso_callback", _external=True)
        user_claims = _sso_engine.process_oidc_callback(code=code, redirect_uri=redirect_uri)
        user = _sso_engine.provision_or_sync_user(user_claims)

        login_user(user)
        session["sso_provider"] = "oidc"

        if request.headers.get("Accept") == "application/json":
            return jsonify({
                "status": "success",
                "message": f"Successfully authenticated via OIDC SSO as {user.username}",
                "user_id": user.id,
                "username": user.username,
                "role": user.role,
            }), 200

        return redirect(url_for("ui_dashboard.dashboard") if "ui_dashboard.dashboard" in session else "/")

    except SSOError as exc:
        return jsonify({"error": str(exc)}), 401
    except Exception:
        logger.exception("Unexpected error during OIDC SSO callback handling")
        return jsonify({"error": "Internal server error during SSO processing."}), 500


@bp.route("/ldap", methods=["POST"])
def ldap_login():
    """
    POST /auth/sso/ldap
    Authenticate user via direct LDAP / Active Directory bind.

    JSON payload:
      username, password
    """
    data = request.get_json(silent=True) or {}
    username = data.get("username") or request.form.get("username")
    password = data.get("password") or request.form.get("password")

    if not username or not password:
        return jsonify({"error": "username and password are required."}), 400

    try:
        user_info = _sso_engine.authenticate_ldap(username=username, password=password)
        user = _sso_engine.provision_or_sync_user(user_info)

        login_user(user)
        session["sso_provider"] = "ldap"

        return jsonify({
            "status": "success",
            "message": f"Authenticated via Active Directory / LDAP as {user.username}",
            "user_id": user.id,
            "username": user.username,
            "role": user.role,
        }), 200

    except SSOError as exc:
        return jsonify({"error": str(exc)}), 401
    except Exception:
        logger.exception("Unexpected error during LDAP authentication")
        return jsonify({"error": "Internal server error during Active Directory login."}), 500


@bp.route("/status", methods=["GET"])
def sso_status():
    """
    GET /auth/sso/status
    Returns SSO configuration and provider health status.
    """
    return jsonify({
        "sso_enabled": _sso_engine.enabled,
        "provider": _sso_engine.provider,
        "oidc_issuer": _sso_engine.oidc_issuer,
        "ldap_server": _sso_engine.ldap_server,
        "supported_methods": ["OIDC (Azure AD/Keycloak)", "Active Directory / LDAP Bind"],
    }), 200
