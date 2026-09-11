"""
OAuth2 Authorization Server for SMART on FHIR.

Endpoints:
  POST /oauth/token — Exchange authorization code for access token
  POST /oauth/revoke — Revoke an access token
  GET  /oauth/metadata — SMART on FHIR discovery endpoint
"""
from datetime import datetime, timezone

from authlib.integrations.flask_oauth2 import AuthorizationServer
from authlib.oauth2 import ResourceProtector
from flask import Blueprint, jsonify, request

from departments.models.oauth2 import OAuth2Client, OAuth2Token
from extensions import db

oauth_bp = Blueprint("oauth2", __name__, url_prefix="/oauth")

# Initialize Authlib OAuth2 server
oauth = AuthorizationServer()
require_oauth = ResourceProtector()


def query_client(client_id):
    """OAuth2 client lookup."""
    return OAuth2Client.query.filter_by(client_id=client_id).first()


def save_token(token_data, request):
    """Save OAuth2 access token."""
    client = request.client
    user = request.user or client.user
    expires_in = token_data.get("expires_in", 3600)

    token = OAuth2Token(
        client_id=client.client_id,
        token_type=token_data.get("token_type", "Bearer"),
        access_token=token_data.get("access_token"),
        refresh_token=token_data.get("refresh_token"),
        scope=token_data.get("scope", ""),
        expires_in=expires_in,
        user_id=user.id,
    )
    db.session.add(token)
    db.session.commit()
    return token


def init_oauth(app):
    """Initialize OAuth2 server with Flask app."""
    oauth.init_app(app, query_client=query_client, save_token=save_token)


@oauth_bp.route("/token", methods=["POST"])
def issue_token():
    """Issue an OAuth2 access token."""
    # For simplicity, use client_credentials grant initially
    # In production, this would validate authorization code or refresh token
    client_id = request.form.get("client_id")
    client_secret = request.form.get("client_secret")
    scope = request.form.get("scope", "")

    client = OAuth2Client.query.filter_by(
        client_id=client_id, client_secret=client_secret
    ).first()
    if not client:
        return jsonify({"error": "invalid_client"}), 401

    # Generate a simple access token
    import secrets
    access_token = secrets.token_urlsafe(32)
    expires_in = 3600  # 1 hour

    token = OAuth2Token(
        client_id=client.client_id,
        token_type="Bearer",
        access_token=access_token,
        scope=scope,
        expires_in=expires_in,
        user_id=client.user_id,
    )
    db.session.add(token)
    db.session.commit()

    return jsonify({
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": expires_in,
        "scope": scope,
    })


@oauth_bp.route("/revoke", methods=["POST"])
def revoke_token():
    """Revoke an OAuth2 access token."""
    token_str = request.form.get("token")
    token = OAuth2Token.query.filter_by(access_token=token_str).first()
    if token:
        token.revoked = True
        db.session.commit()
    return jsonify({"status": "revoked"})


@oauth_bp.route("/metadata", methods=["GET"])
def smart_metadata():
    """SMART on FHIR discovery endpoint."""
    return jsonify({
        "authorization_endpoint": "/oauth/authorize",
        "token_endpoint": "/oauth/token",
        "token_endpoint_auth_methods_supported": ["client_secret_post"],
        "scopes_supported": [
            "patient/Patient.read",
            "patient/Observation.read",
            "patient/Condition.read",
            "patient/MedicationRequest.read",
            "user/Patient.read",
            "user/Encounter.read",
        ],
        "response_types_supported": ["code"],
        "capabilities": ["launch-ehr", "client-public", "client-confidential-symmetric"],
    })


# ── SMART on FHIR Launch Endpoints ─────────────────────────────────────────
@oauth_bp.route("/smart/launch", methods=["GET"])
def smart_launch():
    """
    SMART on FHIR EHR Launch endpoint.

    External EHR redirects here with:
      - iss: FHIR server URL
      - launch: launch context token

    We redirect to our app with the launch context stored in session.
    """
    from flask import redirect, session

    iss = request.args.get("iss")
    launch = request.args.get("launch")

    if not iss or not launch:
        return jsonify({"error": "Missing iss or launch parameter"}), 400

    # Store launch context in session
    session["smart_launch_context"] = {
        "iss": iss,
        "launch": launch,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    # Redirect to the app's main page (or a specific SMART callback handler)
    # For now, redirect to root with a flag
    return redirect("/?smart_launch=true")


@oauth_bp.route("/smart/callback", methods=["GET"])
def smart_callback():
    """
    SMART on FHIR callback after authorization.

    Handles the authorization code exchange and sets up the user session.
    """

    code = request.args.get("code")
    state = request.args.get("state")

    if not code:
        return jsonify({"error": "Missing authorization code"}), 400

    # In a full implementation, this would:
    # 1. Exchange the code for an access token
    # 2. Extract patient/encounter context from the token
    # 3. Log the user in or create a session

    # For now, just acknowledge receipt
    return jsonify({
        "status": "callback_received",
        "code": code[:10] + "...",
        "state": state
    })
