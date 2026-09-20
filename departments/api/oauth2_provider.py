"""
OAuth2 Authorization Server for SMART on FHIR.

EndPoints:
  GET/POST /oauth/authorize — PKCE Authorization Code Grant confirmation
  POST     /oauth/token     — Exchange authorization code or client credentials for token
  POST     /oauth/revoke    — Revoke an access token
  GET      /oauth/metadata  — SMART on FHIR discovery endpoint
"""

import secrets
from datetime import datetime, timezone

from authlib.integrations.flask_oauth2 import AuthorizationServer, ResourceProtector
from authlib.oauth2.rfc6749 import grants
from authlib.oauth2.rfc7636 import CodeChallenge
from flask import Blueprint, jsonify, redirect, render_template_string, request, session

from departments.models.oauth2 import OAuth2AuthorizationCode, OAuth2Client, OAuth2Token
from departments.models.user import User
from extensions import db

oauth_bp = Blueprint("oauth2", __name__, url_prefix="/oauth")

# Initialize Authlib OAuth2 server
oauth = AuthorizationServer()
require_oauth = ResourceProtector()


class AuthorizationCodeGrant(grants.AuthorizationCodeGrant):
    """PKCE-enabled Authorization Code Grant for SMART on FHIR."""

    TOKEN_ENDPOINT_AUTH_METHODS = ["client_secret_basic", "client_secret_post", "none"]

    def save_authorization_code(self, code, request):
        client = request.client
        auth_code = OAuth2AuthorizationCode(
            code=code,
            client_id=client.client_id,
            redirect_uri=request.redirect_uri,
            scope=request.scope,
            user_id=request.user.id if request.user else client.user_id,
            code_challenge=request.code_challenge,
            code_challenge_method=request.code_challenge_method,
        )
        db.session.add(auth_code)
        db.session.commit()
        return auth_code

    def query_authorization_code(self, code, client):
        item = OAuth2AuthorizationCode.query.filter_by(
            code=code, client_id=client.client_id
        ).first()
        if item and not item.is_expired():
            return item
        return None

    def delete_authorization_code(self, authorization_code):
        db.session.delete(authorization_code)
        db.session.commit()

    def authenticate_user(self, authorization_code):
        return User.query.get(authorization_code.user_id)


def query_client(client_id):
    """OAuth2 client lookup."""
    return OAuth2Client.query.filter_by(client_id=client_id).first()


def save_token(token_data, request):
    """Save OAuth2 access token."""
    client = request.client
    user = request.user or (client.user if client else None)
    expires_in = token_data.get("expires_in", 3600)

    token = OAuth2Token(
        client_id=client.client_id if client else None,
        token_type=token_data.get("token_type", "Bearer"),
        access_token=token_data.get("access_token"),
        refresh_token=token_data.get("refresh_token"),
        scope=token_data.get("scope", ""),
        expires_in=expires_in,
        user_id=user.id if user else (client.user_id if client else None),
    )
    db.session.add(token)
    db.session.commit()
    return token


def init_oauth(app):
    """Initialize OAuth2 server with Flask app."""
    oauth.init_app(app, query_client=query_client, save_token=save_token)
    try:
        oauth.register_grant(AuthorizationCodeGrant, [CodeChallenge(required=False)])
    except Exception:
        pass


@oauth_bp.route("/authorize", methods=["GET", "POST"])
def authorize():
    """PKCE Authorization Code Grant authorization endpoint."""
    user_id = session.get("user_id")
    user = User.query.get(user_id) if user_id else None
    if not user:
        # Fallback for API / headless test contexts
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            from departments.api.auth import decode_access_token

            payload = decode_access_token(auth_header.split(" ")[1])
            if payload:
                user = User.query.get(payload.get("sub"))

    if not user:
        return jsonify({"error": "unauthorized_user", "message": "Login required for authorization"}), 401

    if request.method == "GET":
        try:
            grant = oauth.validate_consent_request(end_user=user)
        except Exception as e:
            return jsonify({"error": "invalid_request", "error_description": str(e)}), 400

        return render_template_string(
            """
            <h2>SMART on FHIR Authorization Request</h2>
            <p>Application: <strong>{{ grant.client.client_name or grant.client.client_id }}</strong></p>
            <p>Scopes requested: <code>{{ grant.request.scope }}</code></p>
            <form method="POST">
                <input type="hidden" name="confirm" value="true">
                <button type="submit" style="padding: 10px 20px; background-color: #2563eb; color: white; border: none; border-radius: 4px; cursor: pointer;">Grant Access</button>
            </form>
            """,
            grant=grant,
        )

    confirm = request.form.get("confirm")
    grant_user = user if confirm == "true" else None
    return oauth.create_authorization_response(grant_user=grant_user)


@oauth_bp.route("/token", methods=["POST"])
def issue_token():
    """Issue an OAuth2 access token via Authlib or client_credentials fallback."""
    grant_type = request.form.get("grant_type")
    if grant_type:
        return oauth.create_token_response()

    # Legacy/direct client_credentials lookup fallback
    client_id = request.form.get("client_id")
    client_secret = request.form.get("client_secret")
    scope = request.form.get("scope", "")

    client = OAuth2Client.query.filter_by(
        client_id=client_id, client_secret=client_secret
    ).first()
    if not client:
        return jsonify({"error": "invalid_client"}), 401

    access_token = secrets.token_urlsafe(32)
    expires_in = 3600

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

    return jsonify(
        {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": expires_in,
            "scope": scope,
        }
    )


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
    base_url = request.host_url.rstrip("/")
    return jsonify(
        {
            "authorization_endpoint": f"{base_url}/oauth/authorize",
            "token_endpoint": f"{base_url}/oauth/token",
            "token_endpoint_auth_methods_supported": [
                "client_secret_post",
                "client_secret_basic",
                "none",
            ],
            "scopes_supported": [
                "patient/Patient.read",
                "patient/Observation.read",
                "patient/Condition.read",
                "patient/DiagnosticReport.read",
                "patient/MedicationRequest.read",
                "patient/AllergyIntolerance.read",
                "patient/MedicationAdministration.read",
                "patient/Immunization.read",
                "patient/Procedure.read",
                "user/Patient.read",
                "user/Encounter.read",
                "openid",
                "profile",
                "launch",
                "fhirUser",
            ],
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "client_credentials", "refresh_token"],
            "code_challenge_methods_supported": ["S256", "plain"],
            "capabilities": [
                "launch-ehr",
                "launch-standalone",
                "client-public",
                "client-confidential-symmetric",
                "context-passthrough-patient",
            ],
        }
    )


# ── SMART on FHIR Launch Endpoints ─────────────────────────────────────────
@oauth_bp.route("/smart/launch", methods=["GET"])
def smart_launch():
    """SMART on FHIR EHR Launch endpoint."""
    iss = request.args.get("iss")
    launch = request.args.get("launch")

    if not iss or not launch:
        return jsonify({"error": "Missing iss or launch parameter"}), 400

    session["smart_launch_context"] = {
        "iss": iss,
        "launch": launch,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return redirect("/?smart_launch=true")


@oauth_bp.route("/smart/callback", methods=["GET"])
def smart_callback():
    """SMART on FHIR callback after authorization."""
    code = request.args.get("code")
    state = request.args.get("state")

    if not code:
        return jsonify({"error": "Missing authorization code"}), 400

    return jsonify(
        {"status": "callback_received", "code": code[:10] + "...", "state": state}
    )

