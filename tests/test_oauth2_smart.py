"""Tests for OAuth2 provider and SMART on FHIR launch."""
import pytest
from werkzeug.security import generate_password_hash

from departments.models.oauth2 import OAuth2Client, OAuth2Token
from departments.models.user import User
from extensions import db


def test_oauth2_token_endpoint(app, client):
    """POST /oauth/token issues an access token."""
    with app.app_context():
        # Explicitly create a test user
        user = User.query.filter_by(username="oauth_test_user").first()
        if not user:
            user = User(
                username="oauth_test_user",
                password=generate_password_hash("testpass"),
                role="admin",
            )
            db.session.add(user)
            db.session.commit()

        # Create a test OAuth2 client using Authlib's intended API
        oauth_client = OAuth2Client(
            user_id=user.id,
            client_id="test-client-id",
            client_secret="test-client-secret",
        )
        # Authlib's Mixin requires set_client_metadata() for the metadata dict
        oauth_client.set_client_metadata({
            "client_name": "Test App",
            "redirect_uris": ["http://localhost/callback"],
            "scope": "patient/Patient.read",
        })
        db.session.add(oauth_client)
        db.session.commit()
        
        # Request a token
        resp = client.post("/oauth/token", data={
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
            "scope": "patient/Patient.read",
        })
        
        assert resp.status_code == 200
        data = resp.get_json()
        assert "access_token" in data
        assert data["token_type"] == "Bearer"
        
        # Verify token was saved
        token = OAuth2Token.query.filter_by(access_token=data["access_token"]).first()
        assert token is not None
        assert token.user_id == user.id


def test_smart_launch_endpoint(app, client):
    """GET /oauth/smart/launch stores launch context."""
    resp = client.get("/oauth/smart/launch?iss=http://fhir.example.com&launch=abc123")
    
    assert resp.status_code == 302  # Redirect
    # Check that launch context was stored in session
    with client.session_transaction() as sess:
        assert "smart_launch_context" in sess
        assert sess["smart_launch_context"]["iss"] == "http://fhir.example.com"
        assert sess["smart_launch_context"]["launch"] == "abc123"


def test_smart_launch_missing_params(app, client):
    """GET /oauth/smart/launch without params returns 400."""
    resp = client.get("/oauth/smart/launch")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_oauth2_metadata_endpoint(app, client):
    """GET /oauth/metadata returns SMART discovery document."""
    resp = client.get("/oauth/metadata")
    
    assert resp.status_code == 200
    data = resp.get_json()
    assert "authorization_endpoint" in data
    assert "token_endpoint" in data
    assert "scopes_supported" in data
    assert "patient/Patient.read" in data["scopes_supported"]
