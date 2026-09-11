"""
OAuth2 models for SMART on FHIR authorization server.

Implements the Authlib required schema for OAuth2 Authorization Code flow.
"""

from authlib.integrations.sqla_oauth2 import (
    OAuth2AuthorizationCodeMixin,
    OAuth2ClientMixin,
    OAuth2TokenMixin,
)

from extensions import db


class OAuth2Client(OAuth2ClientMixin, db.Model):
    __tablename__ = "oauth2_clients"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    user = db.relationship("User", backref="oauth_clients")

    def __repr__(self):
        return f"<OAuth2Client {self.client_id}>"


class OAuth2AuthorizationCode(OAuth2AuthorizationCodeMixin, db.Model):
    __tablename__ = "oauth2_authorization_codes"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    user = db.relationship("User", backref="oauth_codes")

    def __repr__(self):
        return f"<OAuth2AuthorizationCode {self.code}>"


class OAuth2Token(OAuth2TokenMixin, db.Model):
    __tablename__ = "oauth2_tokens"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    user = db.relationship("User", backref="oauth_tokens")

    def __repr__(self):
        return f"<OAuth2Token {self.access_token}>"
