import pyotp
from werkzeug.security import generate_password_hash

from departments.admin.routes import validate_password_complexity
from departments.models.user import User
from extensions import db


def test_password_complexity_rules():
    assert validate_password_complexity("short")[0] is False
    assert validate_password_complexity("nodigitsorsymbols")[0] is False
    assert validate_password_complexity("NoSymbols12345")[0] is False
    assert validate_password_complexity("ValidPass123!")[0] is True


def test_mfa_login_flow(client, app):
    with app.app_context():
        secret = pyotp.random_base32()
        user = User(
            username="admin_mfa_user",
            password=generate_password_hash("ValidPass123!"),
            role="admin",
            totp_secret=secret,
            mfa_enabled=True,
        )
        db.session.add(user)
        db.session.commit()

        # Step 1: Submit correct credentials -> redirected to /mfa_verify
        resp = client.post(
            "/login",
            data={"username": "admin_mfa_user", "password": "ValidPass123!"},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        assert "/mfa_verify" in resp.location

        # Step 2: Submit wrong MFA code -> fails
        with client.session_transaction() as sess:
            sess["mfa_pending_user_id"] = user.id

        resp = client.post(
            "/mfa_verify", data={"code": "000000"}, follow_redirects=True
        )
        assert b"Invalid MFA" in resp.data

        # Step 3: Submit valid MFA code -> succeeds
        totp = pyotp.TOTP(secret)
        valid_code = totp.now()

        with client.session_transaction() as sess:
            sess["mfa_pending_user_id"] = user.id

        resp = client.post(
            "/mfa_verify", data={"code": valid_code}, follow_redirects=True
        )
        assert resp.status_code == 200


def test_staff_without_totp_redirects_to_mfa_setup(client, app):
    """Staff login without totp_secret must force redirect to MFA setup."""
    with app.app_context():
        user = User(
            username="staff_no_mfa",
            password=generate_password_hash("ValidPass123!"),
            role="medicine",
            totp_secret=None,
            mfa_enabled=False,
        )
        db.session.add(user)
        db.session.commit()

    resp = client.post(
        "/login",
        data={"username": "staff_no_mfa", "password": "ValidPass123!"},
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert "/mfa/setup" in resp.location


def test_api_token_requires_totp(client, app):
    """API token endpoint requires valid TOTP code for MFA staff."""
    with app.app_context():
        secret = pyotp.random_base32()
        user = User(
            username="api_mfa_user",
            password=generate_password_hash("ValidPass123!"),
            role="admin",
            totp_secret=secret,
            mfa_enabled=True,
        )
        db.session.add(user)
        db.session.commit()

        # Without TOTP code -> 401
        resp = client.post(
            "/api/auth/token",
            json={"username": "api_mfa_user", "password": "ValidPass123!"},
        )
        assert resp.status_code == 401

        # With valid TOTP code -> 200
        totp = pyotp.TOTP(secret)
        code = totp.now()
        resp = client.post(
            "/api/auth/token",
            json={"username": "api_mfa_user", "password": "ValidPass123!", "totp_code": code},
        )
        assert resp.status_code == 200
        assert "access_token" in resp.get_json()


def test_api_token_lockout(client, app):
    """5 failed token attempts must lock the account."""
    with app.app_context():
        user = User(
            username="lockout_user",
            password=generate_password_hash("ValidPass123!"),
            role="admin",
        )
        db.session.add(user)
        db.session.commit()

        for _ in range(5):
            client.post(
                "/api/auth/token",
                json={"username": "lockout_user", "password": "WrongPassword!"},
            )

        db.session.refresh(user)
        assert user.is_locked() is True

