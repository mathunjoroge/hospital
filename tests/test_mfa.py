import pyotp
from werkzeug.security import generate_password_hash
from extensions import db
from departments.models.user import User
from departments.admin.routes import validate_password_complexity

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
            mfa_enabled=True
        )
        db.session.add(user)
        db.session.commit()

        # Step 1: Submit correct credentials -> redirected to /mfa_verify
        resp = client.post('/login', data={'username': 'admin_mfa_user', 'password': 'ValidPass123!'}, follow_redirects=False)
        assert resp.status_code == 302
        assert '/mfa_verify' in resp.location

        # Step 2: Submit wrong MFA code -> fails
        with client.session_transaction() as sess:
            sess['mfa_pending_user_id'] = user.id

        resp = client.post('/mfa_verify', data={'code': '000000'}, follow_redirects=True)
        assert b'Invalid MFA' in resp.data

        # Step 3: Submit valid MFA code -> succeeds
        totp = pyotp.TOTP(secret)
        valid_code = totp.now()

        with client.session_transaction() as sess:
            sess['mfa_pending_user_id'] = user.id

        resp = client.post('/mfa_verify', data={'code': valid_code}, follow_redirects=True)
        assert resp.status_code == 200
