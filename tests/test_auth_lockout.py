import pytest

from app import app
from departments.models.user import User
from extensions import db, limiter


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    app.config["RATELIMIT_ENABLED"] = True
    limiter.enabled = True
    if hasattr(limiter, "_storage") and hasattr(limiter._storage, "reset"):
        limiter._storage.reset()

    with app.app_context():
        db.create_all()
        user = User(
            id=999,
            username="testuser",
            password="pbkdf2:sha256:600000$P98G$2b2d2f2...",
            role="admin",
        )
        db.session.add(user)
        db.session.commit()

        with app.test_client() as client:
            yield client
        db.session.remove()
        db.drop_all()
        limiter.enabled = False


def test_login_lockout(client):
    """Verify that multiple failed logins lock the account."""
    for i in range(5):
        resp = client.post(
            "/login", data={"username": "testuser", "password": "wrongpassword"}
        )
        assert b"Invalid credentials" in resp.data or resp.status_code == 429

    user = User.query.filter_by(username="testuser").first()
    assert user.failed_login_attempts == 5
    assert user.locked_until is not None


def test_login_rate_limiting(client):
    """Verify that we are rate limited after 5 POSTs."""
    user = User(id=888, username="rateuser", password="pwd", role="admin")
    db.session.add(user)
    db.session.commit()

    for i in range(6):
        resp = client.post("/login", data={"username": "rateuser", "password": "pwd"})
        if i == 5:
            assert resp.status_code == 429
