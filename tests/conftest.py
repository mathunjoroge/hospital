import os

import pytest

# ── Set test environment BEFORE any app imports ──
os.environ["FLASK_ENV"] = "testing"
os.environ["SQLALCHEMY_DATABASE_URI"] = "sqlite://"  # in-memory
os.environ.setdefault("SECRET_KEY", "testing_secret_key_1234567890")
os.environ.setdefault("SECURITY_PASSWORD_SALT", "testing_salt_1234567890")

# A stable Fernet key for the entire test suite.
# Without this, get_fernet_key() generates a NEW ephemeral key on every call
# in test mode, so encrypt() and decrypt() use different keys and roundtrips
# fail. Derived from a fixed seed; valid only for testing.
os.environ.setdefault(
    "ENCRYPTION_KEY", "thInEUT_C4EOAyIAvI7aHWq0gmhf29_LfWzD7G6sAwo="
)

from app import app as flask_app
from extensions import db, limiter


@pytest.fixture(autouse=True)
def _set_flask_env():
    os.environ["FLASK_ENV"] = "testing"


@pytest.fixture
def app():
    flask_app.config["TESTING"] = True
    flask_app.config["WTF_CSRF_ENABLED"] = False
    flask_app.config["RATELIMIT_ENABLED"] = False
    flask_app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite://"
    limiter.enabled = False

    with flask_app.app_context():
        # Dispose existing engine connections so SQLAlchemy picks up the new URI
        db.engine.dispose()
        db.create_all()
        yield flask_app
        db.session.remove()
        db.drop_all()
        db.engine.dispose()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def admin_user(client, app):
    """Create and log in an admin user via client POST /login."""
    from werkzeug.security import generate_password_hash

    from departments.models.user import User

    with app.app_context():
        user = User.query.filter_by(username="admin_test_fixture").first()
        if not user:
            user = User(
                username="admin_test_fixture",
                password=generate_password_hash("admin123", method="pbkdf2:sha256"),
                role="admin",
            )
            db.session.add(user)
            db.session.commit()

    client.post("/login", data={"username": "admin_test_fixture", "password": "admin123"})
    yield user

