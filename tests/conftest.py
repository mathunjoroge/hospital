import os

import pytest

# ── Set test environment BEFORE any app imports ──
os.environ['FLASK_ENV'] = 'testing'
os.environ['SQLALCHEMY_DATABASE_URI'] = 'sqlite://'  # in-memory
os.environ.setdefault('SECRET_KEY', 'testing_secret_key_1234567890')
os.environ.setdefault('SECURITY_PASSWORD_SALT', 'testing_salt_1234567890')

from app import app as flask_app  # noqa: E402
from extensions import db, limiter  # noqa: E402


@pytest.fixture(autouse=True)
def _set_flask_env():
    os.environ['FLASK_ENV'] = 'testing'


@pytest.fixture
def app():
    flask_app.config['TESTING'] = True
    flask_app.config['WTF_CSRF_ENABLED'] = False
    flask_app.config['RATELIMIT_ENABLED'] = False
    flask_app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite://'
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
