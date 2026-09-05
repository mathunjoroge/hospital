import os

import pytest


def test_secret_key_enforcement(monkeypatch):
    """Verify that importing app without SECRET_KEY in production mode raises RuntimeError."""

    # Store original env vars
    original_secret = os.environ.get('SECRET_KEY')
    original_env = os.environ.get('FLASK_ENV')

    # Prevent dotenv from reloading SECRET_KEY from .env file during import
    import dotenv
    monkeypatch.setattr(dotenv, 'load_dotenv', lambda *a, **k: None)

    # Unset SECRET_KEY and set FLASK_ENV to production
    if 'SECRET_KEY' in os.environ:
        del os.environ['SECRET_KEY']
    os.environ['FLASK_ENV'] = 'production'

    # Clean sys.modules to force a fresh import of app
    import sys
    if 'app' in sys.modules:
        del sys.modules['app']

    import importlib
    try:
        with pytest.raises(RuntimeError, match="SECRET_KEY environment variable must be set"):
            if 'app' in sys.modules:
                importlib.reload(sys.modules['app'])
            else:
                import app  # noqa: F401
    finally:
        # Restore env vars & sys.modules
        if original_secret:
            os.environ['SECRET_KEY'] = original_secret
        if original_env:
            os.environ['FLASK_ENV'] = original_env
        if 'app' in sys.modules:
            del sys.modules['app']
