import os
import pytest
from flask import Flask

def test_secret_key_enforcement():
    """Verify that importing app without SECRET_KEY and FLASK_ENV!=testing raises RuntimeError."""
    
    # Store original env vars
    original_secret = os.environ.get('SECRET_KEY')
    original_env = os.environ.get('FLASK_ENV')
    
    # Unset SECRET_KEY and set FLASK_ENV to something other than testing
    if 'SECRET_KEY' in os.environ:
        del os.environ['SECRET_KEY']
    os.environ['FLASK_ENV'] = 'production'
    
    # Clean sys.modules to force a fresh import of app
    import sys
    if 'app' in sys.modules:
        del sys.modules['app']
        
    with pytest.raises(RuntimeError, match="SECRET_KEY environment variable must be set."):
        import app
        
    # Restore env vars
    if original_secret:
        os.environ['SECRET_KEY'] = original_secret
    if original_env:
        os.environ['FLASK_ENV'] = original_env
    
    # Restore app import for other tests
    if 'app' in sys.modules:
        del sys.modules['app']
