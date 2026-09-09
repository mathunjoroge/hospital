"""
departments/crypto.py
──────────────────────
Task 2.5 — Data Encryption at Rest using Fernet (AES-128-CBC + HMAC)

Features:
  - Key management from ENCRYPTION_KEY environment variable / Flask config
  - EncryptedString SQLAlchemy TypeDecorator for column-level encryption
  - Automatic transparent encryption on WRITE and decryption on READ
  - Backward compatible: legacy unencrypted plaintext values fall back gracefully

Security requirement:
  ENCRYPTION_KEY must always be set explicitly. In production the app refuses
  to start without it (same pattern as SECRET_KEY). In testing the test suite
  sets it via conftest / environment before importing the app.
  The old DEV_FALLBACK_KEY has been removed — a known static key in the git
  history provides zero protection and silently corrupts production data when
  the env var is accidentally omitted.
"""

import base64
import logging
import os

try:
    from cryptography.fernet import Fernet, InvalidToken
except ImportError:
    Fernet = None
    InvalidToken = Exception

from flask import current_app
from sqlalchemy.types import String, TypeDecorator

logger = logging.getLogger(__name__)


def get_fernet_key() -> bytes:
    """
    Retrieve Fernet encryption key from Flask config or environment.

    Raises RuntimeError if the key is absent and FLASK_ENV != 'testing'.
    In testing the key is set by conftest.py via os.environ before import.
    """
    key = None
    try:
        key = current_app.config.get("ENCRYPTION_KEY")
    except RuntimeError:
        pass  # Outside app context — fall through to env var

    if not key:
        key = os.environ.get("ENCRYPTION_KEY")

    if not key:
        flask_env = os.environ.get("FLASK_ENV", "")
        if flask_env == "testing":
            # Tests that exercise encryption paths must set ENCRYPTION_KEY.
            # generate_key() is safe here because it's scoped to the test run.
            if Fernet:
                generated = Fernet.generate_key()
                logger.warning(
                    "ENCRYPTION_KEY not set in test environment — "
                    "generating ephemeral key for this test run. "
                    "Set ENCRYPTION_KEY in conftest.py to avoid this."
                )
                return generated
            # cryptography not installed — return a dummy bytes value for tests
            return b"0" * 44
        raise RuntimeError(
            "CRITICAL SECURITY ERROR: ENCRYPTION_KEY environment variable is not set. "
            "Patient data cannot be encrypted safely. "
            "Generate a key with: python3 -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\" "
            "and set it in your .env file."
        )

    if isinstance(key, str):
        key_bytes = key.encode("utf-8")
    else:
        key_bytes = key

    # Ensure key is a valid Fernet key (base64-url 32-byte)
    try:
        if Fernet:
            Fernet(key_bytes)
        return key_bytes
    except Exception:
        # If a raw 32-byte string was passed, base64-encode it
        return base64.urlsafe_b64encode(key_bytes.ljust(32)[:32])


def encrypt_value(value: str) -> str:
    """Encrypt plain string value into Fernet token string."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)

    if not Fernet:
        logger.warning("cryptography library not available, returning plaintext")
        return value

    key = get_fernet_key()
    f = Fernet(key)
    encrypted_bytes = f.encrypt(value.encode("utf-8"))
    return f"enc_v1:{encrypted_bytes.decode('utf-8')}"


def decrypt_value(token: str) -> str:
    """Decrypt Fernet token string back into plain string. Falls back to raw value if unencrypted."""
    if token is None:
        return None
    if not isinstance(token, str):
        return str(token)

    # Check if value has encryption prefix
    if not token.startswith("enc_v1:"):
        return token  # Legacy plaintext

    if not Fernet:
        return token

    raw_token = token[7:]  # Strip 'enc_v1:' prefix
    key = get_fernet_key()
    f = Fernet(key)

    try:
        decrypted_bytes = f.decrypt(raw_token.encode("utf-8"))
        return decrypted_bytes.decode("utf-8")
    except (InvalidToken, Exception) as e:
        logger.warning(f"Decryption failed for value: {e}")
        return token


class EncryptedString(TypeDecorator):
    """
    SQLAlchemy Column Type for field-level encryption at rest.
    Encrypts on bind_param (DB INSERT/UPDATE), decrypts on process_result_value (DB SELECT).
    """

    impl = String
    cache_ok = True

    def __init__(self, length=255, *args, **kwargs):
        super().__init__(length=length, *args, **kwargs)

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return encrypt_value(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return decrypt_value(value)
