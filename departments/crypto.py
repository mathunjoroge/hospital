"""
departments/crypto.py
──────────────────────
Task 2.5 — Data Encryption at Rest using Fernet (AES-128-CBC + HMAC)

Features:
  - Key management from ENCRYPTION_KEY environment variable / Flask config
  - EncryptedString SQLAlchemy TypeDecorator for column-level encryption
  - Automatic transparent encryption on WRITE and decryption on READ
  - Backward compatible: legacy unencrypted plaintext values fall back gracefully
"""

import base64
import logging
import os

from cryptography.fernet import Fernet, InvalidToken
from flask import current_app
from sqlalchemy.types import String, TypeDecorator

logger = logging.getLogger(__name__)

# Fallback deterministic key for development/testing if ENCRYPTION_KEY is not set
DEV_FALLBACK_KEY = b'u8N_706K8i-8K2182K_X904L981L76K543210123456='


def get_fernet_key() -> bytes:
    """Retrieve Fernet encryption key from Flask config or environment."""
    key = None
    try:
        key = current_app.config.get('ENCRYPTION_KEY')
    except RuntimeError:
        pass  # Outside app context

    if not key:
        key = os.environ.get('ENCRYPTION_KEY')

    if not key:
        return DEV_FALLBACK_KEY

    if isinstance(key, str):
        key_bytes = key.encode('utf-8')
    else:
        key_bytes = key

    # Ensure key is valid 32-byte base64 URL-safe key
    try:
        Fernet(key_bytes)
        return key_bytes
    except Exception:
        # If raw 32-byte string was passed, base64 encode it
        return base64.urlsafe_b64encode(key_bytes.ljust(32)[:32])


def encrypt_value(value: str) -> str:
    """Encrypt plain string value into Fernet token string."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)

    key = get_fernet_key()
    f = Fernet(key)
    encrypted_bytes = f.encrypt(value.encode('utf-8'))
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

    raw_token = token[7:]  # Strip 'enc_v1:' prefix
    key = get_fernet_key()
    f = Fernet(key)

    try:
        decrypted_bytes = f.decrypt(raw_token.encode('utf-8'))
        return decrypted_bytes.decode('utf-8')
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
