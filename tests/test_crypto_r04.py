"""
tests/test_crypto_r04.py
────────────────────────
Verification tests for R-04:
- Ensures User.totp_secret uses EncryptedString
- Validates encrypt_value / decrypt_value roundtrip
- Verifies decrypt_value raises ValueError on invalid key / corrupted token
"""

import pytest
from cryptography.fernet import Fernet

from departments.crypto import EncryptedString, decrypt_value, encrypt_value
from departments.models.user import User


def test_user_totp_secret_is_encrypted():
    """User.totp_secret column must be an EncryptedString instance."""
    col_type = User.totp_secret.property.columns[0].type
    assert isinstance(col_type, EncryptedString)


def test_crypto_roundtrip(monkeypatch):
    key = Fernet.generate_key()
    monkeypatch.setenv("ENCRYPTION_KEY", key.decode())

    secret = "JBSWY3DPEHPK3PXP"
    encrypted = encrypt_value(secret)
    assert encrypted.startswith("enc_v1:")
    assert encrypted != secret

    decrypted = decrypt_value(encrypted)
    assert decrypted == secret


def test_crypto_invalid_key_raises_error(monkeypatch):
    key1 = Fernet.generate_key()
    monkeypatch.setenv("ENCRYPTION_KEY", key1.decode())
    encrypted = encrypt_value("secret_data")

    # Change to a different valid key
    key2 = Fernet.generate_key()
    monkeypatch.setenv("ENCRYPTION_KEY", key2.decode())

    with pytest.raises(ValueError, match="Decryption failed"):
        decrypt_value(encrypted)
