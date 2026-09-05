"""
tests/test_encryption.py
─────────────────────────
Unit tests for Task 2.5: Encryption at Rest (EncryptedString & Fernet crypto)
"""

from sqlalchemy import Column, Integer

try:
    from extensions import db
except ImportError:
    from extensions import db

from departments.crypto import EncryptedString, decrypt_value, encrypt_value


# Test Model using EncryptedString
class EncryptedSecretModel(db.Model):
    __tablename__ = 'test_encrypted_secrets'
    id = Column(Integer, primary_key=True)
    secret_data = Column(EncryptedString(255), nullable=True)


class TestCryptoHelpers:
    def test_encrypt_decrypt_roundtrip(self):
        plain = "NationalID_12345678"
        cipher = encrypt_value(plain)
        assert cipher != plain
        assert cipher.startswith("enc_v1:")

        decrypted = decrypt_value(cipher)
        assert decrypted == plain

    def test_random_iv_produces_different_ciphertexts(self):
        plain = "Sensitive_Medical_Record"
        cipher1 = encrypt_value(plain)
        cipher2 = encrypt_value(plain)
        assert cipher1 != cipher2
        assert decrypt_value(cipher1) == plain
        assert decrypt_value(cipher2) == plain

    def test_legacy_plaintext_fallback(self):
        legacy_plain = "33445566"
        assert decrypt_value(legacy_plain) == "33445566"

    def test_none_value_handling(self):
        assert encrypt_value(None) is None
        assert decrypt_value(None) is None


class TestEncryptedStringTypeDecorator:
    def test_db_read_write_transparent_encryption(self, app):
        with app.app_context():
            db.create_all()

            # Insert record
            record = EncryptedSecretModel(id=1, secret_data="MySecretPasscode99")
            db.session.add(record)
            db.session.commit()

            # Read back through SQLAlchemy model
            fetched = EncryptedSecretModel.query.get(1)
            assert fetched.secret_data == "MySecretPasscode99"

            # Query raw SQL to verify data at rest in DB is encrypted
            raw_sql = db.session.execute(
                db.text("SELECT secret_data FROM test_encrypted_secrets WHERE id = 1")
            ).scalar()
            assert raw_sql.startswith("enc_v1:")
            assert "MySecretPasscode99" not in raw_sql
