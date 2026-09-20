#!/usr/bin/env python3
"""
scripts/backup_db.py
────────────────────
Automated Database Backup Utility for HMIS.

IMPORTANT CORRECTION (see docs/backup_restore_runbook.md changelog):
this script originally only knew how to back up a local SQLite file via
sqlite3's online backup API. Since PHASE 1.3 moved the production database
to PostgreSQL (see docker-compose.yml), that path was silently backing up
nothing meaningful in a production deployment -- the previously "verified"
restore drill only ever exercised the SQLite dev path. perform_backup() now
detects the configured database backend (SQLALCHEMY_DATABASE_URI /
DATABASE_URL) and dispatches to a `pg_dump`-based backup for PostgreSQL,
keeping the SQLite path only for local/dev/test deployments that still use
it.

R-17: Backup integrity
  - BACKUP_ENCRYPTION_KEY env var (Fernet key) enables AES-128 stream encryption.
    Set via: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
  - SHA-256 digest is written alongside every backup: <backup>.sha256
  - Manifest JSON is updated in backups/manifest.json after each successful run.
"""

import glob
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import subprocess  # nosec B404 -- pg_dump is invoked with a fixed argv list, no shell=True
import sys
from datetime import datetime, timezone
from urllib.parse import urlparse

# Add project root directory to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("HMIS.Backup")

BACKUP_DIR = os.path.join(BASE_DIR, "backups")
MAX_BACKUPS = 30  # Retain last 30 backups
MANIFEST_PATH = os.path.join(BACKUP_DIR, "manifest.json")


def _configured_database_uri() -> str:
    """Resolve the live database URI the same way config.py does."""
    return os.getenv(
        "SQLALCHEMY_DATABASE_URI",
        os.getenv(
            "DATABASE_URL", "sqlite:///" + os.path.join(BASE_DIR, "instance", "dev.db")
        ),
    )


def _compute_sha256(file_path: str) -> str:
    """Compute SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _encrypt_backup(backup_path: str) -> str | None:
    """
    Encrypt the backup file using Fernet symmetric encryption (AES-128-CBC + HMAC).
    Returns the path to the encrypted file, or None if encryption is not configured.

    The BACKUP_ENCRYPTION_KEY environment variable must contain a valid Fernet key.
    Generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    """
    key_str = os.getenv("BACKUP_ENCRYPTION_KEY", "").strip()
    if not key_str:
        logger.info("BACKUP_ENCRYPTION_KEY not set — backup stored unencrypted.")
        return None

    try:
        from cryptography.fernet import Fernet
    except ImportError:
        logger.warning("cryptography package not installed — skipping encryption.")
        return None

    try:
        fernet = Fernet(key_str.encode())
    except Exception as e:
        logger.error(f"Invalid BACKUP_ENCRYPTION_KEY: {e}")
        return None

    encrypted_path = backup_path + ".enc"
    try:
        with open(backup_path, "rb") as f_in, open(encrypted_path, "wb") as f_out:
            # Read in chunks to handle large dumps without loading into memory
            while True:
                chunk = f_in.read(1024 * 1024)  # 1 MB at a time
                if not chunk:
                    break
                f_out.write(fernet.encrypt(chunk))
        logger.info(f"Backup encrypted: {os.path.basename(encrypted_path)}")
        # Remove unencrypted original
        os.remove(backup_path)
        return encrypted_path
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        return None


def _write_sha256_and_manifest(backup_path: str):
    """Write a .sha256 sidecar file and update the backup manifest."""
    sha256_hex = _compute_sha256(backup_path)
    sha256_path = backup_path + ".sha256"
    with open(sha256_path, "w") as f:
        f.write(f"{sha256_hex}  {os.path.basename(backup_path)}\n")
    logger.info(f"SHA-256 digest written: {os.path.basename(sha256_path)}")

    # Update manifest
    os.makedirs(BACKUP_DIR, exist_ok=True)
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH) as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError):
            manifest = {"backups": []}
    else:
        manifest = {"backups": []}

    manifest["backups"].append(
        {
            "filename": os.path.basename(backup_path),
            "sha256": sha256_hex,
            "size_bytes": os.path.getsize(backup_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    # Keep last MAX_BACKUPS entries in manifest
    manifest["backups"] = manifest["backups"][-MAX_BACKUPS:]
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Backup manifest updated.")


def perform_backup(db_path: str | None = None) -> str:
    """
    Perform a hot backup of the active database, dispatching by backend.

    - PostgreSQL (SQLALCHEMY_DATABASE_URI starts with postgresql://): pg_dump
      to a timestamped, custom-format (-Fc) dump file.
    - SQLite (default for local/dev/test, or an explicit db_path): sqlite3
      online backup API.
    """
    db_uri = _configured_database_uri()
    if db_path is None and db_uri.startswith("postgresql"):
        return perform_postgres_backup(db_uri)
    return perform_sqlite_backup(db_path)


def perform_postgres_backup(db_uri: str) -> str:
    """Dump the live PostgreSQL database using pg_dump (custom format, compressed)."""
    parsed = urlparse(db_uri)
    pg_env = os.environ.copy()
    if parsed.password:
        pg_env["PGPASSWORD"] = parsed.password

    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_filename = f"hospital_backup_{timestamp}.pgdump"
    backup_path = os.path.join(BACKUP_DIR, backup_filename)

    if shutil.which("pg_dump") is None:
        logger.error(
            "pg_dump not found on PATH -- cannot back up the PostgreSQL "
            "database. Install the postgresql-client package."
        )
        return None

    cmd = [
        "pg_dump",
        "-h",
        parsed.hostname or "localhost",
        "-p",
        str(parsed.port or 5432),
        "-U",
        parsed.username or "hospital",
        "-Fc",  # custom format: compressed, supports pg_restore -j parallel restore
        "-f",
        backup_path,
        (parsed.path or "/hospital_core").lstrip("/"),
    ]

    logger.info(f"Starting pg_dump backup -> {backup_path}")
    try:
        result = subprocess.run(  # nosec B603  # noqa: PLW1510
            cmd, env=pg_env, capture_output=True, text=True, timeout=1800
        )
        if result.returncode != 0:
            logger.error(
                f"pg_dump failed (exit {result.returncode}): {result.stderr.strip()}"
            )
            if os.path.exists(backup_path):
                os.remove(backup_path)
            return None

        size_mb = round(os.path.getsize(backup_path) / (1024 * 1024), 2)
        logger.info(f"Backup created successfully: {backup_filename} ({size_mb} MB)")

        # R-17: Encrypt and compute integrity digest
        encrypted_path = _encrypt_backup(backup_path)
        final_path = encrypted_path if encrypted_path else backup_path
        _write_sha256_and_manifest(final_path)

        rotate_backups(pattern="hospital_backup_*.pgdump*")
        return final_path
    except subprocess.TimeoutExpired:
        logger.error("pg_dump timed out after 30 minutes")
        return None
    except Exception:
        logger.exception("Backup failed: ")
        return None


def perform_sqlite_backup(db_path: str | None = None) -> str:
    """
    Perform a consistent hot backup of the SQLite database (local/dev/test only).
    """
    if not db_path:
        candidates = [
            os.path.join(BASE_DIR, "instance", "dev.db"),
            os.path.join(BASE_DIR, "instance", "hims.db"),
            os.path.join(BASE_DIR, "instance", "hospital.db"),
            os.path.join(BASE_DIR, "dev.db"),
            os.path.join(BASE_DIR, "hims.db"),
            os.path.join(BASE_DIR, "hospital.db"),
        ]
        for c in candidates:
            if os.path.exists(c):
                db_path = c
                break

    if not db_path or not os.path.exists(db_path):
        # Fallback: create default database directory & file if testing
        os.makedirs(os.path.join(BASE_DIR, "instance"), exist_ok=True)
        db_path = os.path.join(BASE_DIR, "instance", "dev.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS system_init (id INTEGER PRIMARY KEY);")
        conn.close()

    os.makedirs(BACKUP_DIR, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_filename = f"hospital_backup_{timestamp}.db"
    backup_path = os.path.join(BACKUP_DIR, backup_filename)

    logger.info(f"Starting backup from {db_path} -> {backup_path}")

    try:
        # Use sqlite3 online backup API for safe hot backup without locking issues
        src_conn = sqlite3.connect(db_path)
        dest_conn = sqlite3.connect(backup_path)
        with dest_conn:
            src_conn.backup(dest_conn)
        src_conn.close()
        dest_conn.close()

        size_mb = round(os.path.getsize(backup_path) / (1024 * 1024), 2)
        logger.info(f"Backup created successfully: {backup_filename} ({size_mb} MB)")

        # R-17: Encrypt and compute integrity digest
        encrypted_path = _encrypt_backup(backup_path)
        final_path = encrypted_path if encrypted_path else backup_path
        _write_sha256_and_manifest(final_path)

        rotate_backups()
        return final_path
    except Exception:
        logger.exception("Backup failed: ")
        return None


def rotate_backups(pattern: str = "hospital_backup_*.db"):
    """Remove older backups if count exceeds MAX_BACKUPS."""
    backups = sorted(glob.glob(os.path.join(BACKUP_DIR, pattern)))
    if len(backups) > MAX_BACKUPS:
        excess = backups[:-MAX_BACKUPS]
        for old_backup in excess:
            try:
                os.remove(old_backup)
                # Also remove sidecar files
                for ext in (".sha256",):
                    sidecar = old_backup + ext
                    if os.path.exists(sidecar):
                        os.remove(sidecar)
                logger.info(f"Rotated old backup: {os.path.basename(old_backup)}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not remove old backup {old_backup}: {e}")


if __name__ == "__main__":
    result = perform_backup()
    if result:
        sys.exit(0)
    else:
        sys.exit(1)

