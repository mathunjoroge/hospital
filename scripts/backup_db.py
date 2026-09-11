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
"""

import glob
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HMIS.Backup")

BACKUP_DIR = os.path.join(BASE_DIR, "backups")
MAX_BACKUPS = 30  # Retain last 30 backups


def _configured_database_uri() -> str:
    """Resolve the live database URI the same way config.py does."""
    return os.getenv(
        "SQLALCHEMY_DATABASE_URI",
        os.getenv("DATABASE_URL", "sqlite:///" + os.path.join(BASE_DIR, "instance", "dev.db")),
    )


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
        "-h", parsed.hostname or "localhost",
        "-p", str(parsed.port or 5432),
        "-U", parsed.username or "hospital",
        "-Fc",  # custom format: compressed, supports pg_restore -j parallel restore
        "-f", backup_path,
        (parsed.path or "/hospital_core").lstrip("/"),
    ]

    logger.info(f"Starting pg_dump backup -> {backup_path}")
    try:
        result = subprocess.run(  # nosec B603  # noqa: PLW1510
            cmd, env=pg_env, capture_output=True, text=True, timeout=1800
        )
        if result.returncode != 0:
            logger.error(f"pg_dump failed (exit {result.returncode}): {result.stderr.strip()}")
            if os.path.exists(backup_path):
                os.remove(backup_path)
            return None

        size_mb = round(os.path.getsize(backup_path) / (1024 * 1024), 2)
        logger.info(f"Backup created successfully: {backup_filename} ({size_mb} MB)")
        rotate_backups(pattern="hospital_backup_*.pgdump")
        return backup_path
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
            os.path.join(BASE_DIR, "hospital.db")
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

        # Rotate old backups
        rotate_backups()
        return backup_path
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
                logger.info(f"Rotated old backup: {os.path.basename(old_backup)}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not remove old backup {old_backup}: {e}")


if __name__ == "__main__":
    result = perform_backup()
    if result:
        sys.exit(0)
    else:
        sys.exit(1)
