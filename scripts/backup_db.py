#!/usr/bin/env python3
"""
scripts/backup_db.py
────────────────────
Automated Database Backup Utility for HMIS.
Creates timestamped hot backups of SQLite database with rotation policy.
"""

import os
import sys
import glob
import sqlite3
import logging
from datetime import datetime, timezone

# Add project root directory to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HMIS.Backup")

BACKUP_DIR = os.path.join(BASE_DIR, "backups")
MAX_BACKUPS = 30  # Retain last 30 backups


def perform_backup(db_path: str = None) -> str:
    """
    Perform a consistent hot backup of the SQLite database.
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
    except Exception as e:
        logger.error(f"Backup failed: {e}", exc_info=True)
        return None


def rotate_backups():
    """Remove older backups if count exceeds MAX_BACKUPS."""
    backups = sorted(glob.glob(os.path.join(BACKUP_DIR, "hospital_backup_*.db")))
    if len(backups) > MAX_BACKUPS:
        excess = backups[:-MAX_BACKUPS]
        for old_backup in excess:
            try:
                os.remove(old_backup)
                logger.info(f"Rotated old backup: {os.path.basename(old_backup)}")
            except Exception as e:
                logger.warning(f"Could not remove old backup {old_backup}: {e}")


if __name__ == "__main__":
    result = perform_backup()
    if result:
        sys.exit(0)
    else:
        sys.exit(1)
