import logging
import os
import sqlite3
from contextlib import contextmanager
from typing import Dict, List, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from tenacity import retry, stop_after_attempt, wait_exponential

from departments.nlp.src.config import get_config

logger = logging.getLogger("HIMS-NLP")
HIMS_CONFIG = get_config()

# UMLS connect_timeout is in seconds and is passed straight through to libpq
# via psycopg2. NOTE: libpq silently floors connect_timeout at 2 seconds
# regardless of a lower configured value (documented PostgreSQL client
# library behavior) -- see the identical note in departments/shared/drugcentral.py.
UMLS_CONNECT_TIMEOUT = int(os.environ.get("UMLS_CONNECT_TIMEOUT", "3"))

# UMLS Database Setup with connection pooling.
#
# connect_args={"connect_timeout": ...} bounds how long a *new* connection
# attempt can hang if the UMLS Postgres host is slow or unreachable.
# pool_pre_ping issues a lightweight "is this connection still alive" check
# before handing out a pooled connection, so a connection that went stale
# (host restarted, network blip) is quietly recycled instead of raising a
# confusing error deep inside a query. pool_recycle forces connections to be
# refreshed periodically so none of them can go stale indefinitely under
# QueuePool. Without any of this, a slow/unreachable UMLS host hangs NLP
# term-lookup requests indefinitely -- the same failure mode the DrugCentral
# P0 fix addressed for drug-interaction lookups.
umls_engine = create_engine(
    HIMS_CONFIG["UMLS_DB_URL"],
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=1800,
    connect_args={"connect_timeout": UMLS_CONNECT_TIMEOUT},
)
UMLSSession = sessionmaker(bind=umls_engine)


@contextmanager
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_sqlite_connection():
    """Context manager for SQLite connections with retry logic."""
    try:
        conn = sqlite3.connect(HIMS_CONFIG["SQLITE_DB_PATH"])
        conn.row_factory = sqlite3.Row
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        conn.close()


def fetch_soap_notes(limit: int = None) -> List[Dict]:
    """Fetch SOAP notes from the database."""
    try:
        with get_sqlite_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM soap_notes"
            if limit:
                query += f" LIMIT {limit}"
            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        logger.error(f"Error fetching SOAP notes: {e}")
        return []


def fetch_single_soap_note(note_id: int) -> Optional[Dict]:
    """Fetch a single SOAP note by ID."""
    try:
        with get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM soap_notes WHERE id = ?", (note_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    except sqlite3.Error as e:
        logger.error(f"Error fetching SOAP note {note_id}: {e}")
        return None


def update_ai_analysis(note_id: int, ai_analysis_html: str, summary: str) -> bool:
    """Update AI analysis and summary for a specific note."""
    try:
        with get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM soap_notes WHERE id = ?", (note_id,))
            if not cursor.fetchone():
                logger.warning(f"Note ID {note_id} not found for AI analysis update")
                return False

            cursor.execute(
                "UPDATE soap_notes SET ai_analysis = ?, ai_notes = ? WHERE id = ?",
                (ai_analysis_html, summary, note_id),
            )
            conn.commit()
            logger.info(f"Updated AI analysis for note ID {note_id}")
            return True
    except sqlite3.Error as e:
        logger.error(f"Database error updating note {note_id}: {e}")
        return False
    except Exception as e:
        logger.critical(f"Unexpected error updating note {note_id}: {e}", exc_info=True)
        return False
