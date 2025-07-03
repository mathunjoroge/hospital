import sqlite3
from contextlib import contextmanager
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Dict, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from src.config import get_config
import logging

logger = logging.getLogger("HIMS-NLP")
HIMS_CONFIG = get_config()

# UMLS Database Setup with connection pooling
umls_engine = create_engine(
    HIMS_CONFIG["UMLS_DB_URL"],
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10
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

def setup_test_database() -> sqlite3.Connection:
    """Set up an in-memory SQLite database for testing."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE soap_notes (
            id INTEGER PRIMARY KEY,
            patient_id TEXT,
            symptoms TEXT,
            created_at TEXT,
            ai_analysis TEXT,
            ai_notes TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE symptoms (
            id INTEGER PRIMARY KEY,
            name TEXT,
            cui TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE diseases (
            id INTEGER PRIMARY KEY,
            name TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE disease_keywords (
            id INTEGER PRIMARY KEY,
            disease_id INTEGER,
            keyword TEXT,
            cui TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE disease_symptoms (
            disease_id INTEGER,
            symptom_id INTEGER
        )
    """)
    cursor.execute("""
        CREATE TABLE patterns (
            id INTEGER PRIMARY KEY,
            label TEXT,
            pattern TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE disease_management_plans (
            id INTEGER PRIMARY KEY,
            disease_id INTEGER,
            plan TEXT
        )
    """)
    
    cursor.execute("INSERT INTO soap_notes (id, patient_id, symptoms, created_at) VALUES (?, ?, ?, ?)",
                   (1, "PAT001", "fever, cough, chest pain", "2025-07-02T14:00:00"))
    cursor.execute("INSERT INTO symptoms (name, cui) VALUES (?, ?)", ("fever", "C0015967"))
    cursor.execute("INSERT INTO symptoms (name, cui) VALUES (?, ?)", ("cough", "C0010200"))
    cursor.execute("INSERT INTO diseases (name) VALUES (?)", ("pneumonia",))
    cursor.execute("INSERT INTO disease_keywords (disease_id, keyword, cui) VALUES (?, ?, ?)", (1, "pneumonia", "C0032285"))
    cursor.execute("INSERT INTO disease_management_plans (disease_id, plan) VALUES (?, ?)", (1, "Antibiotics, oxygen therapy"))
    conn.commit()
    return conn

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
                (ai_analysis_html, summary, note_id)
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