# database.py
import sqlite3
import json
from typing import List, Dict

def fetch_soap_notes(limit: int = None) -> List[Dict]:
    """Fetch SOAP notes from SQLite database"""
    conn = sqlite3.connect(HIMS_CONFIG["SQLITE_DB_PATH"])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = "SELECT * FROM soap_notes"
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query)
    notes = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return notes

def update_ai_analysis(note_id: int, analysis: dict):
    """Update AI analysis field in database"""
    conn = sqlite3.connect(HIMS_CONFIG["SQLITE_DB_PATH"])
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE soap_notes SET ai_analysis = ? WHERE id = ?",
        (json.dumps(analysis), note_id)
    conn.commit()
    conn.close()