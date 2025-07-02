import logging
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import sqlite3
import re
import time
import json
import os
import argparse
import pickle
from fastapi import FastAPI, HTTPException, Response, requests
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from contextlib import contextmanager
from dotenv import load_dotenv
from datetime import datetime
import uvicorn
import multiprocessing
import unittest
from fastapi.testclient import TestClient
import signal
import sys
import html

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nlp_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HIMS-NLP")

# Load environment variables
load_dotenv()

# Configuration
HIMS_CONFIG = {
    "DEFAULT_DEPARTMENT": "emergency",
    "PRIORITY_SYMPTOMS": ["chest pain", "shortness of breath", "severe headache"],
    "UMLS_THRESHOLD": 0.7,
    "SQLITE_DB_PATH": "/home/mathu/projects/hospital/instance/hims.db",
    "API_HOST": "0.0.0.0",
    "API_PORT": 8000,
    "BATCH_SIZE": 50
}

console = Console()

# Mock database setup for testing
def setup_test_database():
    """Set up an in-memory SQLite database for testing."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Create necessary tables
    cursor.execute("""
        CREATE TABLE soap_notes (
            id INTEGER PRIMARY KEY,
            patient_id TEXT,
            situation TEXT,
            hpi TEXT,
            symptoms TEXT,
            aggravating_factors TEXT,
            alleviating_factors TEXT,
            medical_history TEXT,
            medication_history TEXT,
            assessment TEXT,
            recommendation TEXT,
            additional_notes TEXT,
            ai_notes TEXT,
            ai_analysis TEXT,
            created_at TEXT
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
    
    # Insert sample data
    cursor.execute("INSERT INTO soap_notes (id, patient_id, symptoms, created_at) VALUES (?, ?, ?, ?)",
                   (1, "PAT001", "fever, cough, chest pain", "2025-07-02T14:00:00"))
    cursor.execute("INSERT INTO symptoms (name, cui) VALUES (?, ?)", ("fever", "C0015967"))
    cursor.execute("INSERT INTO symptoms (name, cui) VALUES (?, ?)", ("cough", "C0010200"))
    cursor.execute("INSERT INTO diseases (name) VALUES (?)", ("pneumonia",))
    cursor.execute("INSERT INTO disease_keywords (disease_id, keyword, cui) VALUES (?, ?, ?)", (1, "pneumonia", "C0032285"))
    cursor.execute("INSERT INTO disease_management_plans (disease_id, plan) VALUES (?, ?)", (1, "Antibiotics, oxygen therapy"))
    
    conn.commit()
    return conn

@contextmanager
def get_sqlite_connection():
    """Context manager for SQLite connections to existing hims.db"""
    try:
        conn = sqlite3.connect(HIMS_CONFIG["SQLITE_DB_PATH"])
        conn.row_factory = sqlite3.Row
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        conn.close()

# HTML Response Generator
# HTML Response Generator
def generate_html_response(data: Dict[str, any], status_code: int = 200) -> str:
    """
    Generate an HTML response for the /process_note endpoint.
    
    Args:
        data (Dict[str, any]): The response data (success or error)
        status_code (int): HTTP status code (200 for success, 404 for not found)
    
    Returns:
        str: Formatted HTML response
    """
    if status_code == 404:
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Note Not Found</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                    color: #343a40;
                }}
                .container {{ 
                    max-width: 800px;
                    margin: 40px auto;
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
                    text-align: center;
                }}
                h1 {{ 
                    color: #d32f2f;
                    border-bottom: 2px solid #ffcdd2;
                    padding-bottom: 15px;
                }}
                p {{
                    font-size: 18px;
                    color: #555;
                    line-height: 1.6;
                }}
                .error-icon {{
                    font-size: 48px;
                    color: #d32f2f;
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="error-icon">❌</div>
                <h1>Error: Note Not Found</h1>
                <p>{html.escape(data.get('detail', 'The requested note was not found in the database.'))}</p>
            </div>
        </body>
        </html>
        """

    # Success case: format the data into HTML
    note_id = html.escape(str(data.get("note_id", "Unknown")))
    patient_id = html.escape(data.get("patient_id", "Unknown"))
    primary_diagnosis = data.get("primary_diagnosis", {})
    differential_diagnoses = data.get("differential_diagnoses", [])
    keywords = [html.escape(k) for k in data.get("keywords", [])]
    cuis = [html.escape(c) for c in data.get("cuis", [])]
    entities = data.get("entities", [])
    summary = html.escape(data.get("summary", ""))
    management_plans = {html.escape(k): html.escape(v) for k, v in data.get("management_plans", {}).items()}
    processed_at = html.escape(data.get("processed_at", datetime.now().isoformat()))
    processing_time = data.get("processing_time", 0.0)

    # Generate HTML content sections
    primary_diagnosis_html = ""
    if primary_diagnosis:
        primary_diagnosis_html = f"""
        <div class="section diagnosis">
            <h2>Primary Diagnosis</h2>
            <div class="info-card">
                <strong>Disease:</strong> {html.escape(primary_diagnosis.get("disease", "N/A"))}
                <span class="badge badge-success">Score: {primary_diagnosis.get("score", "N/A")}</span>
            </div>
        </div>
        """
    else:
        primary_diagnosis_html = """
        <div class="section">
            <h2>Primary Diagnosis</h2>
            <p>No primary diagnosis identified</p>
        </div>
        """

    differential_diagnoses_html = ""
    if differential_diagnoses:
        table_rows = "".join(
            f'<tr><td>{html.escape(diag["disease"])}</td><td>{diag["score"]}</td></tr>'
            for diag in differential_diagnoses
        )
        differential_diagnoses_html = f"""
        <div class="section diagnosis">
            <h2>Differential Diagnoses</h2>
            <table>
                <thead>
                    <tr><th>Disease</th><th>Score</th></tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        """

    entities_html = ""
    if entities:
        entity_items = "".join(
            f'<div class="entity"><strong>{html.escape(entity[0])}</strong> '
            f'<span class="badge badge-primary">{html.escape(entity[1])}</span>'
            f'<div><small>Severity: {entity[2]["severity"]}, Temporal: {html.escape(entity[2]["temporal"])}</small></div></div>'
            for entity in entities
        )
        entities_html = f"""
        <div class="section entities">
            <h2>Entities</h2>
            <div>
                {entity_items}
            </div>
        </div>
        """

    management_plans_html = ""
    if management_plans:
        plans_html = "".join(
            f'<div class="management-plan"><strong>{disease}:</strong> {plan}</div>'
            for disease, plan in management_plans.items()
        )
        management_plans_html = f"""
        <div class="section management">
            <h2>Management Plans</h2>
            {plans_html}
        </div>
        """

    # Generate the complete HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Clinical Note Analysis - Note ID {note_id}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
                color: #343a40;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            }}
            h1 {{
                color: #0d6efd;
                border-bottom: 2px solid #e9ecef;
                padding-bottom: 15px;
                margin-bottom: 25px;
            }}
            h2 {{
                color: #495057;
                margin-top: 25px;
                margin-bottom: 15px;
                font-size: 1.4rem;
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                border-radius: 8px;
                background-color: #f8f9fa;
                border-left: 4px solid #0d6efd;
            }}
            .summary {{
                background-color: #e7f5ff;
                border-left-color: #0d6efd;
            }}
            .diagnosis {{
                background-color: #e6fcf5;
                border-left-color: #20c997;
            }}
            .management {{
                background-color: #fff3bf;
                border-left-color: #ffd43b;
            }}
            .entities {{
                background-color: #ffe8e8;
                border-left-color: #fa5252;
            }}
            .info-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }}
            .info-card {{
                background: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }}
            th, td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #e9ecef;
            }}
            th {{
                background-color: #f1f3f5;
                font-weight: 600;
            }}
            tr:hover {{
                background-color: #f8f9fa;
            }}
            ul {{
                list-style-type: none;
                padding: 0;
                margin: 0;
            }}
            li {{
                padding: 8px 12px;
                margin-bottom: 8px;
                background: white;
                border-radius: 6px;
                display: inline-block;
                margin-right: 10px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .entity {{
                padding: 10px;
                margin-bottom: 10px;
                border-radius: 6px;
                background: white;
            }}
            .management-plan {{
                padding: 15px;
                margin-bottom: 15px;
                border-radius: 8px;
                background: white;
                border-left: 4px solid #ffd43b;
            }}
            .badge {{
                display: inline-block;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: 500;
                margin-left: 8px;
            }}
            .badge-primary {{
                background-color: #d0ebff;
                color: #1971c2;
            }}
            .badge-success {{
                background-color: #d3f9d8;
                color: #2f9e44;
            }}
            .badge-warning {{
                background-color: #fff3bf;
                color: #e67700;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Clinical Note Analysis - Note ID {note_id}</h1>
            
            <div class="info-grid">
                <div class="info-card">
                    <strong>Patient ID:</strong> {patient_id}
                </div>
                <div class="info-card">
                    <strong>Processed At:</strong> {processed_at}
                </div>
                <div class="info-card">
                    <strong>Processing Time:</strong> {processing_time:.3f} seconds
                </div>
            </div>

            {primary_diagnosis_html}
            {differential_diagnoses_html}

            <div class="section">
                <h2>Keywords</h2>
                <ul>
                    {"".join(f'<li>{keyword}</li>' for keyword in keywords)}
                </ul>
            </div>

            <div class="section">
                <h2>CUIs</h2>
                <ul>
                    {"".join(f'<li>{cui}</li>' for cui in cuis)}
                </ul>
            </div>

            {entities_html}
            {management_plans_html}

            <div class="section summary">
                <h2>Summary</h2>
                <p>{summary}</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

# SOAP Note Processing Functions
def fetch_soap_notes(limit: int = None) -> List[Dict]:
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
    try:
        with get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM soap_notes WHERE id = ?", (note_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    except sqlite3.Error as e:
        logger.error(f"Error fetching SOAP note {note_id}: {e}")
        return None

def prepare_note_for_nlp(soap_note: Dict) -> str:
    fields = [
        soap_note.get('situation', ''),
        soap_note.get('hpi', ''),
        soap_note.get('symptoms', ''),
        soap_note.get('aggravating_factors', ''),
        soap_note.get('alleviating_factors', ''),
        soap_note.get('medical_history', ''),
        soap_note.get('medication_history', ''),
        soap_note.get('assessment', ''),
        soap_note.get('recommendation', ''),
        soap_note.get('additional_notes', ''),
        soap_note.get('ai_notes', '')
    ]
    return ' '.join(f for f in fields if f).strip()

def generate_summary(text: str, max_sentences: int = 3, doc=None) -> str:
    start_time = time.time()
    if not doc:
        doc = DiseasePredictor.nlp(text)
    sentence_scores = []
    seen_sentences = set()
    clinical_terms = DiseasePredictor.clinical_terms
    
    for i, sent in enumerate(doc.sents):
        sent_text = str(sent).strip()
        if sent_text in seen_sentences:
            continue
        seen_sentences.add(sent_text)
        score = sum(3 for token in sent if token.text.lower() in clinical_terms)
        score += len([token for token in sent if token.pos_ in ["VERB", "AUX"]])
        score += 1 - (i / max(len(list(doc.sents)), 1))
        sentence_scores.append((sent_text, score))
    
    sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
    top_sentences = []
    seen_content = set()
    for sent_text, _ in sorted_sentences:
        if not any(sent_text.lower() in seen or seen in sent_text.lower() for seen in seen_content):
            top_sentences.append(sent_text)
            seen_content.add(sent_text.lower())
        if len(top_sentences) >= max_sentences:
            break
    
    summary = " ".join(top_sentences)
    missing_symptoms = [symptom for symptom in clinical_terms if symptom in text.lower() and symptom not in summary.lower()]
    if missing_symptoms:
        summary += f" Additional symptoms include {', '.join(missing_symptoms)}."
    
    logger.debug(f"Summary generation took {time.time() - start_time:.3f} seconds")
    return summary if summary else text[:200] + "..."

def update_ai_analysis(note_id: int, analysis: Dict, summary: str) -> bool:
    """
    Update AI analysis and summary for a specific note in the soap_notes table.
    
    Args:
        note_id (int): The ID of the note to update
        analysis (Dict): The AI analysis data to store
        summary (str): The summary text to store
    
    Returns:
        bool: True if update successful, False if note not found
    """
    try:
        with get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM soap_notes WHERE id = ?", (note_id,))
            if not cursor.fetchone():
                logger.warning(f"Note ID {note_id} not found for AI analysis update")
                return False
            
            cursor.execute("PRAGMA synchronous = NORMAL")
            cursor.execute(
                "UPDATE soap_notes SET ai_analysis = ?, ai_notes = ? WHERE id = ?",
                (json.dumps(analysis, ensure_ascii=False), summary, note_id)
            )
            conn.commit()
            logger.info(f"Updated AI analysis and summary for note ID {note_id}")
            return True
    except sqlite3.Error as e:
        logger.error(f"Error updating AI analysis for note ID {note_id}: {e}")
        return False

# Enhanced NLP Components
class ClinicalNER:
    @classmethod
    def initialize(cls):
        if DiseasePredictor.clinical_terms is None:
            DiseasePredictor.clinical_terms = cls._load_clinical_terms()
    
    @staticmethod
    def _load_clinical_terms():
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM symptoms UNION SELECT keyword FROM disease_keywords")
                return {row['name'].lower() for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Error loading clinical terms: {e}")
            return {
                "headache", "fever", "cough", "pain", "nausea", 
                "vomiting", "dizziness", "rash", "fatigue", "chest pain",
                "shortness of breath", "abdominal pain", "diarrhea", "back pain",
                "chills", "loss of appetite", "jaundice"
            }

    def __init__(self):
        self.nlp = DiseasePredictor.nlp
        self.negation_terms = {"no", "not", "denies", "without", "absent", "negative"}
        self.lemmatizer = WordNetLemmatizer()
        self.patterns = self._load_patterns()
        self.compiled_patterns = [(label, re.compile(pattern, re.IGNORECASE)) for label, pattern in self.patterns]
    
    def _load_patterns(self):
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT label, pattern FROM patterns")
                return [(row['label'], row['pattern']) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
            return [
                ("PAIN", r"\b(pain|ache|discomfort|tenderness)\b"),
                ("FEVER", r"\b(fever|pyrexia|hyperthermia)\b"),
                ("RESPIRATORY", r"\b(cough|dyspnea|shortness of breath|wheez)\b"),
                ("CARDIO", r"\b(chest pain|palpitation|tachycardia)\b"),
                ("GASTRO", r"\b(nausea|vomiting|diarrhea|constipation|abdominal pain)\b"),
                ("NEURO", r"\b(headache|dizziness|vertigo|confusion|seizure)\b"),
                ("BACK_PAIN", r"\b(back pain|backpain|lumbago)\b"),
                ("CHILLS", r"\b(chills|shivering)\b"),
                ("APPETITE_LOSS", r"\b(loss of appetite|anorexia)\b"),
                ("JAUNDICE", r"\b(jaundice|yellowing)\b")
            ]
    
    def extract_entities(self, text: str, doc=None) -> List[Tuple[str, str, dict]]:
        start_time = time.time()
        if not doc:
            doc = self.nlp(text)
        entities = []
        seen_entities = set()
        
        temporal_patterns = [
            (re.compile(r"\b(\d+\s*(day|days|week|weeks|month|months|year|years)\s*(ago)?)\b", re.IGNORECASE), "DURATION"),
            (re.compile(r"\b(since|for)\s*(\d+\s*(day|days|week|weeks|month|months|year|years))\b", re.IGNORECASE), "DURATION"),
            (re.compile(r"\b\d+\s*days?\b", re.IGNORECASE), "DURATION"),
            (re.compile(r"\bfor\s*(three|four|five|six|seven|eight|nine|ten|[1-9]\d*)\s*days?\b", re.IGNORECASE), "DURATION")
        ]
        temporal_matches = []
        for pattern, label in temporal_patterns:
            for match in pattern.finditer(text):
                temporal_matches.append((match.group(), label))
        
        clinical_terms = DiseasePredictor.clinical_terms
        for term in clinical_terms:
            if term in text.lower() and term not in seen_entities:
                context = {"severity": 1, "temporal": "UNSPECIFIED"}
                for temp_text, temp_label in temporal_matches:
                    if temp_text.lower() in text.lower():
                        context["temporal"] = temp_text.lower()
                        break
                entities.append((term, "CLINICAL_TERM", context))
                seen_entities.add(term)
        
        for label, pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
                match_text = match.group().lower()
                if match_text not in seen_entities and not any(match_text in seen for seen in seen_entities):
                    context = {"severity": 1, "temporal": "UNSPECIFIED"}
                    for temp_text, temp_label in temporal_matches:
                        if temp_text.lower() in text.lower():
                            context["temporal"] = temp_text.lower()
                            break
                    entities.append((match.group(), label, context))
                    seen_entities.add(match_text)
        
        logger.debug(f"Entity extraction took {time.time() - start_time:.3f} seconds")
        return entities
    
    def extract_keywords_and_cuis(self, note: Dict) -> Tuple[List[str], List[str]]:
        start_time = time.time()
        text = ' '.join(filter(None, [
            note.get('situation', ''),
            note.get('hpi', ''),
            note.get('symptoms', ''),
            note.get('assessment', '')
        ]))
        if not text:
            return [], []
        
        entities = self.extract_entities(text)
        terms = {ent[0].lower() for ent in entities if ent}
        
        expected_keywords = []
        reference_cuis = []
        
        try:
            disease_keywords = DiseasePredictor.disease_keywords
            symptom_cuis = DiseasePredictor.symptom_cuis
            for term in terms:
                for keyword, cui in disease_keywords.items():
                    if keyword in term or term in keyword:
                        if keyword not in expected_keywords:
                            expected_keywords.append(keyword)
                            reference_cuis.append(cui)
                        break
                if term in symptom_cuis and term not in expected_keywords:
                    expected_keywords.append(term)
                    if symptom_cuis[term]:
                        reference_cuis.append(symptom_cuis[term])
        except Exception as e:
            logger.error(f"Error fetching keywords from database: {e}")
            disease_keywords = {
                "malaria": "C0024530",
                "pneumonia": "C0032285",
                "meningitis": "C0025289",
                "uti": "C0033578",
                "urinary tract infection": "C0033578",
                "influenza": "C0021400",
                "tuberculosis": "C0041296",
                "gastroenteritis": "C0017160",
                "dengue": "C0011311",
                "cholera": "C0008344",
                "bronchitis": "C0006277",
                "hepatitis": "C0019158",
                "viral hepatitis": "C0019158",
                "asthma": "C0004096",
                "myocardial infarction": "C0027051",
                "stroke": "C0038454",
                "diabetes": "C0011849",
                "hypertension": "C0020538",
                "back pain": "C0004604",
                "musculoskeletal back pain": "C0026857",
                "typhoid": "C0041466"
            }
            for term in terms:
                for keyword, cui in disease_keywords.items():
                    if keyword in term or term in keyword:
                        if keyword not in expected_keywords:
                            expected_keywords.append(keyword)
                            reference_cuis.append(cui)
                        break
        
        logger.debug(f"Keyword and CUI extraction took {time.time() - start_time:.3f} seconds")
        return list(set(expected_keywords)), list(set(reference_cuis))

class UMLSMapper:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UMLSMapper, cls).__new__(cls)
            cls._instance.cache_file = ":memory:"
            cls._instance.cui_cache = {}
            cls._instance.term_cache = {}
            cls._instance.semantic_cache = {}
            cls._instance._cache_modified = False
            cls._instance._load_cache()
            common_terms = [
                "headache", "fever", "chills", "nausea", "vomiting",
                "loss of appetite", "jaundice", "malaria", "gastroenteritis",
                "viral hepatitis"
            ]
            cls._instance.map_terms_to_cuis(common_terms)
        return cls._instance
    
    def _load_cache(self):
        pass  # In-memory cache for testing
    
    def save_cache(self):
        pass  # No persistent cache in test mode
    
    def map_term_to_cui(self, term: str) -> List[str]:
        term = term.lower()
        if term in self.term_cache:
            return self.term_cache[term]
        
        cuis = []  # Mock CUIs for testing
        self.term_cache[term] = cuis
        return cuis
    
    def map_terms_to_cuis(self, terms: List[str]) -> Dict[str, List[str]]:
        start_time = time.time()
        terms = [t.lower() for t in terms]
        results = {t: [] for t in terms}  # Mock results for testing
        logger.debug(f"UMLS mapping took {time.time() - start_time:.3f} seconds")
        return results
    
    def resolve_cui(self, cui: str) -> str:
        return cui  # Mock for testing
    
    def is_infectious_disease(self, cui: str) -> bool:
        return False  # Mock for testing

class DiseasePredictor:
    nlp = None
    clinical_terms = None
    disease_signatures = None
    disease_keywords = None
    symptom_cuis = None
    management_plans = None
    _initialized = False

    @classmethod
    def initialize(cls):
        if cls._initialized:
            return
        if cls.nlp is None:
            try:
                cls.nlp = spacy.load("en_core_sci_sm", disable=["ner", "lemmatizer"])
                cls.nlp.add_pipe("sentencizer")
                logger.info("Initialized shared en_core_sci_sm with sentencizer")
            except OSError:
                logger.error("SpaCy model not available")
                raise
        if cls.clinical_terms is None:
            cls.clinical_terms = ClinicalNER._load_clinical_terms()
        if cls.disease_signatures is None:
            cls.disease_signatures = cls._load_disease_signatures_static()
        if cls.disease_keywords is None or cls.symptom_cuis is None or cls.management_plans is None:
            cls.disease_keywords, cls.symptom_cuis, cls.management_plans = cls._load_keyword_cui_plans()
        cls._initialized = True

    @staticmethod
    def _load_disease_signatures_static():
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT d.name, s.name as symptom
                    FROM diseases d
                    JOIN disease_symptoms ds ON d.id = ds.disease_id
                    JOIN symptoms s ON ds.symptom_id = s.id
                """)
                signatures = defaultdict(set)
                for row in cursor.fetchall():
                    signatures[row['name']].add(row['symptom'])
                return signatures
        except Exception as e:
            logger.error(f"Error loading disease signatures: {e}")
            return {
                "pneumonia": {"cough", "fever", "chest pain"},
                "myocardial_infarction": {"chest pain", "shortness of breath", "sweating"},
                "stroke": {"headache", "weakness", "speech difficulty", "facial droop"},
                "gastroenteritis": {"diarrhea", "nausea", "vomiting", "abdominal pain"},
                "uti": {"dysuria", "frequency", "urgency", "suprapubic pain"},
                "asthma": {"wheezing", "shortness of breath", "cough"},
                "diabetes": {"thirst", "polyuria", "fatigue", "blurred vision"},
                "hypertension": {"headache", "dizziness", "nosebleed"},
                "musculoskeletal_back_pain": {"back pain"},
                "malaria": {"fever", "chills", "headache", "nausea", "vomiting", "loss of appetite", "jaundice"},
                "viral_hepatitis": {"jaundice", "nausea", "vomiting", "loss of appetite", "fatigue"},
                "dengue": {"fever", "headache", "nausea", "vomiting", "fatigue"},
                "typhoid": {"fever", "headache", "abdominal pain", "loss of appetite"}
            }

    @staticmethod
    def _load_keyword_cui_plans():
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT dk.keyword, dk.cui
                    FROM disease_keywords dk
                    JOIN diseases d ON dk.disease_id = d.id
                """)
                disease_keywords = {row['keyword'].lower(): row['cui'] for row in cursor.fetchall()}
                
                cursor.execute("SELECT name, cui FROM symptoms")
                symptom_cuis = {row['name'].lower(): row['cui'] for row in cursor.fetchall()}
                
                cursor.execute("""
                    SELECT d.name, dmp.plan
                    FROM disease_management_plans dmp
                    JOIN diseases d ON dmp.disease_id = d.id
                """)
                management_plans = {row['name'].lower(): row['plan'] for row in cursor.fetchall()}
                
                return disease_keywords, symptom_cuis, management_plans
        except Exception as e:
            logger.error(f"Error loading keywords, CUIs, and plans: {e}")
            return {
                "pneumonia": "C0032285",
                "malaria": "C0024530"
            }, {
                "fever": "C0015967",
                "cough": "C0010200"
            }, {
                "pneumonia": "Antibiotics, oxygen therapy"
            }

    def __init__(self, ner_model=None):
        self.ner = ner_model or ClinicalNER()
        self.umls_mapper = UMLSMapper()
        self.primary_threshold = 2.0
    
    def predict_from_text(self, text: str) -> Dict:
        start_time = time.time()
        entities = self.ner.extract_entities(text)
        if not entities:
            logger.debug(f"Prediction took {time.time() - start_time:.3f} seconds")
            return {"primary_diagnosis": None, "differential_diagnoses": []}
        
        terms = {ent[0].lower() for ent in entities}
        disease_scores = defaultdict(float)
        
        for disease, signature in self.disease_signatures.items():
            matches = len(terms.intersection(signature))
            if matches > 0:
                disease_scores[disease] = matches
        
        sorted_diseases = sorted(
            [{"disease": k, "score": v} for k, v in disease_scores.items()],
            key=lambda x: x["score"],
            reverse=True
        )[:5]
        
        primary_diagnosis = None
        differential_diagnoses = []
        if sorted_diseases:
            if sorted_diseases[0]["score"] >= self.primary_threshold:
                primary_diagnosis = sorted_diseases[0]
                differential_diagnoses = sorted_diseases[1:] if len(sorted_diseases) > 1 else []
            else:
                differential_diagnoses = sorted_diseases
        
        logger.debug(f"Prediction took {time.time() - start_time:.3f} seconds")
        return {
            "primary_diagnosis": primary_diagnosis,
            "differential_diagnoses": differential_diagnoses
        }
    
    def process_soap_note(self, note: Dict) -> Dict:
        start_time = time.time()
        text = prepare_note_for_nlp(note)
        logger.debug(f"Text preparation took {time.time() - start_time:.3f} seconds")
        if not text:
            return {"error": "No valid text in note"}
        
        t = time.time()
        doc = self.ner.nlp(text)
        logger.debug(f"spaCy processing took {time.time() - t:.3f} seconds")
        
        t = time.time()
        summary = generate_summary(text, doc=doc)
        logger.debug(f"Summary generation took {time.time() - t:.3f} seconds")
        
        t = time.time()
        expected_keywords, reference_cuis = self.ner.extract_keywords_and_cuis(note)
        logger.debug(f"Keyword/CUI extraction took {time.time() - t:.3f} seconds")
        
        t = time.time()
        entities = self.ner.extract_entities(text, doc=doc)
        logger.debug(f"Entity extraction took {time.time() - t:.3f} seconds")
        
        t = time.time()
        terms = set()
        for ent, _, _ in entities:
            clean_text = re.sub(r'[^\w\s]', '', ent).strip().lower()
            if clean_text:
                terms.add(clean_text)
                for word in clean_text.split():
                    if len(word) > 3:
                        lemma = self.ner.lemmatizer.lemmatize(word)
                        terms.add(lemma)
        
        terms.update([kw.lower() for kw in expected_keywords])
        symptom_cuis_map = self.umls_mapper.map_terms_to_cuis(list(terms))
        logger.debug(f"UMLS mapping took {time.time() - t:.3f} seconds")
        
        t = time.time()
        predictions = self.predict_from_text(text)
        logger.debug(f"Prediction took {time.time() - t:.3f} seconds")
        
        t = time.time()
        management_plans = {}
        try:
            if predictions["primary_diagnosis"]:
                disease = predictions["primary_diagnosis"]["disease"].lower()
                if disease in self.management_plans:
                    management_plans[disease] = self.management_plans[disease]
            for disease in predictions["differential_diagnoses"]:
                disease_name = disease["disease"].lower()
                if disease_name in self.management_plans:
                    management_plans[disease_name] = self.management_plans[disease_name]
        except Exception as e:
            logger.error(f"Error fetching management plans: {e}")
        logger.debug(f"Management plans retrieval took {time.time() - t:.3f} seconds")
        
        t = time.time()
        result = {
            "note_id": note["id"],
            "patient_id": note["patient_id"],
            "primary_diagnosis": predictions["primary_diagnosis"],
            "differential_diagnoses": predictions["differential_diagnoses"],
            "keywords": expected_keywords,
            "cuis": reference_cuis,
            "entities": entities,
            "summary": summary,
            "management_plans": management_plans,
            "processed_at": datetime.utcnow().isoformat(),
            "processing_time": time.time() - start_time
        }
        
        update_ai_analysis(note["id"], result, summary)
        logger.debug(f"Database update took {time.time() - t:.3f} seconds")
        logger.info(f"Total processing time for note ID {note['id']}: {result['processing_time']:.3f} seconds")
        return result

# FastAPI Application
app = FastAPI(
    title="Clinical NLP API",
    description="Real-time clinical NLP processing service",
    version="1.0.0"
)

# Initialize shared resources
DiseasePredictor.initialize()

text_predictor = DiseasePredictor()
note_processor = DiseasePredictor()

class PredictionRequest(BaseModel):
    text: str
    department: Optional[str] = None

class ProcessNoteRequest(BaseModel):
    note_id: int

class DiseasePrediction(BaseModel):
    disease: str
    score: float

class EntityContext(BaseModel):
    severity: int
    temporal: str

class EntityDetail(BaseModel):
    text: str
    label: str
    context: EntityContext

class PredictionResponse(BaseModel):
    primary_diagnosis: Optional[DiseasePrediction]
    differential_diagnoses: List[DiseasePrediction]
    processing_time: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()
    predictions = text_predictor.predict_from_text(request.text)
    return PredictionResponse(
        primary_diagnosis=predictions["primary_diagnosis"],
        differential_diagnoses=predictions["differential_diagnoses"],
        processing_time=time.time() - start_time
    )

@app.post("/process_note")
async def process_note(request: ProcessNoteRequest):
    start_time = time.time()
    note = fetch_single_soap_note(request.note_id)
    if not note:
        return Response(
            content=generate_html_response({"detail": "Note not found"}, 404),
            status_code=404,
            media_type="text/html"
        )
    
    result = note_processor.process_soap_note(note)
    if "error" in result:
        return Response(
            content=generate_html_response({"detail": result["error"]}, 400),
            status_code=400,
            media_type="text/html"
        )
    
    return Response(
        content=generate_html_response(result, 200),
        status_code=200,
        media_type="text/html"
    )

# Function to start the server programmatically
def start_server():
    """Start the FastAPI server in a separate process."""
    logger.info("Starting FastAPI server programmatically")
    uvicorn.run(app, host=HIMS_CONFIG["API_HOST"], port=HIMS_CONFIG["API_PORT"], log_level="info")

# Enhanced CLI
class HIMSCLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='HIMS Clinical NLP System',
            formatter_class=argparse.RawTextHelpFormatter
        )
        self._setup_commands()
    
    def _setup_commands(self):
        subparsers = self.parser.add_subparsers(dest='command')
        
        status_parser = subparsers.add_parser('status', help='System status')
        status_parser.add_argument('--detail', action='store_true', help='Detailed status')
        
        predict_parser = subparsers.add_parser('predict', help='Run prediction')
        predict_parser.add_argument('text', help='Clinical text to analyze')
        
        process_parser = subparsers.add_parser('process', help='Process SOAP notes')
        process_parser.add_argument('--note-id', type=int, help='Process specific note by ID')
        process_parser.add_argument('--all', action='store_true', help='Process all unprocessed notes')
        process_parser.add_argument('--limit', type=int, default=10, help='Limit number of notes to process')
        process_parser.add_argument('--latest', action='store_true', help='Process the most recently inserted note')
        
        test_parser = subparsers.add_parser('test', help='Run unit tests with server')
    
    def run(self):
        args = self.parser.parse_args()
        
        if not args.command:
            self.parser.print_help()
            return
        
        if args.command == 'status':
            self._show_status(args.detail)
        elif args.command == 'predict':
            self._run_prediction(args.text)
        elif args.command == 'process':
            self._process_notes(args.note_id, args.all, args.limit, args.latest)
        elif args.command == 'test':
            self._run_tests()
    
    def _show_status(self, detail=False):
        status = {
            "NLP Model": "Loaded" if DiseasePredictor.nlp else "Error",
            "SQLite Database": HIMS_CONFIG["SQLITE_DB_PATH"]
        }
        
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM soap_notes")
                note_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM soap_notes WHERE ai_analysis IS NOT NULL")
                processed_count = cursor.fetchone()[0]
                status["SOAP Notes"] = f"{processed_count}/{note_count} processed"
        except:
            status["SOAP Notes"] = "Unknown"
        
        table = Table(title="System Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        for k, v in status.items():
            table.add_row(k, v)
        
        console.print(table)
        
        if detail:
            try:
                with get_sqlite_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT id, patient_id, created_at, 
                               CASE WHEN ai_analysis IS NULL THEN 'Pending' ELSE 'Processed' END AS status
                        FROM soap_notes
                        ORDER BY created_at DESC
                        LIMIT 5
                    """)
                    recent_notes = cursor.fetchall()
                    
                    if recent_notes:
                        note_table = Table(title="Recent SOAP Notes")
                        note_table.add_column("ID", style="cyan")
                        note_table.add_column("Patient ID", style="magenta")
                        note_table.add_column("Created At", style="yellow")
                        note_table.add_column("Status", style="green")
                        
                        for note in recent_notes:
                            note_table.add_row(
                                str(note['id']),
                                note['patient_id'],
                                note['created_at'],
                                note['status']
                            )
                        console.print(note_table)
            except Exception as e:
                console.print(f"[yellow]Couldn't fetch recent notes: {e}[/yellow]")
    
    def _run_prediction(self, text):
        console.print(Panel("Clinical Text Analysis", style="bold blue"))
        console.print(f"Input: {text[:200]}...\n")
        
        result = text_predictor.predict_from_text(text)
        
        if not result["primary_diagnosis"] and not result["differential_diagnoses"]:
            console.print("[yellow]No diseases predicted[/yellow]")
            return
        
        if result["primary_diagnosis"]:
            table = Table(title="Primary Diagnosis")
            table.add_column("Disease", style="magenta")
            table.add_column("Score", style="green")
            table.add_row(result["primary_diagnosis"]["disease"], str(result["primary_diagnosis"]["score"]))
            console.print(table)
        
        if result["differential_diagnoses"]:
            table = Table(title="Differential Diagnoses")
            table.add_column("Disease", style="magenta")
            table.add_column("Score", style="green")
            for disease in result["differential_diagnoses"]:
                table.add_row(disease["disease"], str(disease["score"]))
            console.print(table)
    
    def _process_notes(self, note_id, process_all, limit, latest=False):
        processor = DiseasePredictor()
        
        if latest:
            console.print(Panel("Processing Latest Note", style="bold green"))
            try:
                with get_sqlite_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT id FROM soap_notes ORDER BY created_at DESC LIMIT 1")
                    result = cursor.fetchone()
                    if result:
                        note_id = result['id']
                    else:
                        console.print("[yellow]No notes found in database[/yellow]")
                        return
            except Exception as e:
                logger.error(f"Error fetching latest note: {e}")
                console.print("[red]Failed to fetch latest note[/red]")
                return
        
        if note_id:
            console.print(Panel(f"Processing Note ID: {note_id}", style="bold green"))
            note = fetch_single_soap_note(note_id)
            if note:
                result = processor.process_soap_note(note)
                
                console.print(Panel("Processing Results", style="bold cyan"))
                console.print(f"Note ID: {result['note_id']}")
                console.print(f"Patient ID: {result['patient_id']}")
                
                console.print(Panel("AI Summary", style="bold yellow"))
                console.print(result['summary'] + "\n")
                
                if result['primary_diagnosis']:
                    table = Table(title="Primary Diagnosis")
                    table.add_column("Disease", style="magenta")
                    table.add_column("Score", style="green")
                    table.add_row(result["primary_diagnosis"]["disease"], str(result["primary_diagnosis"]["score"]))
                    console.print(table)
                
                if result['differential_diagnoses']:
                    table = Table(title="Differential Diagnoses")
                    table.add_column("Disease", style="magenta")
                    table.add_column("Score", style="green")
                    for disease in result['differential_diagnoses']:
                        table.add_row(disease['disease'], str(disease['score']))
                    console.print(table)
                
                if not result['primary_diagnosis'] and not result['differential_diagnoses']:
                    console.print("[yellow]No diseases predicted[/yellow]")
                
                if result['management_plans']:
                    plan_table = Table(title="Management Plans")
                    plan_table.add_column("Disease", style="magenta")
                    plan_table.add_column("Plan", style="green")
                    for disease, plan in result['management_plans'].items():
                        plan_table.add_row(disease, plan)
                    console.print(plan_table)
                
                if result['keywords']:
                    console.print(f"Keywords: {', '.join(result['keywords'])}")
                
                if result['cuis']:
                    console.print(f"CUIs: {', '.join(result['cuis'])}")
                
                if result['entities']:
                    entity_table = Table(title="Extracted Entities")
                    entity_table.add_column("Text", style="cyan")
                    entity_table.add_column("Label", style="magenta")
                    entity_table.add_column("Temporal", style="yellow")
                    for ent in result['entities']:
                        entity_table.add_row(ent[0], ent[1], ent[2]["temporal"])
                    console.print(entity_table)
            else:
                console.print(f"[red]Note ID {note_id} not found[/red]")
        elif process_all:
            console.print(Panel("Processing All Notes", style="bold green"))
            notes = fetch_soap_notes()
            if not notes:
                console.print("[yellow]No notes found in database[/yellow]")
                return
                
            notes_to_process = notes[:limit]
            success_count = 0
            
            for note in track(notes_to_process, description="Processing..."):
                try:
                    processor.process_soap_note(note)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Failed to process note {note.get('id')}: {e}")
            
            console.print(f"[green]Successfully processed {success_count}/{len(notes_to_process)} notes[/green]")
        else:
            console.print("[yellow]Specify --note-id, --all, or --latest to process notes[/yellow]")
    
    def _run_tests(self):
        """Run unit tests with server."""
        suite = unittest.TestLoader().loadTestsFromTestCase(TestNLPApi)
        unittest.TextTestRunner().run(suite)

# Test Suite
class TestNLPApi(unittest.TestCase):
    server_process = None

    @classmethod
    def setUpClass(cls):
        """Start the server and set up test database."""
        # Set up in-memory database
        global get_sqlite_connection
        cls.test_db = setup_test_database()
        def mock_sqlite_connection():
            return cls.test_db
        get_sqlite_connection = mock_sqlite_connection
        
        # Initialize resources
        nltk.download('wordnet', quiet=True)
        DiseasePredictor.initialize()
        
        # Start server in a separate process
        cls.server_process = multiprocessing.Process(target=start_server)
        cls.server_process.start()
        time.sleep(2)  # Wait for server to start
        
        # Initialize test client
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        """Stop the server."""
        if cls.server_process:
            cls.server_process.terminate()
            cls.server_process.join()
            logger.info("Server process terminated")

    def test_server_is_running(self):
        """Test if the server is running."""
        try:
            response = requests.get(f"http://{HIMS_CONFIG['API_HOST']}:{HIMS_CONFIG['API_PORT']}/docs", timeout=5)
            self.assertEqual(response.status_code, 200)
        except requests.ConnectionError:
            self.fail("Server is not running")

    def test_predict_endpoint(self):
        """Test the /predict endpoint."""
        response = self.client.post("/predict", json={"text": "Patient has fever and cough"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("primary_diagnosis", data)
        self.assertIn("differential_diagnoses", data)
        self.assertIn("processing_time", data)
        if data["primary_diagnosis"]:
            self.assertEqual(data["primary_diagnosis"]["disease"], "pneumonia")

    def test_process_note_endpoint(self):
        """Test the /process_note endpoint."""
        response = self.client.post("/process_note", json={"note_id": 1})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        html_content = response.text
        self.assertIn("Clinical Note Analysis - Note ID 1", html_content)
        self.assertIn("PAT001", html_content)
        self.assertIn("pneumonia", html_content)
        self.assertIn("Antibiotics, oxygen therapy", html_content)

    def test_process_note_not_found(self):
        """Test the /process_note endpoint with invalid note_id."""
        response = self.client.post("/process_note", json={"note_id": 999})
        self.assertEqual(response.status_code, 404)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Note Not Found", response.text)

# Save UMLS cache on program exit
import atexit
atexit.register(UMLSMapper().save_cache)

# Main Execution
if __name__ == '__main__':
    nltk.download('wordnet', quiet=True)
    DiseasePredictor.initialize()
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        HIMSCLI().run()
    else:
        start_server()