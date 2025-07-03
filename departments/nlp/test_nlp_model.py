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
from fastapi import FastAPI, HTTPException, Response
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
import sys
import html
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import requests
from resources.priority_symptoms import PRIORITY_SYMPTOMS
from resources.common_terms import common_terms
import sqlite3
from resources.default_patterns import DEFAULT_PATTERNS
from resources.default_clinical_terms import DEFAULT_CLINICAL_TERMS
from resources.default_disease_keywords import DEFAULT_DISEASE_KEYWORDS
from resources.common_fallbacks import (
    fallback_disease_keywords,
    fallback_symptom_cuis,
    fallback_management_plans
)
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
    "PRIORITY_SYMPTOMS": PRIORITY_SYMPTOMS,
    "UMLS_THRESHOLD": 0.7,
    "SQLITE_DB_PATH": "/home/mathu/projects/hospital/instance/hims.db",
    "API_HOST": "0.0.0.0",
    "API_PORT": 8000,
    "BATCH_SIZE": 50,
    "UMLS_DB_URL": "postgresql://postgres:postgres@localhost:5432/hospital_umls",
    "TRUSTED_SOURCES": ['MSH', 'SNOMEDCT_US', 'ICD10CM', 'ICD9CM', 'LNC'],
    "UMLS_LANGUAGE": 'ENG'
}

console = Console()
# UMLS Database Setup
umls_engine = create_engine(HIMS_CONFIG["UMLS_DB_URL"])
UMLSSession = sessionmaker(bind=umls_engine)
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

# UMLS Mapper
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

# Disease-Symptom Mapper
class DiseaseSymptomMapper:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DiseaseSymptomMapper, cls).__new__(cls)
            cls._instance.cache = {}
        return cls._instance
    
    def get_disease_symptoms(self, disease_cui: str) -> List[Dict]:
        """Get symptoms associated with a disease CUI from UMLS"""
        if disease_cui in self.cache:
            return self.cache[disease_cui]
            
        try:
            session = UMLSSession()
            # Get symptoms through UMLS relationships
            symptoms = session.execute(text("""
                SELECT DISTINCT c2.str AS symptom_name, c2.cui AS symptom_cui
                FROM umls.mrrel r
                JOIN umls.mrconso c1 ON r.cui1 = c1.cui
                JOIN umls.mrconso c2 ON r.cui2 = c2.cui
                WHERE r.cui1 = :disease_cui
                    AND r.rela IN ('manifestation_of', 'has_finding', 'has_sign_or_symptom')
                    AND c1.lat = :language AND c1.suppress = 'N'
                    AND c2.lat = :language AND c2.suppress = 'N'
                    AND c1.sab IN :trusted_sources
                    AND c2.sab IN :trusted_sources
            """), {
                'disease_cui': disease_cui,
                'language': HIMS_CONFIG["UMLS_LANGUAGE"],
                'trusted_sources': tuple(HIMS_CONFIG["TRUSTED_SOURCES"])
            }).fetchall()
            
            session.close()
            
            # Cache results
            symptom_list = [{'name': row.symptom_name, 'cui': row.symptom_cui} 
                           for row in symptoms]
            self.cache[disease_cui] = symptom_list
            return symptom_list
            
        except Exception as e:
            logger.error(f"Error fetching symptoms for disease CUI {disease_cui}: {e}")
            return []
    
    def get_symptom_diseases(self, symptom_cui: str) -> List[Dict]:
        """Get diseases associated with a symptom CUI from UMLS"""
        try:
            session = UMLSSession()
            # Get diseases through UMLS relationships
            diseases = session.execute(text("""
                SELECT DISTINCT c1.str AS disease_name, c1.cui AS disease_cui
                FROM umls.mrrel r
                JOIN umls.mrconso c1 ON r.cui1 = c1.cui
                JOIN umls.mrconso c2 ON r.cui2 = c2.cui
                WHERE r.cui2 = :symptom_cui
                    AND r.rela IN ('manifestation_of', 'has_finding', 'has_sign_or_symptom')
                    AND c1.lat = :language AND c1.suppress = 'N'
                    AND c2.lat = :language AND c2.suppress = 'N'
                    AND c1.sab IN :trusted_sources
                    AND c2.sab IN :trusted_sources
            """), {
                'symptom_cui': symptom_cui,
                'language': HIMS_CONFIG["UMLS_LANGUAGE"],
                'trusted_sources': tuple(HIMS_CONFIG["TRUSTED_SOURCES"])
            }).fetchall()
            
            session.close()
            
            return [{'name': row.disease_name, 'cui': row.disease_cui} 
                   for row in diseases]
            
        except Exception as e:
            logger.error(f"Error fetching diseases for symptom CUI {symptom_cui}: {e}")
            return []
    
    def build_disease_signatures(self) -> Dict[str, set]:
        """Build disease signatures using UMLS relationships"""
        disease_signatures = defaultdict(set)
        umls_mapper = UMLSMapper()        
        try:
            # Get all diseases from application database
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, name FROM diseases")
                diseases = cursor.fetchall()
                
                for disease_id, disease_name in diseases:
                    # Get CUI for disease
                    disease_cui = None
                    cursor.execute("""
                        SELECT dk.cui 
                        FROM disease_keywords dk
                        WHERE dk.disease_id = ?
                        LIMIT 1
                    """, (disease_id,))
                    if row := cursor.fetchone():
                        disease_cui = row['cui']
                    
                    # If no CUI in app DB, try UMLS mapping
                    if not disease_cui:
                        disease_cuis = umls_mapper.map_term_to_cui(disease_name)
                        disease_cui = disease_cuis[0] if disease_cuis else None
                    
                    if disease_cui:
                        # Get symptoms from UMLS
                        symptoms = self.get_disease_symptoms(disease_cui)
                        for symptom in symptoms:
                            disease_signatures[disease_name].add(symptom['name'].lower())
        
            return disease_signatures
        
        except Exception as e:
            logger.error(f"Error building disease signatures: {e}")
            return DiseasePredictor._load_disease_signatures_static()

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
    <div class="container">
        <div class="error-icon">❌</div>
        <h5>Error: Note Not Found</h5>
        <p>{html.escape(data.get('detail', 'The requested note was not found in the database.'))}</p>
    </div>
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
            <h5>Primary Diagnosis</h5>
            <div class="info-card">
                <strong>Disease:</strong> {html.escape(primary_diagnosis.get("disease", "N/A"))}
                <span class="badge badge-success">Score: {primary_diagnosis.get("score", "N/A")}</span>
            </div>
        </div>
        """
    else:
        primary_diagnosis_html = """
        <div class="section">
            <h5>Primary Diagnosis</h5>
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
            <h5>Differential Diagnoses</h5>
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
        entity_items = ""
        for entity in entities:
            text, label, context = entity
            diseases = context.get("associated_diseases", [])
            disease_symptom_map = context.get("disease_symptom_map", {})
            disease_list = ""
            
            if diseases:
                # Group diseases by their base name (without parenthetical qualifiers)
                disease_groups = defaultdict(list)
                for disease in diseases:
                    base_name = re.sub(r'\(.*?\)', '', disease).strip()
                    disease_groups[base_name].append(disease)
                
                # Create disease list with symptom details
                disease_items = []
                for base_name, variants in disease_groups.items():
                    # Get all symptoms associated with these disease variants
                    all_symptoms = set()
                    for variant in variants:
                        if variant in disease_symptom_map:
                            all_symptoms.update(disease_symptom_map[variant])
                    
                    # Format symptoms list
                    symptom_list = ", ".join(sorted(all_symptoms)[:3])  # Show up to 3 symptoms
                    if len(all_symptoms) > 3:
                        symptom_list += ", ..."
                    
                    # Use the first variant as representative name
                    disease_items.append(f"{html.escape(variants[0])}: {html.escape(symptom_list)}")
                
                disease_list = "<div class='diseases'><strong>Associated Diseases:</strong><ul><li>" + \
                               "</li><li>".join(disease_items) + "</li></ul></div>"
            
            entity_items += f"""
            <div class="entity">
                <div class="entity-header">
                    <strong>{html.escape(text)}</strong>
                    <span class="badge badge-primary">{html.escape(label)}</span>
                </div>
                <div class="entity-details">
                    <div><small>Severity: {context["severity"]}, Temporal: {html.escape(context["temporal"])}</small></div>
                    {disease_list}
                </div>
            </div>
            """
        
        entities_html = f"""
        <div class="section entities">
            <h5>Entities</h5>
            <div class="entity-container">
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
            <h5>Management Plans</h5>
            {plans_html}
        </div>
        """
    # Generate the complete HTML
    html_content = f"""
    <div class="container">
        <h5>Clinical Note Analysis - Note ID {note_id}</h5>
        
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
            <h5>Keywords</h5>
            <ul>
                {"".join(f'<li>{keyword}</li>' for keyword in keywords)}
            </ul>
        </div>

        <div class="section">
            <h5>CUIs</h5>
            <ul>
                {"".join(f'<li>{cui}</li>' for cui in cuis)}
            </ul>
        </div>

        {entities_html}
        {management_plans_html}

        <div class="section summary">
            <h5>Summary</h5>
            <p>{summary}</p>
        </div>
    </div>
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
def generate_summary(text: str, max_sentences: int = 4, doc=None) -> str:
    start_time = time.time()
    if not doc:
        doc = DiseasePredictor.nlp(text)
    
    # Enhanced clinical term recognition
    clinical_terms = DiseasePredictor.clinical_terms
    temporal_pattern = re.compile(r"\b(\d+\s*(day|days|week|weeks|month|months|year|years)\s*(ago)?)\b", re.IGNORECASE)
    
    # Extract key information
    symptoms = []
    temporal_info = []
    aggravating_factors = defaultdict(list)
    alleviating_factors = defaultdict(list)
    medications = []
    findings = []
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        
        # Extract temporal information
        if not temporal_info:
            match = temporal_pattern.search(sent_text)
            if match:
                temporal_info.append(f"started {match.group(0)}")
        
        # Extract symptoms with context
        for term in clinical_terms:
            if term in sent_text.lower():
                if term not in symptoms:
                    symptoms.append(term)
                
                # Extract modifying phrases
                if "makes" in sent_text or "worsen" in sent_text:
                    aggravating_factors[term].append(sent_text.split(term)[-1].split(".")[0].strip())
                elif "alleviates" in sent_text or "relieves" in sent_text or "better" in sent_text:
                    alleviating_factors[term].append(sent_text.split(term)[-1].split(".")[0].strip())
        
        # Extract medications
        if "mg" in sent_text or "take" in sent_text or "medication" in sent_text:
            medications.append(sent_text)
        
        # Extract findings
        if "found" in sent_text or "show" in sent_text or "reveal" in sent_text or "report" in sent_text:
            findings.append(sent_text)
    
    # Construct natural summary
    summary_parts = []
    
    # Patient presentation
    if symptoms:
        symptoms_str = ", ".join(symptoms[:-1]) + " and " + symptoms[-1] if len(symptoms) > 1 else symptoms[0]
        presentation = f"The patient presents with {symptoms_str}"
        if temporal_info:
            presentation += f" that {temporal_info[0]}"
        summary_parts.append(presentation + ".")
    
    # Symptom modifiers
    for symptom, factors in aggravating_factors.items():
        if factors:
            summary_parts.append(f"{symptom.capitalize()} is aggravated by {factors[0]}.")
    
    for symptom, factors in alleviating_factors.items():
        if factors:
            summary_parts.append(f"{symptom.capitalize()} is alleviated by {factors[0]}.")
    
    # Medical history and medications
    if medications:
        meds_str = ". ".join(medications[:2])  # Take max 2 medication mentions
        summary_parts.append(meds_str)
    
    # Clinical findings
    if findings:
        findings_str = ". ".join(findings[:2])  # Take max 2 findings
        summary_parts.append(findings_str)
    
    # Ensure we don't exceed max sentences
    summary = " ".join(summary_parts[:max_sentences])
    
    # Add missing critical symptoms not captured in sentences
    missing_symptoms = [symptom for symptom in symptoms if symptom not in summary.lower()]
    if missing_symptoms:
        if len(missing_symptoms) > 1:
            missing_str = ", ".join(missing_symptoms[:-1]) + " and " + missing_symptoms[-1]
        else:
            missing_str = missing_symptoms[0]
        summary += f" Additional symptoms include {missing_str}."
    
    # Final cleanup
    summary = re.sub(r"\s+", " ", summary)  # Remove extra spaces
    summary = re.sub(r"\.+", ".", summary)  # Remove multiple periods
    summary = summary.strip()
    
    # Capitalize first letter
    if summary and summary[0].islower():
        summary = summary[0].upper() + summary[1:]
    
    logger.debug(f"Summary generation took {time.time() - start_time:.3f} seconds")
    return summary if summary else text[:200] + "..."

def update_ai_analysis(note_id: int, ai_analysis_html: str, summary: str) -> bool:
    """
    Update AI analysis and summary for a specific note in the soap_notes table.
    Stores HTML content in ai_analysis column.
    
    Args:
        note_id (int): The ID of the note to update
        ai_analysis_html (str): The HTML content to store
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
                (ai_analysis_html, summary, note_id)
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
                conn.row_factory = sqlite3.Row  # ensure dict-style row access
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT name FROM symptoms
                    UNION
                    SELECT keyword FROM disease_keywords
                """)
                return {row['name'].lower() for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Error loading clinical terms: {e}")
            return DEFAULT_CLINICAL_TERMS

    def __init__(self):
        self.nlp = DiseasePredictor.nlp
        self.negation_terms = {"no", "not", "denies", "without", "absent", "negative"}
        self.lemmatizer = WordNetLemmatizer()
        self.patterns = self._load_patterns()
        self.compiled_patterns = [(label, re.compile(pattern, re.IGNORECASE)) for label, pattern in self.patterns]
        self.symptom_mapper = DiseaseSymptomMapper()
    
    def _load_patterns(self):
        try:
            with get_sqlite_connection() as conn:
                conn.row_factory = sqlite3.Row  # Ensure dict-like access to row data
                cursor = conn.cursor()
                cursor.execute("SELECT label, pattern FROM patterns")
                return [(row['label'], row['pattern']) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
            return DEFAULT_PATTERNS    
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
        # First pass: collect all symptom-disease relationships
        symptom_disease_map = defaultdict(set)
        disease_symptom_count = defaultdict(int)
        disease_symptom_map = defaultdict(set)  # Maps disease to set of symptoms
        
        for i, (entity_text, entity_label, context) in enumerate(entities):
            if entity_label in ["CLINICAL_TERM"] + [label for label, _ in self.compiled_patterns]:
                try:
                    with get_sqlite_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT cui FROM symptoms WHERE name = ?
                        """, (entity_text.lower(),))
                        if row := cursor.fetchone():
                            symptom_cui = row['cui']
                            diseases = self.symptom_mapper.get_symptom_diseases(symptom_cui)
                            if diseases:
                                for disease in diseases:
                                    disease_name = disease['name']
                                    symptom_disease_map[entity_text.lower()].add(disease_name)
                                    disease_symptom_count[disease_name] += 1
                                    disease_symptom_map[disease_name].add(entity_text.lower())
                except Exception as e:
                    logger.error(f"Error looking up symptom: {entity_text}: {e}")
        
        # Second pass: filter diseases based on symptom count
        for i, (entity_text, entity_label, context) in enumerate(entities):
            if entity_label in ["CLINICAL_TERM"] + [label for label, _ in self.compiled_patterns]:
                diseases = symptom_disease_map.get(entity_text.lower(), set())
                # Filter to diseases with at least 2 matching symptoms
                filtered_diseases = [
                    d for d in diseases 
                    if disease_symptom_count.get(d, 0) >= 2
                ]
                if filtered_diseases:
                    context["associated_diseases"] = filtered_diseases
                    context["disease_symptom_map"] = disease_symptom_map
                    entities[i] = (entity_text, entity_label, context)
        
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
                        #--
        except Exception as e:
            logger.error(f"Error fetching keywords from database: {e}")
            disease_keywords = DEFAULT_DISEASE_KEYWORDS
            for term in terms:
                for keyword, cui in disease_keywords.items():
                    if keyword in term or term in keyword:
                        if keyword not in expected_keywords:
                            expected_keywords.append(keyword)
                            reference_cuis.append(cui)
                        break        
        logger.debug(f"Keyword and CUI extraction took {time.time() - start_time:.3f} seconds")
        return list(set(expected_keywords)), list(set(reference_cuis))
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
            mapper = DiseaseSymptomMapper()
            cls.disease_signatures = mapper.build_disease_signatures()
            logger.info(f"Loaded {len(cls.disease_signatures)} disease signatures from UMLS")
        if cls.disease_keywords is None or cls.symptom_cuis is None or cls.management_plans is None:
            cls.disease_keywords, cls.symptom_cuis, cls.management_plans = cls._load_keyword_cui_plans()
        cls._initialized = True

    @staticmethod
    def _load_keyword_cui_plans():
        """Load keywords, CUIs, and management plans from databases."""
        try:
            with get_sqlite_connection() as conn:
                conn.row_factory = sqlite3.Row  # Ensure rows can be accessed as dictionaries
                cursor = conn.cursor()

                # Load disease keywords
                cursor.execute("""
                    SELECT d.name, dk.keyword, dk.cui
                    FROM diseases d
                    JOIN disease_keywords dk ON d.id = dk.disease_id
                """)
                disease_keywords = {
                    row['keyword'].lower(): row['cui']
                    for row in cursor.fetchall()
                }

                # Load symptom CUIs
                cursor.execute("SELECT name, cui FROM symptoms")
                symptom_cuis = {
                    row['name'].lower(): row['cui']
                    for row in cursor.fetchall()
                }

                # Load management plans
                cursor.execute("""
                    SELECT d.name, dmp.plan
                    FROM disease_management_plans dmp
                    JOIN diseases d ON dmp.disease_id = d.id
                """)
                management_plans = {
                    row['name'].lower(): row['plan']
                    for row in cursor.fetchall()
                }

            return disease_keywords, symptom_cuis, management_plans

        except Exception as e:
            logger.error("Failed to load data from SQLite DB, using fallback data. Error: %s", str(e))
            return (
                fallback_disease_keywords,
                fallback_symptom_cuis,
                fallback_management_plans
            )

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
        
        # Only consider diseases with at least 2 matching symptoms
        for disease, signature in self.disease_signatures.items():
            matches = len(terms.intersection(signature))
            if matches >= 2:  # Filter condition
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
    # Generate HTML and update database
    html_content = generate_html_response(result, 200)
    update_ai_analysis(note["id"], html_content, result['summary'])    
    return Response(
        content=html_content,
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
                    entity_table.add_column("Associated Diseases", style="green")
                    for ent in result['entities']:
                        diseases = ent[2].get("associated_diseases", [])
                        disease_symptom_map = ent[2].get("disease_symptom_map", {})
                        disease_details = []
                        for disease in diseases:
                            symptoms = disease_symptom_map.get(disease, set())
                            disease_details.append(f"{disease}: {', '.join(sorted(symptoms)[:3])}")
                        entity_table.add_row(
                            ent[0], 
                            ent[1], 
                            ent[2]["temporal"], 
                            "\n".join(disease_details) if disease_details else "None"
                        )
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
                    result = processor.process_soap_note(note)
                    html_content = generate_html_response(result, 200)
                    update_ai_analysis(note["id"], html_content, result['summary'])
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