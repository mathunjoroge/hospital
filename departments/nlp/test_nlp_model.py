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
from fastapi import FastAPI, HTTPException, Response, Request
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
from functools import lru_cache
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import hashlib
import concurrent.futures

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

# Configuration Management
class AppConfig:
    _config = None
    
    @classmethod
    def load(cls):
        if cls._config is None:
            cls._config = {
                "DEFAULT_DEPARTMENT": os.getenv("DEFAULT_DEPARTMENT", "emergency"),
                "PRIORITY_SYMPTOMS": PRIORITY_SYMPTOMS,
                "UMLS_THRESHOLD": float(os.getenv("UMLS_THRESHOLD", 0.7)),
                "SQLITE_DB_PATH": os.getenv("SQLITE_DB_PATH", "/home/mathu/projects/hospital/instance/hims.db"),
                "API_HOST": os.getenv("API_HOST", "0.0.0.0"),
                "API_PORT": int(os.getenv("API_PORT", 8000)),
                "BATCH_SIZE": int(os.getenv("BATCH_SIZE", 50)),
                "UMLS_DB_URL": os.getenv("UMLS_DB_URL", "postgresql://postgres:postgres@localhost:5432/hospital_umls"),
                "TRUSTED_SOURCES": os.getenv("TRUSTED_SOURCES", "MSH,SNOMEDCT_US,ICD10CM,ICD9CM,LNC").split(','),
                "UMLS_LANGUAGE": os.getenv("UMLS_LANGUAGE", "ENG"),
                "MAX_WORKERS": int(os.getenv("MAX_WORKERS", multiprocessing.cpu_count())),
                "RATE_LIMIT": os.getenv("RATE_LIMIT", "10/minute")
            }
        return cls._config

def get_config():
    return AppConfig.load()

HIMS_CONFIG = get_config()

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

# UMLS Mapper with caching
class UMLSMapper:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UMLSMapper, cls).__new__(cls)
            cls._instance.term_cache = {}
            cls._instance.map_terms_to_cuis(common_terms)
        return cls._instance
    
    @lru_cache(maxsize=5000)
    def map_term_to_cui(self, term: str) -> List[str]:
        term = term.lower()
        if term in self.term_cache:
            return self.term_cache[term]
        
        # Mock implementation - in real system, query UMLS database
        cuis = []  
        self.term_cache[term] = cuis
        return cuis
    
    def map_terms_to_cuis(self, terms: List[str]) -> Dict[str, List[str]]:
        start_time = time.time()
        terms = [t.lower() for t in terms]
        results = {t: self.map_term_to_cui(t) for t in terms}
        logger.debug(f"UMLS mapping took {time.time() - start_time:.3f} seconds")
        return results

# Disease-Symptom Mapper with caching
class DiseaseSymptomMapper:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DiseaseSymptomMapper, cls).__new__(cls)
            cls._instance.cache = {}
        return cls._instance
    
    @lru_cache(maxsize=1000)
    def get_disease_symptoms(self, disease_cui: str) -> List[Dict]:
        """Get symptoms associated with a disease CUI from UMLS"""
        try:
            session = UMLSSession()
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
            
            symptom_list = [{'name': row.symptom_name, 'cui': row.symptom_cui} 
                           for row in symptoms]
            return symptom_list
            
        except Exception as e:
            logger.error(f"Error fetching symptoms for disease CUI {disease_cui}: {e}")
            return []
    
    @lru_cache(maxsize=1000)
    def get_symptom_diseases(self, symptom_cui: str) -> List[Dict]:
        """Get diseases associated with a symptom CUI from UMLS"""
        try:
            session = UMLSSession()
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
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, name FROM diseases")
                diseases = cursor.fetchall()
                
                for disease_id, disease_name in diseases:
                    disease_cui = None
                    cursor.execute("""
                        SELECT dk.cui 
                        FROM disease_keywords dk
                        WHERE dk.disease_id = ?
                        LIMIT 1
                    """, (disease_id,))
                    if row := cursor.fetchone():
                        disease_cui = row['cui']
                    
                    if not disease_cui:
                        disease_cuis = umls_mapper.map_term_to_cui(disease_name)
                        disease_cui = disease_cuis[0] if disease_cuis else None
                    
                    if disease_cui:
                        symptoms = self.get_disease_symptoms(disease_cui)
                        for symptom in symptoms:
                            disease_signatures[disease_name].add(symptom['name'].lower())
        
            return disease_signatures
        
        except Exception as e:
            logger.error(f"Error building disease signatures: {e}")
            return {}

# HTML Response Generator - Modularized
def generate_primary_diagnosis_html(primary_diagnosis: dict) -> str:
    if not primary_diagnosis:
        return """
        <div class="section">
            <h5>Primary Diagnosis</h5>
            <p>No primary diagnosis identified</p>
        </div>
        """
    
    return f"""
    <div class="section diagnosis">
        <h5>Primary Diagnosis</h5>
        <div class="info-card">
            <strong>Disease:</strong> {html.escape(primary_diagnosis.get("disease", "N/A"))}
            <span class="badge badge-success">Score: {primary_diagnosis.get("score", "N/A")}</span>
        </div>
    </div>
    """

def generate_differential_diagnoses_html(differential_diagnoses: list) -> str:
    if not differential_diagnoses:
        return ""
    
    table_rows = "".join(
        f'<tr><td>{html.escape(diag["disease"])}</td><td>{diag["score"]}</td></tr>'
        for diag in differential_diagnoses
    )
    return f"""
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

def generate_entities_html(entities: list) -> str:
    if not entities:
        return ""
    
    entity_items = ""
    for entity in entities:
        text, label, context = entity
        diseases = context.get("associated_diseases", [])
        disease_symptom_map = context.get("disease_symptom_map", {})
        disease_list = ""
        
        if diseases:
            disease_groups = defaultdict(list)
            for disease in diseases:
                base_name = re.sub(r'\(.*?\)', '', disease).strip()
                disease_groups[base_name].append(disease)
            
            disease_items = []
            for base_name, variants in disease_groups.items():
                all_symptoms = set()
                for variant in variants:
                    if variant in disease_symptom_map:
                        all_symptoms.update(disease_symptom_map[variant])
                
                symptom_list = ", ".join(sorted(all_symptoms)[:3])
                if len(all_symptoms) > 3:
                    symptom_list += ", ..."
                
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
    
    return f"""
    <div class="section entities">
        <h5>Entities</h5>
        <div class="entity-container">
            {entity_items}
        </div>
    </div>
    """

def generate_management_plans_html(management_plans: dict) -> str:
    if not management_plans:
        return ""
    
    plans_html = "".join(
        f'<div class="management-plan"><strong>{disease}:</strong> {plan}</div>'
        for disease, plan in management_plans.items()
    )
    return f"""
    <div class="section management">
        <h5>Management Plans</h5>
        {plans_html}
    </div>
    """

def generate_html_response(data: Dict[str, any], status_code: int = 200) -> str:
    """
    Generate an HTML response for the /process_note endpoint.
    """
    if status_code == 404:
        return f"""
    <div class="container">
        <div class="error-icon">❌</div>
        <h5>Error: Note Not Found</h5>
        <p>{html.escape(data.get('detail', 'The requested note was not found in the database.'))}</p>
    </div>
        """

    # Extract data
    note_id = html.escape(str(data.get("note_id", "Unknown")))
    patient_id = html.escape(data.get("patient_id", "Unknown"))
    primary_diagnosis = data.get("primary_diagnosis", {})
    differential_diagnoses = data.get("differential_diagnoses", [])
    keywords = [html.escape(k) for k in data.get("keywords", [])]
    cuis = [html.escape(c) for c in data.get("cuis", [])]
    entities = data.get("entities", [])
    summary = html.escape(data.get("summary", ""))
    management_plans = data.get("management_plans", {})
    processed_at = html.escape(data.get("processed_at", datetime.now().isoformat()))
    processing_time = data.get("processing_time", 0.0)

    # Generate sections
    primary_html = generate_primary_diagnosis_html(primary_diagnosis)
    differential_html = generate_differential_diagnoses_html(differential_diagnoses)
    entities_html = generate_entities_html(entities)
    management_html = generate_management_plans_html(management_plans)

    return f"""
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

        {primary_html}
        {differential_html}
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
        {management_html}

        <div class="section summary">
            <h5>Summary</h5>
            <p>{summary}</p>
        </div>
    </div>
    """

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
    if not text:
        return ""
    
    start_time = time.time()
    if not doc:
        doc = DiseasePredictor.nlp(text)
    
    # Use optimized clinical term recognition
    clinical_terms = DiseasePredictor.clinical_terms
    
   
    temporal_pattern = re.compile(
        r"\b(\d+\s*(?:day|days|week|weeks|month|months|year|years)\s*(?:ago)?)\b", 
        re.IGNORECASE
    )
    
    # Extract key information
    symptoms = set()
    temporal_info = ""
    aggravating_factors = defaultdict(list)
    alleviating_factors = defaultdict(list)
    medications = []
    findings = []
    
    # Pre-calculate term positions
    term_positions = {}
    text_lower = text.lower()
    for term in clinical_terms:
        start = text_lower.find(term)
        if start != -1:
            term_positions[term] = (start, start + len(term))
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        sent_lower = sent_text.lower()
        
        # Extract temporal information
        if not temporal_info:
            match = temporal_pattern.search(sent_text)
            if match:
                temporal_info = match.group(0)
        
        # Extract symptoms with context using pre-calculated positions
        for term in clinical_terms:
            if term in term_positions:
                start, end = term_positions[term]
                context = text[max(0, start-20):min(len(text), end+20)]
                
                symptoms.add(term)
                
                # Extract modifying phrases
                if "makes" in context or "worsen" in context:
                    aggravating_factors[term].append(context.split(term)[-1].split(".")[0].strip())
                elif "alleviates" in context or "relieves" in context or "better" in context:
                    alleviating_factors[term].append(context.split(term)[-1].split(".")[0].strip())
        
        # Extract medications and findings
        if "mg" in sent_lower or "take" in sent_lower or "medication" in sent_lower:
            medications.append(sent_text)
        if "found" in sent_lower or "show" in sent_lower or "reveal" in sent_lower or "report" in sent_lower:
            findings.append(sent_text)
    
    # Construct natural summary
    summary_parts = []
    
    # Patient presentation
    if symptoms:
        symptoms_list = sorted(symptoms)
        if len(symptoms_list) > 1:
            symptoms_str = ", ".join(symptoms_list[:-1]) + " and " + symptoms_list[-1]
        else:
            symptoms_str = symptoms_list[0]
            
        presentation = f"The patient presents with {symptoms_str}"
        if temporal_info:
            presentation += f" that started {temporal_info}"
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
        summary_parts.append("Medications: " + ". ".join(medications[:2]))
    
    # Clinical findings
    if findings:
        summary_parts.append("Findings: " + ". ".join(findings[:2]))
    
    # Ensure we don't exceed max sentences
    summary = " ".join(summary_parts[:max_sentences])
    
    # Final cleanup
    summary = re.sub(r"\s+", " ", summary)
    summary = re.sub(r"\.+", ".", summary)
    summary = summary.strip()
    
    # Capitalize first letter
    if summary and summary[0].islower():
        summary = summary[0].upper() + summary[1:]
    
    logger.debug(f"Summary generation took {time.time() - start_time:.3f} seconds")
    return summary if summary else text[:200] + "..."

def update_ai_analysis(note_id: int, ai_analysis_html: str, summary: str) -> bool:
    """
    Update AI analysis and summary for a specific note in the soap_notes table.
    """
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
            logger.info(f"Updated AI analysis and summary for note ID {note_id}")
            return True
    except sqlite3.Error as e:
        logger.error(f"Database error updating note {note_id}: {e}")
        return False
    except Exception as e:
        logger.critical(f"Unexpected error updating note {note_id}: {e}", exc_info=True)
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
        
        # Precompute temporal matches
        temporal_patterns = [
            (re.compile(r"\b(\d+\s*(day|days|week|weeks|month|months|year|years)\s*(ago)?)\b", re.IGNORECASE), "DURATION"),
            (re.compile(r"\b(since|for)\s*(\d+\s*(day|days|week|weeks|month|months|year|years))\b", re.IGNORECASE), "DURATION"),
            (re.compile(r"\b\d+\s*days?\b", re.IGNORECASE), "DURATION"),
            (re.compile(r"\bfor\s*(three|four|five|six|seven|eight|nine|ten|[1-9]\d*)\s*days?\b", re.IGNORECASE), "DURATION")
        ]
        temporal_matches = []
        for pattern, label in temporal_patterns:
            temporal_matches.extend(match.group() for match in pattern.finditer(text))
        
        # Create single regex for clinical terms
        clinical_terms = DiseasePredictor.clinical_terms
        if clinical_terms:
            terms_regex = re.compile(
                r'\b(' + '|'.join(map(re.escape, sorted(clinical_terms, key=len, reverse=True))) + r')\b', 
                re.IGNORECASE
            )
            term_matches = {match.group().lower() for match in terms_regex.finditer(text)}
        else:
            term_matches = set()
        
        entities = []
        seen_entities = set()
        
        # Process clinical terms
        for term in term_matches:
            if term not in seen_entities:
                context = {"severity": 1, "temporal": "UNSPECIFIED"}
                for temp_text in temporal_matches:
                    if temp_text.lower() in text.lower():
                        context["temporal"] = temp_text.lower()
                        break
                entities.append((term, "CLINICAL_TERM", context))
                seen_entities.add(term)
        
        # Process patterns
        for label, pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
                match_text = match.group().lower()
                if match_text not in seen_entities:
                    context = {"severity": 1, "temporal": "UNSPECIFIED"}
                    for temp_text in temporal_matches:
                        if temp_text.lower() in text.lower():
                            context["temporal"] = temp_text.lower()
                            break
                    entities.append((match.group(), label, context))
                    seen_entities.add(match_text)
        
        # Associate diseases with symptoms
        symptom_disease_map = defaultdict(set)
        disease_symptom_count = defaultdict(int)
        disease_symptom_map = defaultdict(set)
        
        for i, (entity_text, entity_label, context) in enumerate(entities):
            if entity_label in ["CLINICAL_TERM"] + [label for label, _ in self.compiled_patterns]:
                try:
                    with get_sqlite_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT cui FROM symptoms WHERE name = ?", (entity_text.lower(),))
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
        
        # Add disease associations
        for i, (entity_text, entity_label, context) in enumerate(entities):
            if entity_label in ["CLINICAL_TERM"] + [label for label, _ in self.compiled_patterns]:
                diseases = symptom_disease_map.get(entity_text.lower(), set())
                filtered_diseases = [
                    d for d in diseases 
                    if disease_symptom_count.get(d, 0) >= 2
                ]
                if filtered_diseases:
                    context["associated_diseases"] = filtered_diseases
                    context["disease_symptom_map"] = disease_symptom_map
                    entities[i] = (entity_text, entity_label, context)
        
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
                # Check disease keywords
                for keyword, cui in disease_keywords.items():
                    if keyword in term or term in keyword:
                        if keyword not in expected_keywords:
                            expected_keywords.append(keyword)
                            reference_cuis.append(cui)
                        break
                # Check symptom CUIs
                if term in symptom_cuis and term not in expected_keywords:
                    expected_keywords.append(term)
                    if symptom_cuis[term]:
                        reference_cuis.append(symptom_cuis[term])
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
        
        # Lazy load heavy resources
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

# FastAPI Application with Rate Limiting
app = FastAPI(
    title="Clinical NLP API",
    description="Real-time clinical NLP processing service",
    version="1.0.0"
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
@limiter.limit(HIMS_CONFIG["RATE_LIMIT"])
async def predict(request: Request, payload: PredictionRequest):
    start_time = time.time()
    predictions = text_predictor.predict_from_text(payload.text)
    return PredictionResponse(
        primary_diagnosis=predictions["primary_diagnosis"],
        differential_diagnoses=predictions["differential_diagnoses"],
        processing_time=time.time() - start_time
    )

@app.post("/process_note")
@limiter.limit(HIMS_CONFIG["RATE_LIMIT"])
async def process_note(request: Request, payload: ProcessNoteRequest):
    start_time = time.time()
    note = fetch_single_soap_note(payload.note_id)
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

# Enhanced CLI with Parallel Processing
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
        process_parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
        
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
            self._process_notes(args.note_id, args.all, args.limit, args.latest, args.parallel)
        elif args.command == 'test':
            self._run_tests()
    
    def _show_status(self, detail=False):
        status = {
            "NLP Model": "Loaded" if DiseasePredictor.nlp else "Error",
            "SQLite Database": HIMS_CONFIG["SQLITE_DB_PATH"],
            "UMLS Connection": HIMS_CONFIG["UMLS_DB_URL"],
            "API Endpoint": f"http://{HIMS_CONFIG['API_HOST']}:{HIMS_CONFIG['API_PORT']}",
            "Rate Limit": HIMS_CONFIG["RATE_LIMIT"],
            "Max Workers": HIMS_CONFIG["MAX_WORKERS"]
        }
        
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM soap_notes")
                note_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM soap_notes WHERE ai_analysis IS NOT NULL")
                processed_count = cursor.fetchone()[0]
                status["SOAP Notes"] = f"{processed_count}/{note_count} processed"
                
                cursor.execute("SELECT COUNT(*) FROM diseases")
                disease_count = cursor.fetchone()[0]
                status["Diseases"] = disease_count
                
                cursor.execute("SELECT COUNT(*) FROM symptoms")
                symptom_count = cursor.fetchone()[0]
                status["Symptoms"] = symptom_count
        except:
            status["SOAP Notes"] = "Unknown"
        
        table = Table(title="System Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        for k, v in status.items():
            table.add_row(k, str(v))
        
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
    
    def _process_single_note(self, note_id):
        """Process a single note and return result"""
        processor = DiseasePredictor()
        note = fetch_single_soap_note(note_id)
        if note:
            result = processor.process_soap_note(note)
            html_content = generate_html_response(result, 200)
            update_success = update_ai_analysis(note["id"], html_content, result['summary'])
            return update_success
        return False
    
    def _process_notes(self, note_id, process_all, limit, latest=False, parallel=False):
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
            success = self._process_single_note(note_id)
            if success:
                console.print(f"[green]Successfully processed note {note_id}[/green]")
            else:
                console.print(f"[red]Failed to process note {note_id}[/red]")
        elif process_all:
            console.print(Panel("Processing All Notes", style="bold green"))
            notes = fetch_soap_notes()
            if not notes:
                console.print("[yellow]No notes found in database[/yellow]")
                return
                
            notes_to_process = notes[:limit]
            note_ids = [note['id'] for note in notes_to_process]
            
            if parallel:
                console.print(f"[cyan]Using parallel processing with {HIMS_CONFIG['MAX_WORKERS']} workers[/cyan]")
                with concurrent.futures.ProcessPoolExecutor(max_workers=HIMS_CONFIG["MAX_WORKERS"]) as executor:
                    results = list(track(
                        executor.map(self._process_single_note, note_ids),
                        total=len(note_ids),
                        description="Processing..."
                    ))
                    success_count = sum(results)
            else:
                success_count = 0
                for note in track(notes_to_process, description="Processing..."):
                    try:
                        if self._process_single_note(note['id']):
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

# Main Execution
if __name__ == '__main__':
    nltk.download('wordnet', quiet=True)
    DiseasePredictor.initialize()
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        HIMSCLI().run()
    else:
        start_server()