import unittest
import logging
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import psycopg2
from psycopg2.extras import DictCursor, RealDictCursor
import sqlite3
import re
import time
import json
import os
import argparse
import subprocess
import sys
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple, Union
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.panel import Panel
from rich.progress import track
import requests
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
from dotenv import load_dotenv
from datetime import datetime

# Initialize logging first
logger = logging.getLogger("HIMS-NLP")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nlp_service.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

# Configuration - Centralized and simplified
HIMS_CONFIG = {
    "DEFAULT_DEPARTMENT": "emergency",
    "PRIORITY_SYMPTOMS": ["chest pain", "shortness of breath", "severe headache"],
    "UMLS_THRESHOLD": 0.7,
    "CLINICAL_TERMS_TABLE": "clinical_terms",
    "SQLITE_DB_PATH": "/home/mathu/projects/hospital/instance/hims.db",
    "API_HOST": "0.0.0.0",
    "API_PORT": 8000,
    "POSTGRES_HOST": os.getenv("POSTGRES_HOST", "localhost"),
    "POSTGRES_PORT": os.getenv("POSTGRES_PORT", "5432"),
    "POSTGRES_DB": os.getenv("POSTGRES_DB", "hospital_umls"),
    "POSTGRES_USER": os.getenv("POSTGRES_USER", "postgres"),
    "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD", "postgres"),
    "BATCH_SIZE": 50
}

console = Console()

# Initialize PostgreSQL connection pool
try:
    postgres_pool = SimpleConnectionPool(
        minconn=5,
        maxconn=20,
        host=HIMS_CONFIG["POSTGRES_HOST"],
        port=HIMS_CONFIG["POSTGRES_PORT"],
        dbname=HIMS_CONFIG["POSTGRES_DB"],
        user=HIMS_CONFIG["POSTGRES_USER"],
        password=HIMS_CONFIG["POSTGRES_PASSWORD"],
        cursor_factory=RealDictCursor,
        connect_timeout=10
    )
    logger.info("PostgreSQL connection pool initialized")
except Exception as e:
    logger.error(f"Failed to initialize PostgreSQL pool: {e}")
    postgres_pool = None

@contextmanager
def get_postgres_connection(readonly=False):
    """Context manager for PostgreSQL connections"""
    conn = None
    try:
        conn = postgres_pool.getconn()
        conn.set_session(readonly=readonly, autocommit=False)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        yield cursor
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            cursor.close()
            postgres_pool.putconn(conn)

@contextmanager
def get_sqlite_connection():
    """Context manager for SQLite connections"""
    conn = sqlite3.connect(HIMS_CONFIG["SQLITE_DB_PATH"])
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# ===================== SOAP Note Processing Functions =====================
def fetch_soap_notes(limit: int = None) -> List[Dict]:
    """Fetches SOAP notes from the SQLite database"""
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
    """Fetches a single SOAP note by ID"""
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
    """Combines relevant fields from a SOAP note for NLP processing"""
    # Prioritize symptoms field if available
    symptoms = soap_note.get('symptoms', '') or ''
    
    # Combine all relevant fields
    fields = [
        soap_note.get('situation', ''),
        soap_note.get('hpi', ''),
        symptoms,  # Use symptoms field
        soap_note.get('aggravating_factors', ''),
        soap_note.get('alleviating_factors', ''),
        soap_note.get('medical_history', ''),
        soap_note.get('medication_history', ''),
        soap_note.get('assessment', ''),
        soap_note.get('recommendation', ''),
        soap_note.get('additional_notes', ''),
        soap_note.get('ai_notes', '')  # Include any existing AI notes
    ]
    return ' '.join(filter(None, fields)).strip()

# NEW: Add function to generate AI summaries
# Update the generate_summary function
def generate_summary(text: str, max_sentences: int = 3) -> str:
    """Generate extractive summary using available SpaCy models"""
    if not text:
        return ""
    
    try:
        # Try to use the clinical model if available
        try:
            nlp = spacy.load("en_core_sci_sm")
            logger.info("Using en_core_sci_sm model for summarization")
        except OSError:
            # Fallback to English model
            logger.warning("en_core_sci_sm not available, using en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        
        doc = nlp(text)
        
        # Score sentences by clinical relevance
        sentence_scores = []
        for i, sent in enumerate(doc.sents):
            score = 0
            # Prioritize sentences with clinical terms
            for token in sent:
                if token.text.lower() in ClinicalNER().clinical_terms:  # Use our clinical terms list
                    score += 2
            # Prioritize sentences with verbs (actions)
            score += len([token for token in sent if token.pos_ in ["VERB", "AUX"]])
            # Prioritize beginning of document
            score += 1 - (i / max(len(list(doc.sents)), 1))
            sentence_scores.append((sent, score))
        
        # Sort by score and select top sentences
        sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
        top_sentences = [str(sent[0]) for sent in sorted_sentences[:max_sentences]]
        
        return " ".join(top_sentences)
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        # Robust fallback: return key sentences from start/middle/end
        sentences = text.split('.')
        if len(sentences) > 3:
            return ". ".join([
                sentences[0].strip(),
                sentences[len(sentences)//2].strip(),
                sentences[-2].strip()
            ]) + "."
        return text[:200] + "..." if len(text) > 200 else text

# UPDATED: Modified to handle summary
def update_ai_analysis(note_id: int, analysis: Dict, summary: str):
    """Updates the AI analysis and summary fields in the database"""
    try:
        with get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE soap_notes SET ai_analysis = ?, ai_notes = ? WHERE id = ?",
                (json.dumps(analysis), summary, note_id)
            )
            conn.commit()
            logger.info(f"Updated AI analysis and summary for note ID {note_id}")
    except sqlite3.Error as e:
        logger.error(f"Error updating AI analysis: {e}")

# ===================== Enhanced NLP Components =====================
class ClinicalNER:
    """Clinical Named Entity Recognition with caching and error handling"""
    def __init__(self, model_name="en_core_sci_sm"):
        try:
            self.nlp = spacy.load(model_name, disable=["lemmatizer"])
            logger.info(f"Loaded SpaCy model: {model_name}")
        except OSError:
            logger.warning("SpaCy model not found, using blank English model")
            self.nlp = spacy.blank("en")
        
        self.negation_terms = {"no", "not", "denies", "without", "absent", "negative"}
        self.clinical_terms = self._load_clinical_terms()
        self.lemmatizer = WordNetLemmatizer()
        
    def _load_clinical_terms(self):
        """Load clinical terms from database"""
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT term FROM {HIMS_CONFIG['CLINICAL_TERMS_TABLE']}"
                )
                return {row['term'].lower() for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Error loading clinical terms: {e}")
            return {
                "headache", "fever", "cough", "pain", "nausea", 
                "vomiting", "dizziness", "rash", "fatigue", "chest pain",
                "shortness of breath", "abdominal pain", "diarrhea"
            }
    
    def extract_entities(self, text: str) -> List[Tuple[str, str, dict]]:
        """Extract clinical entities from text"""
        if not text:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        # Process noun chunks as potential entities
        for chunk in doc.noun_chunks:
            if chunk.text.lower() in self.clinical_terms:
                entities.append((
                    chunk.text,
                    "CLINICAL_TERM",
                    {"severity": 1, "temporal": "UNSPECIFIED"}
                ))
        
        # Add custom pattern matching
        patterns = {
            "PAIN": r"\b(pain|ache|discomfort|tenderness)\b",
            "FEVER": r"\b(fever|pyrexia|hyperthermia)\b",
            "RESPIRATORY": r"\b(cough|dyspnea|shortness of breath|wheez)\b",
            "CARDIO": r"\b(chest pain|palpitation|tachycardia)\b",
            "GASTRO": r"\b(nausea|vomiting|diarrhea|constipation|abdominal pain)\b",
            "NEURO": r"\b(headache|dizziness|vertigo|confusion|seizure)\b"
        }
        
        for label, pattern in patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append((
                    match.group(),
                    label,
                    {"severity": 1, "temporal": "UNSPECIFIED"}
                ))
        
        return entities
    
    def extract_keywords_and_cuis(self, note: Dict) -> Tuple[List[str], List[str]]:
        """Extracts expected keywords and reference CUIs from the note"""
        # Prioritize symptoms field if available
        text = note.get('symptoms', '') or note.get('assessment', '') or ''
        if not text:
            return [], []
        
        entities = self.extract_entities(text)
        terms = {ent[0].lower() for ent in entities if ent}
        
        # Disease keyword mapping
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
            "asthma": "C0004096",
            "myocardial infarction": "C0027051",
            "stroke": "C0038454",
            "diabetes": "C0011849",
            "hypertension": "C0020538"
        }
        
        expected_keywords = []
        reference_cuis = []
        
        # Match terms to disease keywords
        for term in terms:
            for keyword, cui in disease_keywords.items():
                if keyword in term or term in keyword:
                    expected_keywords.append(keyword)
                    reference_cuis.append(cui)
                    break
        
        # If no disease-specific keywords found, use symptom-based fallback
        if not expected_keywords:
            symptom_cuis = {
                "fever": "C0015967",
                "pain": "C0030193",
                "headache": "C0018681",
                "cough": "C0010200",
                "vomiting": "C0042963",
                "diarrhea": "C0011991"
            }
            for term in terms:
                if term in symptom_cuis:
                    expected_keywords.append(term)
                    reference_cuis.append(symptom_cuis[term])
        
        # Final fallback to common symptoms
        if not expected_keywords:
            expected_keywords = ["fever", "pain", "headache", "cough"]
            reference_cuis = []
        
        return expected_keywords, reference_cuis

class UMLSMapper:
    """UMLS Concept Mapper with caching and batch operations"""
    def __init__(self):
        self.cui_cache = {}
        self.term_cache = {}
        self.semantic_cache = {}
        
    def map_term_to_cui(self, term: str) -> List[str]:
        """Map a clinical term to UMLS CUIs"""
        if term in self.term_cache:
            return self.term_cache[term]
        
        cuis = []
        try:
            with get_postgres_connection(readonly=True) as cursor:
                cursor.execute("""
                    SELECT DISTINCT cui
                    FROM umls.mrconso
                    WHERE LOWER(str) = LOWER(%s)
                    AND lat = 'ENG'
                    AND suppress = 'N'
                    LIMIT 5
                """, (term,))
                cuis = [row['cui'] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"CUI mapping failed for {term}: {e}")
        
        self.term_cache[term] = cuis
        return cuis
    
    def resolve_cui(self, cui: str) -> str:
        """Resolve a CUI to its current version"""
        if cui in self.cui_cache:
            return self.cui_cache[cui]
        
        try:
            with get_postgres_connection(readonly=True) as cursor:
                cursor.execute("""
                    WITH RECURSIVE merge_chain AS (
                        SELECT pcui, cui, 1 AS depth
                        FROM umls.mergedcui 
                        WHERE pcui = %s
                        UNION
                        SELECT m.pcui, m.cui, mc.depth + 1
                        FROM umls.mergedcui m
                        JOIN merge_chain mc ON m.pcui = mc.cui
                        WHERE mc.depth < 5 AND m.pcui != m.cui
                    )
                    SELECT DISTINCT ON (pcui) pcui, cui 
                    FROM merge_chain
                    ORDER BY pcui, depth DESC 
                """, (cui,))
                result = cursor.fetchone()
                resolved_cui = result['cui'] if result else cui
                self.cui_cache[cui] = resolved_cui
                return resolved_cui
        except Exception as e:
            logger.error(f"CUI resolution failed for {cui}: {e}")
            return cui
    
    def is_infectious_disease(self, cui: str) -> bool:
        """Check if a CUI represents an infectious disease"""
        cache_key = f"infectious_{cui}"
        if cache_key in self.semantic_cache:
            return self.semantic_cache[cache_key]
        
        try:
            with get_postgres_connection(readonly=True) as cursor:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1
                        FROM umls.mrsty 
                        WHERE cui = %s
                        AND sty IN (
                            'Bacterial Infectious Disease', 'Viral Infectious Disease',
                            'Parasitic Infectious Disease', 'Fungal Infectious Disease',
                            'Infectious Disease'
                        )
                    ) AS is_infectious
                """, (cui,))
                result = cursor.fetchone()
                is_infectious = result['is_infectious'] if result else False
                self.semantic_cache[cache_key] = is_infectious
                return is_infectious
        except Exception as e:
            logger.error(f"Semantic check failed for {cui}: {e}")
            return False

# ===================== Enhanced Disease Prediction =====================
class DiseasePredictor:
    """Predict diseases based on clinical text"""
    def __init__(self, ner_model=None):
        self.ner = ner_model or ClinicalNER()
        self.umls_mapper = UMLSMapper()
        self.disease_signatures = {
            "pneumonia": {"cough", "fever", "chest pain"},
            "myocardial_infarction": {"chest pain", "shortness of breath", "sweating"},
            "stroke": {"headache", "weakness", "speech difficulty", "facial droop"},
            "gastroenteritis": {"diarrhea", "nausea", "vomiting", "abdominal pain"},
            "uti": {"dysuria", "frequency", "urgency", "suprapubic pain"},
            "asthma": {"wheezing", "shortness of breath", "cough"},
            "diabetes": {"thirst", "polyuria", "fatigue", "blurred vision"},
            "hypertension": {"headache", "dizziness", "nosebleed"}
        }
    
    def predict_from_text(self, text: str) -> List[Dict]:
        """Predict top diseases based on clinical text"""
        entities = self.ner.extract_entities(text)
        if not entities:
            return []
        
        # Collect unique terms
        terms = {ent[0].lower() for ent in entities}
        
        # Map terms to CUIs
        term_cui_map = {}
        for term in terms:
            term_cui_map[term] = self.umls_mapper.map_term_to_cui(term)
        
        # Calculate disease scores
        disease_scores = defaultdict(float)
        for disease, signature in self.disease_signatures.items():
            score = sum(1 for term in signature if term in terms)
            disease_scores[disease] = score
        
        # Convert to sorted results
        return sorted(
            [{"disease": k, "score": v} for k, v in disease_scores.items()],
            key=lambda x: x["score"],
            reverse=True
        )[:5]
    
    # UPDATED: Added summary generation and database update
    def process_soap_note(self, note: Dict) -> Dict:
        """Full processing pipeline for a SOAP note"""
        # 1. Prepare text
        text = prepare_note_for_nlp(note)
        if not text:
            return {"error": "No valid text in note"}
        
        # 2. Generate AI summary
        summary = generate_summary(text)
        
        # 3. Extract keywords and CUIs
        expected_keywords, reference_cuis = self.ner.extract_keywords_and_cuis(note)
        
        # 4. Extract entities
        entities = self.ner.extract_entities(text)
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
        
        # 5. Map terms to CUIs
        symptom_cuis_map = {}
        for term in terms:
            symptom_cuis_map[term] = self.umls_mapper.map_term_to_cui(term)
        
        # 6. Predict diseases
        top_diseases = self.predict_from_text(text)
        
        # 7. Prepare results
        result = {
            "note_id": note['id'],
            "patient_id": note['patient_id'],
            "diseases": top_diseases,
            "keywords": expected_keywords,
            "cuis": reference_cuis,
            "entities": entities,
            "summary": summary,  # NEW: Add summary to results
            "processed_at": datetime.utcnow().isoformat()
        }
        
        # 8. Save to database
        update_ai_analysis(note['id'], result, summary)  # UPDATED: Pass summary
        
        return result

# ===================== FastAPI Application =====================
app = FastAPI(
    title="Clinical NLP API",
    description="Real-time clinical NLP processing service",
    version="1.0.0"
)

# Initialize predictors
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

# FIX: Create specialized context model to handle mixed types
class EntityContext(BaseModel):
    severity: int
    temporal: str

class EntityDetail(BaseModel):
    text: str
    label: str
    context: EntityContext  # Use the specialized context model

class PredictionResponse(BaseModel):
    diseases: List[DiseasePrediction]
    processing_time: float

# UPDATED: Added summary field
class NoteProcessingResponse(BaseModel):
    note_id: int
    patient_id: str
    diseases: List[DiseasePrediction]
    keywords: List[str]
    cuis: List[str]
    entities: List[EntityDetail]  # Now uses the fixed EntityDetail model
    summary: str  # NEW: Summary field
    processing_time: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()
    diseases = text_predictor.predict_from_text(request.text)
    return PredictionResponse(
        diseases=diseases,
        processing_time=time.time() - start_time
    )

@app.post("/process_note", response_model=NoteProcessingResponse)
async def process_note(request: ProcessNoteRequest):
    start_time = time.time()
    note = fetch_single_soap_note(request.note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    
    result = note_processor.process_soap_note(note)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    # Format entities for response using the fixed model
    formatted_entities = []
    for ent in result["entities"]:
        # Create the EntityContext object
        context = EntityContext(
            severity=ent[2]["severity"],
            temporal=ent[2]["temporal"]
        )
        # Create the EntityDetail object
        formatted_entities.append(
            EntityDetail(
                text=ent[0],
                label=ent[1],
                context=context
            )
        )
    
    return NoteProcessingResponse(
        note_id=result["note_id"],
        patient_id=result["patient_id"],
        diseases=result["diseases"],
        keywords=result["keywords"],
        cuis=result["cuis"],
        entities=formatted_entities,
        summary=result["summary"],  # NEW: Include summary
        processing_time=time.time() - start_time
    )

# ===================== Enhanced CLI =====================
class HIMSCLI:
    """Command Line Interface for HIMS NLP"""
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='HIMS Clinical NLP System',
            formatter_class=argparse.RawTextHelpFormatter
        )
        self._setup_commands()
    
    def _setup_commands(self):
        """Configure CLI commands"""
        subparsers = self.parser.add_subparsers(dest='command')
        
        # Status command
        status_parser = subparsers.add_parser('status', help='System status')
        status_parser.add_argument('--detail', action='store_true', help='Detailed status')
        
        # Predict command
        predict_parser = subparsers.add_parser('predict', help='Run prediction')
        predict_parser.add_argument('text', help='Clinical text to analyze')
        
        # Process command
        process_parser = subparsers.add_parser('process', help='Process SOAP notes')
        process_parser.add_argument('--note-id', type=int, help='Process specific note by ID')
        process_parser.add_argument('--all', action='store_true', help='Process all unprocessed notes')
        process_parser.add_argument('--limit', type=int, default=10, help='Limit number of notes to process')
        
        # API command
        api_parser = subparsers.add_parser('api', help='API management')
        api_parser.add_argument('action', choices=['start', 'stop', 'status'])
    
    def run(self):
        """Execute CLI commands"""
        args = self.parser.parse_args()
        
        if not args.command:
            self.parser.print_help()
            return
        
        if args.command == 'status':
            self._show_status(args.detail)
        elif args.command == 'predict':
            self._run_prediction(args.text)
        elif args.command == 'process':
            self._process_notes(args.note_id, args.all, args.limit)
        elif args.command == 'api':
            self._manage_api(args.action)
    
    def _show_status(self, detail=False):
        """Display system status"""
        status = {
            "PostgreSQL": "Connected" if postgres_pool else "Disconnected",
            "NLP Model": "Loaded" if hasattr(ClinicalNER, 'nlp') else "Error",
            "SQLite Database": HIMS_CONFIG["SQLITE_DB_PATH"],
            "UMLS Database": f"{HIMS_CONFIG['POSTGRES_HOST']}:{HIMS_CONFIG['POSTGRES_PORT']}/{HIMS_CONFIG['POSTGRES_DB']}"
        }
        
        # Count SOAP notes
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
            # Show recent notes status
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
        """Run prediction from CLI"""
        console.print(Panel("Clinical Text Analysis", style="bold blue"))
        console.print(f"Input: {text[:200]}...\n")
        
        results = text_predictor.predict_from_text(text)
        
        if not results:
            console.print("[yellow]No diseases predicted[/yellow]")
            return
        
        table = Table(title="Prediction Results")
        table.add_column("Disease", style="magenta")
        table.add_column("Score", style="green")
        for res in results:
            table.add_row(res['disease'], str(res['score']))
        
        console.print(table)
    
    # UPDATED: Added summary display
    def _process_notes(self, note_id, process_all, limit):
        """Process SOAP notes"""
        processor = DiseasePredictor()
        
        if note_id:
            console.print(Panel(f"Processing Note ID: {note_id}", style="bold green"))
            note = fetch_single_soap_note(note_id)
            if note:
                result = processor.process_soap_note(note)
                
                # Print results
                console.print(Panel("Processing Results", style="bold cyan"))
                console.print(f"Note ID: {result['note_id']}")
                console.print(f"Patient ID: {result['patient_id']}")
                
                # NEW: Display summary
                console.print(Panel("AI Summary", style="bold yellow"))
                console.print(result['summary'] + "\n")
                
                if result['diseases']:
                    disease_table = Table(title="Predicted Diseases")
                    disease_table.add_column("Disease", style="magenta")
                    disease_table.add_column("Score", style="green")
                    for disease in result['diseases']:
                        disease_table.add_row(disease['disease'], str(disease['score']))
                    console.print(disease_table)
                else:
                    console.print("[yellow]No diseases predicted[/yellow]")
                
                if result['keywords']:
                    console.print(f"Keywords: {', '.join(result['keywords'])}")
                
                if result['entities']:
                    entity_table = Table(title="Extracted Entities")
                    entity_table.add_column("Text", style="cyan")
                    entity_table.add_column("Label", style="magenta")
                    for ent in result['entities']:
                        entity_table.add_row(ent[0], ent[1])
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
            console.print("[yellow]Specify --note-id or --all to process notes[/yellow]")
    
    def _manage_api(self, action):
        """Manage API server"""
        if action == 'start':
            self._start_api()
        elif action == 'stop':
            self._stop_api()
        elif action == 'status':
            self._api_status()
    
    def _start_api(self):
        """Start the API server"""
        console.print("Starting API server...")
        try:
            subprocess.Popen(
                ["uvicorn", "main:app", "--host", HIMS_CONFIG["API_HOST"], "--port", str(HIMS_CONFIG["API_PORT"])],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            time.sleep(2)  # Give server time to start
            console.print(f"[green]API server running at {HIMS_CONFIG['API_HOST']}:{HIMS_CONFIG['API_PORT']}[/green]")
            console.print(f"[yellow]Access documentation: http://{HIMS_CONFIG['API_HOST']}:{HIMS_CONFIG['API_PORT']}/docs[/yellow]")
        except Exception as e:
            console.print(f"[red]Failed to start API: {e}[/red]")
    
    def _stop_api(self):
        """Stop the API server"""
        console.print("Stopping API server...")
        try:
            # Find and kill the Uvicorn process
            result = subprocess.run(
                ["pkill", "-f", "uvicorn main:app"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            if result.returncode == 0:
                console.print("[green]API server stopped[/green]")
            else:
                console.print("[yellow]No running API server found[/yellow]")
        except Exception as e:
            console.print(f"[red]Error stopping API: {e}[/red]")
    
    def _api_status(self):
        """Check API status"""
        try:
            response = requests.get(
                f"http://{HIMS_CONFIG['API_HOST']}:{HIMS_CONFIG['API_PORT']}/docs",
                timeout=2
            )
            status = "[green]RUNNING[/green]" if response.status_code == 200 else "[red]ERROR[/red]"
            console.print(f"API Status: {status}")
            console.print(f"Documentation: http://{HIMS_CONFIG['API_HOST']}:{HIMS_CONFIG['API_PORT']}/docs")
        except:
            console.print("[red]API Status: DOWN[/red]")

# ===================== Main Execution =====================
if __name__ == '__main__':
    # Initialize NLP resources
    nltk.download('wordnet', quiet=True)
    
    # Command line handling
    if len(sys.argv) > 1:
        HIMSCLI().run()
    else:
        # Start API by default if no commands
        console.print(Panel("Starting HIMS NLP API Server", style="bold blue"))
        console.print(f"Host: {HIMS_CONFIG['API_HOST']}")
        console.print(f"Port: {HIMS_CONFIG['API_PORT']}")
        console.print(f"Documentation: http://{HIMS_CONFIG['API_HOST']}:{HIMS_CONFIG['API_PORT']}/docs")
        uvicorn.run(app, host=HIMS_CONFIG["API_HOST"], port=HIMS_CONFIG["API_PORT"])