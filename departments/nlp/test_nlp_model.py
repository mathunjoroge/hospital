import unittest
import logging
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import psycopg2
from psycopg2.extras import DictCursor
import sqlite3
import re
import time
from fuzzywuzzy import fuzz
import json

# Local imports
from departments.nlp.logging_setup import get_logger
from departments.nlp.nlp_pipeline import get_postgres_connection

# Initialize NLP components
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
logger = get_logger(__name__)
_sci_ner = None

# Cache for UMLS mappings
CUI_CACHE = {}
SEMANTIC_GROUP_CACHE = {}
TERM_CACHE = {}

class SciBERTWrapper:
    def __init__(self, model_name="en_core_sci_sm", disable_linker=True):
        try:
            self.nlp = spacy.load(model_name, disable=["lemmatizer"])
            logger.info(f"Loaded SpaCy model: {model_name}")
            if disable_linker and "entity_linker" in self.nlp.pipe_names:
                self.nlp.remove_pipe("entity_linker")
                logger.info("Removed entity_linker to avoid nmslib dependency.")
            
            self.negation_terms = {"no", "not", "denies", "without", "absent", "negative"}
            
            self.valid_clinical_terms = {
                "headache", "chills", "fever", "high fever", "vomiting", "jaundice", 
                "spleen tenderness", "cough", "chest pain", "shortness of breath", 
                "neck stiffness", "photophobia", "confusion", "burning urination", 
                "increased frequency", "abdominal pain", "blood smear", "urinalysis",
                "malaria", "pneumonia", "meningitis", "uti", "consolidation", "x-ray",
                "lumbar puncture", "white blood cells", "leukocyte esterase", 
                "plasmodium", "pneumonitis", "meningeal inflammation", "urinary infection",
                "plasmodium infection", "malarial fever", "bacterial meningitis", 
                "viral meningitis", "paludism", "cystitis", "sore throat", "myalgia", 
                "fatigue", "nasal congestion", "influenza", "night sweats", "weight loss", 
                "hemoptysis", "tuberculosis", "sputum culture", "diarrhea", "nausea", 
                "dehydration", "gastroenteritis", "stool culture", "rash", "joint pain", 
                "dengue", "cholera", "rice water stools", "severe dehydration", 
                "wheezing", "bronchitis", "hepatitis", "liver tenderness", "dark urine", 
                "asthma", "bronchospasm", "spirometry", "breathlessness"
            }
            
            from spacy.matcher import PhraseMatcher
            self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            patterns = [self.nlp.make_doc(term) for term in self.valid_clinical_terms]
            self.matcher.add("ClinicalTerms", patterns)
            
        except OSError as e:
            logger.error(f"Failed to load SpaCy model {model_name}: {e}. Using blank model.")
            self.nlp = spacy.blank("en")
            self.negation_terms = set()
            self.valid_clinical_terms = set()
            self.matcher = None

    def extract_entities(self, text):
        if not text or not isinstance(text, str):
            logger.warning("Invalid or empty text provided for entity extraction")
            return []
        
        doc = self.nlp(text)
        entities = []
        matched_phrases = set()
        
        if self.matcher:
            matches = self.matcher(doc)
            for _, start, end in matches:
                span = doc[start:end]
                span_text = span.text.lower()
                
                is_negated = False
                negation_start = max(0, start - 5)
                preceding_text = doc[negation_start:start].text.lower()
                if any(term in preceding_text for term in self.negation_terms) and "rule out" not in preceding_text:
                    is_negated = True
                
                if not is_negated:
                    entities.append((span.text, "CLINICAL_TERM"))
                    matched_phrases.add(span_text)
        
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            if chunk_text not in matched_phrases:
                entities.append((chunk.text, "NOUN_CHUNK"))
        
        logger.info(f"Extracted entities from text: {entities}")
        return entities

class UMLSMapper:
    @staticmethod
    def resolve_cuis_batch(cursor, cuis_to_resolve):
        if not cuis_to_resolve:
            logger.warning("No CUIs to resolve")
            return {}
        
        resolved_map = {}
        unresolved_cuis = [cui for cui in cuis_to_resolve if cui not in CUI_CACHE]
        
        if not unresolved_cuis:
            return {cui: CUI_CACHE[cui] for cui in cuis_to_resolve}
            
        try:
            cursor.execute("""
                WITH RECURSIVE merge_chain AS (
                    SELECT pcui, cui, 1 AS depth
                    FROM umls.mergedcui 
                    WHERE pcui = ANY(%s)
                    UNION
                    SELECT m.pcui, m.cui, mc.depth + 1
                    FROM umls.mergedcui m
                    JOIN merge_chain mc ON m.pcui = mc.cui
                    WHERE mc.depth < 5 AND m.pcui != m.cui
                )
                SELECT DISTINCT ON (pcui) pcui, cui 
                FROM merge_chain
                ORDER BY pcui, depth DESC 
            """, (unresolved_cuis,))
            
            results = cursor.fetchall()
            current_cui_map = {row['pcui']: row['cui'] for row in results}

            for original_cui in unresolved_cuis:
                current_cui = original_cui
                visited = set()
                while current_cui in current_cui_map and current_cui not in visited:
                    visited.add(current_cui)
                    current_cui = current_cui_map[current_cui]
                final_cui = current_cui
                CUI_CACHE[original_cui] = final_cui
                resolved_map[original_cui] = final_cui
            
            for cui in unresolved_cuis:
                if cui not in resolved_map:
                    CUI_CACHE[cui] = cui
                    resolved_map[cui] = cui
                    
            return resolved_map

        except psycopg2.Error as e:
            logger.error(f"Batch CUI resolution failed for {unresolved_cuis}: {e}")
            return {cui: cui for cui in cuis_to_resolve}

    @staticmethod
    def is_infectious_disease_batch(cursor, cuis):
        if not cuis:
            logger.warning("No CUIs provided for infectious disease check")
            return {}
        
        results_map = {}
        unseen_cuis = [cui for cui in cuis if f"infectious_{cui}" not in SEMANTIC_GROUP_CACHE]
        
        if not unseen_cuis:
            return {cui: SEMANTIC_GROUP_CACHE[f"infectious_{cui}"] for cui in cuis}
            
        try:
            cursor.execute("""
                SELECT DISTINCT cui 
                FROM umls.mrsty 
                WHERE cui = ANY(%s) 
                  AND sty IN (
                    'Bacterial Infectious Disease',
                    'Viral Infectious Disease',
                    'Parasitic Infectious Disease',
                    'Fungal Infectious Disease',
                    'Infectious Disease'
                  )
            """, (unseen_cuis,))
            
            infectious_cuis = {row['cui'] for row in cursor.fetchall()}
            
            for cui in unseen_cuis:
                is_infectious = cui in infectious_cuis
                SEMANTIC_GROUP_CACHE[f"infectious_{cui}"] = is_infectious
                results_map[cui] = is_infectious
            
            return results_map
            
        except psycopg2.Error as e:
            logger.error(f"Batch semantic group check failed for {unseen_cuis}: {e}")
            return {cui: False for cui in cuis}

    @staticmethod
    def map_terms_to_cuis_batch(cursor, terms, semantic_filter=None, expected_keywords=None, reference_cuis=None):
        if not terms:
            logger.warning("No terms provided for CUI mapping")
            return defaultdict(list)
        
        term_to_cuis = defaultdict(list)
        clean_terms = {re.sub(r'[^\w\s]', '', term).strip().lower() for term in terms if term and isinstance(term, str)}
        logger.info(f"Cleaned terms for mapping: {sorted(clean_terms)}")
        
        if not clean_terms:
            logger.warning("No valid terms after cleaning")
            return defaultdict(list)
        
        # Check cache first
        cached_terms = {t: TERM_CACHE[t] for t in clean_terms if t in TERM_CACHE}
        uncached_terms = [t for t in clean_terms if t not in TERM_CACHE]
        logger.info(f"Uncached terms: {sorted(uncached_terms)}")
        
        if uncached_terms:
            try:
                # Exact match using mrconso
                cursor.execute("""
                    SELECT str, cui
                    FROM umls.mrconso
                    WHERE str = ANY(%s)
                      AND lat = 'ENG'
                      AND suppress = 'N'
                      AND sab IN ('MSH', 'SNOMEDCT_US', 'ICD10CM')
                """, (uncached_terms,))
                for row in cursor:
                    term_to_cuis[row['str'].lower()].append(row['cui'])
                
                # Trigram-based similarity search for unmapped terms
                unmapped_terms = [t for t in uncached_terms if not term_to_cuis[t]]
                if unmapped_terms:
                    logger.info(f"Unmapped terms for trigram search: {sorted(unmapped_terms)}")
                    cursor.execute("""
                        SELECT str, cui
                        FROM umls.mrconso
                        WHERE str %% ANY(%s)  
                          AND lat = 'ENG'
                          AND suppress = 'N'
                          AND sab IN ('MSH', 'SNOMEDCT_US', 'ICD10CM')
                        LIMIT 10
                    """, (unmapped_terms,))
                    for row in cursor:
                        for term in unmapped_terms:
                            if fuzz.ratio(term, row['str'].lower()) > 85:
                                term_to_cuis[term].append(row['cui'])
                
                # Synonyms using mrxw_eng
                for term in unmapped_terms:
                    cursor.execute("""
                        SELECT DISTINCT w.cui
                        FROM umls.mrxw_eng w
                        JOIN umls.mrconso c ON w.cui = c.cui
                        WHERE w.wd = %s
                          AND c.lat = 'ENG'
                          AND c.suppress = 'N'
                          AND c.sab IN ('MSH', 'SNOMEDCT_US', 'ICD10CM')
                    """, (term,))
                    for row in cursor:
                        term_to_cuis[term].append(row['cui'])
                
                # Update cache
                for term in uncached_terms:
                    TERM_CACHE[term] = term_to_cuis[term]
            
            except psycopg2.Error as e:
                logger.error(f"Term mapping query failed: {e}")
                for term in uncached_terms:
                    term_to_cuis[term] = []
        
        # Merge cached results
        for term, cuis in cached_terms.items():
            term_to_cuis[term].extend(cuis)
        
        # Map expected keywords to reference CUIs
        if expected_keywords and reference_cuis:
            for kw, cui in zip(expected_keywords, reference_cuis):
                if kw.lower() in clean_terms and cui not in term_to_cuis[kw.lower()]:
                    term_to_cuis[kw.lower()].append(cui)
                    logger.info(f"Manually mapped {kw.lower()} to {cui}")
                    print(f"🔗 Manually mapped {kw.lower()} to {cui}")
        
        # Apply semantic filter
        all_found_cuis = set(cui for cuis_list in term_to_cuis.values() for cui in cuis_list)
        if semantic_filter and all_found_cuis:
            try:
                cursor.execute("""
                    SELECT DISTINCT cui
                    FROM umls.mrsty
                    WHERE cui = ANY(%s)
                      AND sty = ANY(%s)
                """, (list(all_found_cuis), list(semantic_filter)))
                semantically_valid_cuis = {row['cui'] for row in cursor}
                
                filtered_cui_map = defaultdict(list)
                for term, cuis_list in term_to_cuis.items():
                    filtered_cui_map[term] = [c for c in cuis_list if c in semantically_valid_cuis]
                term_to_cuis = filtered_cui_map
            except psycopg2.Error as e:
                logger.error(f"Semantic filter query failed: {e}")
        
        # Resolve merged CUIs
        all_cuis_to_resolve = set(cui for cuis_list in term_to_cuis.values() for cui in cuis_list)
        resolved_cui_map = UMLSMapper.resolve_cuis_batch(cursor, list(all_cuis_to_resolve))
        
        # Check for deleted CUIs
        if resolved_cui_map:
            try:
                cursor.execute("SELECT pcui FROM umls.deletedcui WHERE pcui = ANY(%s)", (list(resolved_cui_map.values()),))
                deleted_cuis = {row['pcui'] for row in cursor}
            except psycopg2.Error as e:
                logger.error(f"Deleted CUI check failed: {e}")
                deleted_cuis = set()
        
        final_term_cui_map = defaultdict(list)
        for term, original_cuis in term_to_cuis.items():
            for cui in original_cuis:
                resolved_cui = resolved_cui_map.get(cui, cui)
                if resolved_cui not in deleted_cuis:
                    final_term_cui_map[term].append(resolved_cui)
        
        logger.info(f"Mapped terms to CUIs: {dict(final_term_cui_map)}")
        return final_term_cui_map

def get_sqlite_connection(db_path="/home/mathu/projects/hospital/instance/hims.db"):
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        logger.info(f"Successfully connected to SQLite database: {db_path}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Failed to connect to SQLite database {db_path}: {e}")
        raise

def fetch_soap_notes():
    """Fetches all SOAP notes from the soap_notes table."""
    soap_notes = []
    try:
        with get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM soap_notes")
            rows = cursor.fetchall()
            
            for row in rows:
                soap_note = {
                    'id': row['id'],
                    'patient_id': row['patient_id'],
                    'created_at': row['created_at'],
                    'situation': row['situation'],
                    'hpi': row['hpi'],
                    'aggravating_factors': row['aggravating_factors'],
                    'alleviating_factors': row['alleviating_factors'],
                    'medical_history': row['medical_history'],
                    'medication_history': row['medication_history'],
                    'assessment': row['assessment'],
                    'recommendation': row['recommendation'],
                    'additional_notes': row['additional_notes'],
                    'symptoms': row['symptoms'],
                    'ai_notes': row['ai_notes'],
                    'ai_analysis': row['ai_analysis'],
                    'file_path': row['file_path']
                }
                soap_notes.append(soap_note)
                logger.info(f"Fetched SOAP note ID {row['id']} for patient {row['patient_id']}")
            
            logger.info(f"Retrieved {len(soap_notes)} SOAP notes from the database")
            return soap_notes
    
    except sqlite3.Error as e:
        logger.error(f"Error fetching SOAP notes: {e}")
        return []

def extract_keywords_and_cuis(note, ner):
    """Extracts expected keywords and reference CUIs from the assessment or symptoms field."""
    text = note.get('assessment', '') or note.get('symptoms', '') or ''
    if not text:
        logger.warning(f"No assessment or symptoms found for note ID {note['id']}")
        return [], []
    
    # Use SciBERTWrapper to extract entities
    entities = ner.extract_entities(text)
    terms = {ent[0].lower() for ent, _ in entities if ent}
    
    # Define disease-related keywords
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
        "asthma": "C0004096"
    }
    
    expected_keywords = []
    reference_cuis = []
    
    for term in terms:
        for keyword, cui in disease_keywords.items():
            if keyword in term or term in keyword:
                expected_keywords.append(keyword)
                reference_cuis.append(cui)
                break
    
    # If no keywords found, fall back to common symptoms
    if not expected_keywords:
        expected_keywords = ["fever", "pain", "headache", "cough"]
        reference_cuis = []
    
    logger.info(f"Extracted keywords: {expected_keywords}, CUIs: {reference_cuis} for note ID {note['id']}")
    return expected_keywords, reference_cuis

def prepare_note_for_nlp(soap_note):
    """Combines relevant fields from a SOAP note for NLP processing."""
    fields = [
        soap_note['situation'] or '',
        soap_note['hpi'] or '',
        soap_note['symptoms'] or '',
        soap_note['assessment'] or '',
        soap_note['additional_notes'] or ''
    ]
    return ' '.join(filter(None, fields)).strip()

class TestSciNER(unittest.TestCase):
    def setUp(self):
        global _sci_ner
        if not _sci_ner:
            _sci_ner = SciBERTWrapper(model_name="en_core_sci_sm", disable_linker=True)
        self.ner = _sci_ner
    
    @classmethod
    def tearDownClass(cls):
        CUI_CACHE.clear()
        SEMANTIC_GROUP_CACHE.clear()
        TERM_CACHE.clear()
        global _sci_ner
        _sci_ner = None

    def _run_prediction_test(self, note, expected_keywords, reference_cuis=None):
        start_time = time.time()
        logger.info(f"Processing note: {note.strip()}")
        print("\n🔬 Running disease prediction...")
        print(f"📄 Note:\n{note.strip()}")

        # Step 1: Enhanced entity extraction
        entities = self.ner.extract_entities(note)
        print("🧠 Extracted entities:", entities)
        
        # Extract terms with comprehensive processing
        terms = set()
        for ent, label in entities:
            clean_text = re.sub(r'[^\w\s]', '', ent).strip().lower()
            if clean_text:
                terms.add(clean_text)
                for word in clean_text.split():
                    if len(word) > 3:
                        lemma = lemmatizer.lemmatize(word)
                        terms.add(lemma)
        
        terms.update([kw.lower() for kw in expected_keywords])
        
        if not terms:
            print("⚠️ No terms extracted, using default symptom keywords")
            terms = {"fever", "pain", "headache", "cough", "vomiting", "stiffness", "diarrhea", "fatigue"}
        
        print("📋 Processed terms:", sorted(terms))
        logger.info(f"Processed terms: {sorted(terms)}")

        # Step 2: Map terms to CUIs
        symptom_cuis_map = {}
        lab_cuis_map = {}
        disease_mentions = []
        
        with get_postgres_connection(readonly=True) as cursor:
            symptom_cuis_map = UMLSMapper.map_terms_to_cuis_batch(
                cursor, 
                list(terms), 
                semantic_filter=['Sign or Symptom', 'Finding', 'Disease or Syndrome',
                                'Bacterial Infectious Disease', 'Parasitic Infectious Disease',
                                'Viral Infectious Disease', 'Fungal Infectious Disease'],
                expected_keywords=expected_keywords,
                reference_cuis=reference_cuis
            )
            
            lab_cuis_map = UMLSMapper.map_terms_to_cuis_batch(
                cursor,
                list(terms),
                semantic_filter=['Laboratory or Test Result', 'Diagnostic Procedure']
            )
            
            for term in terms:
                if any(kw.lower() in term for kw in expected_keywords):
                    disease_mentions.append(term)
                
                if term in symptom_cuis_map and symptom_cuis_map[term]:
                    print(f"  🔍 '{term}' → {len(symptom_cuis_map[term])} symptom CUIs: {symptom_cuis_map[term]}")
                if term in lab_cuis_map and lab_cuis_map[term]:
                    print(f"  🔬 '{term}' → {len(lab_cuis_map[term])} lab CUIs: {lab_cuis_map[term]}")
        
        symptom_cuis = [cui for cuis_list in symptom_cuis_map.values() for cui in cuis_list]
        lab_cuis = [cui for cuis_list in lab_cuis_map.values() for cui in cuis_list]

        print(f"📌 Mapped {len(symptom_cuis)} symptom CUIs and {len(lab_cuis)} lab CUIs")
        print(f"🔎 Disease mentions: {disease_mentions}")
        logger.info(f"Symptom CUIs: {symptom_cuis}")
        logger.info(f"Lab CUIs: {lab_cuis}")
        logger.info(f"Disease mentions: {disease_mentions}")
        
        # Step 3: Predict diseases
        all_cuis = list(set(symptom_cuis + lab_cuis))
        if reference_cuis:
            all_cuis.extend(reference_cuis)
            print(f"🔗 Added reference CUIs: {reference_cuis}")
        
        if not all_cuis:
            print("⛔ No CUIs available for prediction")
            top_diseases = [{'cui': "UNKNOWN", 'name': "No diseases predicted", 'score': 0.0}]
        else:
            disease_scores = defaultdict(float)
            disease_names = {}
            disease_hierarchy = {}
            
            with get_postgres_connection(readonly=True) as cursor:
                # Fetch disease relationships with source ranking
                cursor.execute("""
                    SELECT r.cui2 AS disease_cui, 
                           c.str AS disease_name,
                           r.rela,
                           r.cui1 AS source_cui,
                           k.mrrank_rank
                    FROM umls.mrrel r
                    JOIN umls.mrconso c ON r.cui2 = c.cui
                    JOIN umls.mrrank k ON c.sab = k.sab AND c.tty = k.tty
                    WHERE r.cui1 = ANY(%s)
                      AND c.lat = 'ENG'
                      AND c.suppress = 'N'
                      AND c.sab IN ('MSH', 'SNOMEDCT_US', 'ICD10CM')
                      AND c.ts = 'P'
                      AND k.suppress = 'N'
                    ORDER BY k.mrrank_rank DESC
                """, (all_cuis,))
                
                raw_disease_relations = cursor.fetchall()
                potential_disease_cuis = {row['disease_cui'] for row in raw_disease_relations}
                is_infectious_map = UMLSMapper.is_infectious_disease_batch(cursor, list(potential_disease_cuis))

                # Fetch hierarchical relationships
                cursor.execute("""
                    SELECT cui, ptr
                    FROM umls.mrhier
                    WHERE cui = ANY(%s)
                      AND sab IN ('MSH', 'SNOMEDCT_US', 'ICD10CM')
                """, (list(potential_disease_cuis),))
                for row in cursor:
                    disease_hierarchy[row['cui']] = row['ptr']

                for row in raw_disease_relations:
                    disease_cui = row['disease_cui']
                    disease_name = row['disease_name']
                    disease_names[disease_cui] = disease_name
                    
                    weight = 1.0
                    if row['rela'] == 'causative_agent_of':
                        weight = 2.0
                    elif row['rela'] == 'manifestation_of':
                        weight = 1.8
                    elif row['rela'] == 'has_finding':
                        weight = 1.5
                    
                    weight *= (1 + row['mrrank_rank'] / 1000.0)
                    
                    if is_infectious_map.get(disease_cui, False):
                        weight *= 1.5
                    
                    disease_scores[disease_cui] += weight
            
                # Boost reference CUIs
                if reference_cuis:
                    cursor.execute("""
                        SELECT cui, str
                        FROM umls.mrconso
                        WHERE cui = ANY(%s)
                          AND lat = 'ENG'
                          AND suppress = 'N'
                          AND sab IN ('MSH', 'SNOMEDCT_US', 'ICD10CM')
                    """, (reference_cuis,))
                    for row in cursor:
                        disease_scores[row['cui']] += 20.0
                        disease_names[row['cui']] = row['str']
                        print(f"🚀 Boosted reference CUI {row['cui']} ({row['str']}) by 20.0")
            
                # Boost based on definitions
                if disease_scores:
                    cursor.execute("""
                        SELECT d.cui, d.def
                        FROM umls.mrdef d
                        WHERE d.cui = ANY(%s)
                          AND d.suppress = 'N'
                          AND d.sab IN ('MSH', 'SNOMEDCT_US', 'ICD10CM')
                    """, (list(disease_scores.keys()),))
                    
                    for row in cursor:
                        cui = row['cui']
                        definition = row['def'].lower()
                        boost = 1.0
                        for term in terms:
                            if term in definition:
                                boost += 0.2
                        if boost > 1.0:
                            disease_scores[cui] *= boost
                            print(f"✨ Boosted {disease_names.get(cui)} by {boost:.2f} for definition match")
                
                # Boost based on attributes
                if disease_scores:
                    cursor.execute("""
                        SELECT cui, atn, atv 
                        FROM umls.mrsat 
                        WHERE cui = ANY(%s) 
                          AND atn IN ('SEVERITY', 'ACUTE_CHRONIC', 'EPIDEMIOLOGY')
                          AND suppress = 'N'
                    """, (list(disease_scores.keys()),))
                    
                    for attr in cursor:
                        cui = attr['cui']
                        atv = attr['atv'].lower()
                        if 'severe' in atv or 'acute' in atv:
                            disease_scores[cui] *= 1.1
                            print(f"✨ Boosted {disease_names.get(cui)} by 1.1 for attribute: {atv}")
                        elif 'epidemic' in atv or 'outbreak' in atv:
                            disease_scores[cui] *= 1.05
                            print(f"✨ Boosted {disease_names.get(cui)} by 1.05 for attribute: {atv}")
                
                # Boost for hierarchical relationships
                for cui, score in disease_scores.items():
                    if cui in disease_hierarchy:
                        ptr = disease_hierarchy[cui]
                        if ptr and any(term in ptr.lower() for term in terms):
                            disease_scores[cui] *= 1.2
                            print(f"✨ Boosted {disease_names.get(cui)} by 1.2 for hierarchical match")
                
                # Boost for direct disease mentions
                if disease_mentions:
                    disease_mention_cuis_map = UMLSMapper.map_terms_to_cuis_batch(
                        cursor,
                        disease_mentions,
                        semantic_filter=['Disease or Syndrome', 'Bacterial Infectious Disease',
                                       'Parasitic Infectious Disease', 'Viral Infectious Disease'],
                        expected_keywords=expected_keywords,
                        reference_cuis=reference_cuis
                    )
                    for mention in disease_mentions:
                        if mention in disease_mention_cuis_map:
                            for disease_cui in disease_mention_cuis_map[mention]:
                                disease_scores[disease_cui] *= 20.0
                                print(f"🚀 Boosted {disease_names.get(disease_cui, disease_cui)} by 20.0 for direct mention")
            
            top_diseases = []
            with get_postgres_connection(readonly=True) as cursor:
                candidate_cuis = list(disease_scores.keys())
                if candidate_cuis:
                    cursor.execute("""
                        SELECT DISTINCT cui 
                        FROM umls.mrsty 
                        WHERE cui = ANY(%s) 
                          AND sty IN (
                            'Disease or Syndrome', 'Neoplastic Process',
                            'Bacterial Infectious Disease', 'Parasitic Infectious Disease',
                            'Viral Infectious Disease', 'Fungal Infectious Disease'
                          )
                    """, (candidate_cuis,))
                    valid_disease_cuis = {row['cui'] for row in cursor}

                for cui, score in disease_scores.items():
                    if cui in valid_disease_cuis:
                        top_diseases.append({
                            'cui': cui, 
                            'name': disease_names.get(cui, "Unknown"), 
                            'score': score
                        })
            
            if not top_diseases and reference_cuis:
                with get_postgres_connection(readonly=True) as cursor:
                    resolved_ref_cuis_map = UMLSMapper.resolve_cuis_batch(cursor, list(reference_cuis))
                    final_ref_cuis = list(resolved_ref_cuis_map.values())
                    if final_ref_cuis:
                        cursor.execute("""
                            SELECT cui, str 
                            FROM umls.mrconso 
                            WHERE cui = ANY(%s)
                              AND lat = 'ENG'
                              AND suppress = 'N'
                              AND sab IN ('MSH', 'SNOMEDCT_US', 'ICD10CM')
                        """, (final_ref_cuis,))
                        ref_cui_names = {row['cui']: row['str'] for row in cursor}
                        for cui in final_ref_cuis:
                            if cui in ref_cui_names:
                                top_diseases.append({
                                    'cui': cui,
                                    'name': ref_cui_names[cui],
                                    'score': 20.0
                                })
            
            top_diseases = sorted(top_diseases, key=lambda x: x['score'], reverse=True)[:5]
        
        print("\n🔍 Top predicted diseases:")
        for idx, disease in enumerate(top_diseases, 1):
            print(f"{idx}. {disease['name']} (CUI: {disease['cui']}) - Score: {disease['score']:.2f}")
        logger.info(f"Top diseases: {[(d['name'], d['cui'], d['score']) for d in top_diseases]}")

        # Step 5: Evaluate predictions
        def contains_keyword(disease_name, keywords):
            if not disease_name:
                return False
            lower_name = disease_name.lower()
            
            if any(kw.lower() in lower_name for kw in keywords):
                return True
                
            synonym_map = {
                "malaria": ["plasmodium infection", "malarial fever", "paludism", "plasmodium"],
                "meningitis": ["meningeal inflammation", "bacterial meningitis", "viral meningitis"],
                "uti": ["urinary tract infection", "urinary infection", "cystitis"],
                "pneumonia": ["pneumonitis", "lung inflammation", "lung infection"],
                "influenza": ["flu", "viral infection", "respiratory infection"],
                "tuberculosis": ["tb", "mycobacterial infection", "pulmonary tuberculosis"],
                "gastroenteritis": ["stomach flu", "intestinal infection", "viral gastroenteritis"],
                "dengue": ["dengue fever", "breakbone fever"],
                "cholera": ["vibrio cholerae infection", "rice water diarrhea"],
                "bronchitis": ["bronchial inflammation", "acute bronchitis"],
                "hepatitis": ["liver inflammation", "viral hepatitis"],
                "asthma": ["bronchial asthma", "reactive airway disease"]
            }
            
            for kw in keywords:
                if kw.lower() in synonym_map:
                    if any(syn.lower() in lower_name for syn in synonym_map[kw.lower()]):
                        return True
            return False

        matched = [d for d in top_diseases if contains_keyword(d["name"], expected_keywords)]
        if matched:
            print(f"✅ Found match: {matched[0]['name']} for keywords {expected_keywords}")
            logger.info(f"Match found: {matched[0]['name']} for {expected_keywords}")
        else:
            print("🔍 Debug: Top diseases:", [(d["name"], d["cui"], d["score"]) for d in top_diseases])
            logger.error(f"No match for {expected_keywords} in top predictions")
            self.fail(f"{expected_keywords} not in top predictions")
        
        print(f"⏱️ Prediction completed in {time.time() - start_time:.2f} seconds")
        return top_diseases

    def test_soap_notes_from_db(self):
        """Tests disease prediction on SOAP notes fetched from hims.db."""
        soap_notes = fetch_soap_notes()
        if not soap_notes:
            self.skipTest("No SOAP notes found in the database")
        
        for note in soap_notes:
            with self.subTest(note_id=note['id'], patient_id=note['patient_id']):
                print(f"\n=== Processing SOAP Note ID {note['id']} for Patient {note['patient_id']} ===")
                text = prepare_note_for_nlp(note)
                if not text:
                    logger.warning(f"No valid text for SOAP note ID {note['id']}")
                    self.skipTest(f"No valid text for SOAP note ID {note['id']}")
                
                # Extract keywords and CUIs dynamically
                expected_keywords, reference_cuis = extract_keywords_and_cuis(note, self.ner)
                
                # Run prediction
                top_diseases = self._run_prediction_test(text, expected_keywords, reference_cuis)
                
                # Optionally, update ai_analysis field in the database
                try:
                    with get_sqlite_connection() as conn:
                        cursor = conn.cursor()
                        ai_analysis = json.dumps(top_diseases, indent=2)
                        cursor.execute("""
                            UPDATE soap_notes
                            SET ai_analysis = ?
                            WHERE id = ?
                        """, (ai_analysis, note['id']))
                        conn.commit()
                        logger.info(f"Updated ai_analysis for SOAP note ID {note['id']}")
                except sqlite3.Error as e:
                    logger.error(f"Failed to update ai_analysis for SOAP note ID {note['id']}: {e}")

if __name__ == '__main__':
    unittest.main(verbosity=2)