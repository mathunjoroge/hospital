import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import re
import time
from typing import List, Dict, Optional, Tuple
from sqlalchemy import text
from functools import lru_cache
import bleach
import logging
from cachetools import LRUCache, cached
from concurrent.futures import ThreadPoolExecutor
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf

import numpy as np
from PIL import Image
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
from src.database import get_sqlite_connection, UMLSSession
from src.config import get_config
from resources.common_terms import common_terms
from resources.default_patterns import DEFAULT_PATTERNS
from resources.default_clinical_terms import DEFAULT_CLINICAL_TERMS
from resources.default_disease_keywords import DEFAULT_DISEASE_KEYWORDS
from resources.common_fallbacks import (
    fallback_disease_keywords,
    fallback_symptom_cuis,
    fallback_management_plans,
    COMMON_SYMPTOM_DISEASE_MAP,
    SYMPTOM_NORMALIZATIONS
)

logger = logging.getLogger("HIMS-NLP")
HIMS_CONFIG = get_config()

# Enhanced UMLS relationship types
SYMPTOM_RELATIONSHIPS = [
    'manifestation_of', 
    'has_finding', 
    'has_sign_or_symptom',
    'indication_of',
    'symptom_of',
    'associated_with',
    'finding_site_of',
    'due_to'
]

class UMLSMapper:
    """Maps clinical terms to UMLS CUIs with enhanced caching and normalization."""
    
    def __init__(self):
        self.term_cache = LRUCache(maxsize=10000)
        self.map_terms_to_cuis_batch(common_terms)
        self.symptom_normalizations = self._load_symptom_normalizations()

    @classmethod
    def get_instance(cls) -> 'UMLSMapper':
        """Get a new instance of UMLSMapper."""
        return cls()
    
    def _load_symptom_normalizations(self) -> Dict[str, str]:
        """Load common symptom normalizations."""
        return SYMPTOM_NORMALIZATIONS
    
    def normalize_symptom(self, symptom: str) -> str:
        """Normalize symptom names to common base forms."""
        if not hasattr(self, 'symptom_normalizations') or self.symptom_normalizations is None:
            self.symptom_normalizations = self._load_symptom_normalizations()
        symptom_lower = symptom.lower()
        return self.symptom_normalizations.get(symptom_lower, symptom_lower)
    
    @lru_cache(maxsize=10000)
    def map_term_to_cui(self, term: str) -> List[str]:
        """Map a single term to UMLS CUIs."""
        term = self.normalize_symptom(term)
        if term in self.term_cache:
            return self.term_cache[term]
        
        try:
            with UMLSSession() as session:
                query = text("""
                    SELECT DISTINCT cui
                    FROM umls.mrconso
                    WHERE LOWER(str) = :term
                    AND lat = :language AND suppress = 'N'
                    AND sab IN :trusted_sources
                    LIMIT 1
                """)
                result = session.execute(
                    query,
                    {
                        'term': term,
                        'language': HIMS_CONFIG["UMLS_LANGUAGE"], 
                        'trusted_sources': tuple(HIMS_CONFIG["TRUSTED_SOURCES"])
                    }
                ).fetchone()
                
                cui = [result[0]] if result else []
                self.term_cache[term] = cui
                return cui
        except Exception as e:
            logger.error(f"Error mapping term '{term}' to CUI: {e}")
            return []
    
    def map_terms_to_cuis_batch(self, terms: List[str]) -> Dict[str, List[str]]:
        """Batch map terms to UMLS CUIs with symptom normalization."""
        start_time = time.time()
        if not terms:
            return {}
        normalized_terms = [self.normalize_symptom(t) for t in terms]
        
        # Check cache first
        results = {}
        uncached_terms = []
        
        for term in normalized_terms:
            if term in self.term_cache:
                results[term] = self.term_cache[term]
            else:
                uncached_terms.append(term)
        
        # Process uncached terms in batch
        if uncached_terms:
            try:
                with UMLSSession() as session:
                    query = text("""
                        SELECT LOWER(str) AS term_str, cui
                        FROM umls.mrconso
                        WHERE LOWER(str) IN :terms
                        AND lat = :language AND suppress = 'N'
                        AND sab IN :trusted_sources
                    """)
                    db_results = session.execute(
                        query,
                        {
                            'terms': tuple(uncached_terms), 
                            'language': HIMS_CONFIG["UMLS_LANGUAGE"], 
                            'trusted_sources': tuple(HIMS_CONFIG["TRUSTED_SOURCES"])
                        }
                    ).fetchall()
                    
                    # Group results by term
                    term_map = defaultdict(list)
                    for row in db_results:
                        term_map[row.term_str].append(row.cui)
                    
                    # Update cache and results
                    for term in uncached_terms:
                        cuis = term_map.get(term, [])
                        self.term_cache[term] = cuis
                        results[term] = cuis
            except Exception as e:
                logger.error(f"Error in batch UMLS query: {e}")
                # Fallback to individual lookups
                for term in uncached_terms:
                    results[term] = self.map_term_to_cui(term)
        
        # Map back to original terms
        original_results = {}
        for orig_term, normalized_term in zip(terms, normalized_terms):
            original_results[orig_term] = results.get(normalized_term, [])
        
        logger.debug(f"Batch UMLS mapping took {time.time() - start_time:.3f} seconds")
        return original_results

class DiseaseSymptomMapper:
    """Maps diseases to symptoms and vice versa using UMLS relationships."""
    
    def __init__(self):
        self.cache = LRUCache(maxsize=1000)
        self.umls_mapper = UMLSMapper.get_instance()
    
    @classmethod
    def get_instance(cls) -> 'DiseaseSymptomMapper':
        """Get a new instance of DiseaseSymptomMapper."""
        return cls()
    
    @lru_cache(maxsize=1000)
    def get_disease_symptoms(self, disease_cui: str) -> List[Dict]:
        """Get symptoms associated with a disease CUI from UMLS."""
        try:
            with UMLSSession() as session:
                query = text(f"""
                    SELECT DISTINCT c2.str AS symptom_name, c2.cui AS symptom_cui
                    FROM umls.mrrel r
                    JOIN umls.mrconso c1 ON r.cui1 = c1.cui
                    JOIN umls.mrconso c2 ON r.cui2 = c2.cui
                    WHERE r.cui1 = :disease_cui
                        AND r.rela IN ({', '.join([f"'{rel}'" for rel in SYMPTOM_RELATIONSHIPS])})
                        AND c1.lat = :language AND c1.suppress = 'N'
                        AND c2.lat = :language AND c2.suppress = 'N'
                        AND c1.sab IN :trusted_sources
                        AND c2.sab IN :trusted_sources
                """)
                symptoms = session.execute(
                    query,
                    {
                        'disease_cui': disease_cui,
                        'language': HIMS_CONFIG["UMLS_LANGUAGE"],
                        'trusted_sources': tuple(HIMS_CONFIG["TRUSTED_SOURCES"])
                    }
                ).fetchall()
                
                logger.debug(f"Found {len(symptoms)} symptoms for disease CUI {disease_cui}")
                return [{'name': row.symptom_name, 'cui': row.symptom_cui} for row in symptoms]
        except Exception as e:
            logger.error(f"Error fetching symptoms for disease CUI {disease_cui}: {e}")
            return []
    
    @lru_cache(maxsize=1000)
    def get_symptom_diseases(self, symptom_cui: str) -> List[Dict]:
        """Get diseases associated with a symptom CUI from UMLS."""
        try:
            with UMLSSession() as session:
                query = text(f"""
                    SELECT DISTINCT c1.str AS disease_name, c1.cui AS disease_cui
                    FROM umls.mrrel r
                    JOIN umls.mrconso c1 ON r.cui1 = c1.cui
                    JOIN umls.mrconso c2 ON r.cui2 = c2.cui
                    WHERE r.cui2 = :symptom_cui
                        AND r.rela IN ({', '.join([f"'{rel}'" for rel in SYMPTOM_RELATIONSHIPS])})
                        AND c1.lat = :language AND c1.suppress = 'N'
                        AND c2.lat = :language AND c2.suppress = 'N'
                        AND c1.sab IN :trusted_sources
                        AND c2.sab IN :trusted_sources
                """)
                diseases = session.execute(
                    query,
                    {
                        'symptom_cui': symptom_cui,
                        'language': HIMS_CONFIG["UMLS_LANGUAGE"],
                        'trusted_sources': tuple(HIMS_CONFIG["TRUSTED_SOURCES"])
                    }
                ).fetchall()
                
                logger.debug(f"Found {len(diseases)} diseases for symptom CUI {symptom_cui}")
                return [{'name': row.disease_name, 'cui': row.disease_cui} for row in diseases]
        except Exception as e:
            logger.error(f"Error fetching diseases for symptom CUI {symptom_cui}: {e}")
            return []
    
    def get_symptom_diseases_fallback(self, symptom_text: str) -> List[str]:
        """Fallback method to get diseases for common symptoms when UMLS fails."""
        normalized = self.umls_mapper.normalize_symptom(symptom_text)
        return COMMON_SYMPTOM_DISEASE_MAP.get(normalized, [])
    
    def build_disease_signatures(self) -> Dict[str, set]:
        """Build disease signatures using UMLS relationships with fallback."""
        disease_signatures = defaultdict(set)
        umls_mapper = UMLSMapper.get_instance()
        try:
            # Get all diseases first
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, name FROM diseases")
                diseases = cursor.fetchall()
            
            # Add cancer-specific diseases
            cancer_diseases = {
                'prostate cancer': {'cui': 'C0376358', 'symptoms': {'weight loss', 'fatigue', 'pelvic pain'}},
                'lymphoma': {'cui': 'C0024299', 'symptoms': {'night sweats', 'weight loss', 'fatigue', 'lymphadenopathy'}},
                'leukemia': {'cui': 'C0023418', 'symptoms': {'fatigue', 'weight loss', 'fever', 'easy bruising'}},
                'lung cancer': {'cui': 'C0242379', 'symptoms': {'cough', 'weight loss', 'chest pain', 'hemoptysis'}},
                'colorectal cancer': {'cui': 'C0009402', 'symptoms': {'abdominal pain', 'weight loss', 'rectal bleeding'}},
                'ovarian cancer': {'cui': 'C0029925', 'symptoms': {'abdominal bloating', 'weight loss', 'pelvic pain'}},
                'pancreatic cancer': {'cui': 'C0235974', 'symptoms': {'weight loss', 'jaundice', 'abdominal pain'}},
                'liver cancer': {'cui': 'C2239176', 'symptoms': {'weight loss', 'jaundice', 'abdominal pain'}},
                'breast cancer': {'cui': 'C0006142', 'symptoms': {'breast lump', 'weight loss', 'nipple discharge'}}
            }
            
            # Process diseases in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for disease_id, disease_name in diseases:
                    futures.append(
                        executor.submit(
                            self._process_disease_signature, 
                            disease_id, disease_name, umls_mapper
                        )
                    )
                
                for future in futures:
                    disease_name, symptoms = future.result()
                    if symptoms:
                        disease_signatures[disease_name] = symptoms
            
            # Add cancer-specific signatures
            for disease, data in cancer_diseases.items():
                disease_signatures[disease].update(data['symptoms'])
            
            logger.info(f"Built disease signatures for {len(disease_signatures)} diseases")
            return disease_signatures
        except Exception as e:
            logger.error(f"Error building disease signatures: {e}")
            return {}
    
    def _process_disease_signature(self, disease_id, disease_name, umls_mapper):
        """Process a single disease signature (for parallel execution)."""
        symptoms = set()
        try:
            # Create a new database connection for this thread
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                
                # Get disease CUI from database
                cursor.execute("SELECT cui FROM disease_keywords WHERE disease_id = ? LIMIT 1", (disease_id,))
                row = cursor.fetchone()
                disease_cui = row['cui'] if row else None
                
                # If not found, try to map disease name to CUI
                if not disease_cui:
                    disease_cuis = umls_mapper.map_term_to_cui(disease_name)
                    disease_cui = disease_cuis[0] if disease_cuis else None
            
                # Get symptoms from UMLS
                if disease_cui:
                    symptom_data = self.get_disease_symptoms(disease_cui)
                    for symptom in symptom_data:
                        symptoms.add(symptom['name'].lower())
        
            # If no UMLS results, try to get from disease_symptoms table
            if not symptoms:
                with get_sqlite_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT s.name 
                        FROM disease_symptoms ds
                        JOIN symptoms s ON ds.symptom_id = s.id
                        WHERE ds.disease_id = ?
                    """, (disease_id,))
                    for row in cursor.fetchall():
                        symptoms.add(row['name'].lower())
        except Exception as e:
            logger.error(f"Error processing disease signature for {disease_name}: {e}")
        
        return disease_name, symptoms

class ClinicalNER:
    """Named Entity Recognition for clinical text with enhanced symptom handling."""
    
    @classmethod
    def initialize(cls):
        """Initialize clinical terms if not already loaded."""
        if DiseasePredictor.clinical_terms is None:
            DiseasePredictor.clinical_terms = cls._load_clinical_terms()
    
    @staticmethod
    def _load_clinical_terms() -> set:
        """Load clinical terms from the database."""
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT name FROM symptoms
                    UNION
                    SELECT keyword FROM disease_keywords
                """)
                terms = {row['name'].lower() for row in cursor.fetchall()}
                logger.info(f"Loaded {len(terms)} clinical terms from database")
                return terms
        except Exception as e:
            logger.error(f"Error loading clinical terms: {e}")
            return DEFAULT_CLINICAL_TERMS

    def __init__(self, umls_mapper: UMLSMapper = None):
        self.nlp = DiseasePredictor.nlp
        self.negation_terms = {"no", "not", "denies", "without", "absent", "negative"}
        self.lemmatizer = WordNetLemmatizer()
        self.patterns = self._load_patterns()
        self.compiled_patterns = [(label, re.compile(pattern, re.IGNORECASE)) for label, pattern in self.patterns]
        self.temporal_patterns = [
            (re.compile(r"\b(\d+\s*(day|days|week|weeks|month|months|year|years)\s*(ago)?)\b", re.IGNORECASE), "DURATION"),
            (re.compile(r"\b(since|for)\s*(\d+\s*(day|days|week|weeks|month|months|year|years))\b", re.IGNORECASE), "DURATION"),
            (re.compile(r"\b\d+\s*days?\b", re.IGNORECASE), "DURATION"),
            (re.compile(r"\bfor\s*(three|four|five|six|seven|eight|nine|ten|[1-9]\d*)\s*days?\b", re.IGNORECASE), "DURATION")
        ]
        self.terms_regex = re.compile(
            r'\b(' + '|'.join(map(re.escape, sorted(DiseasePredictor.clinical_terms, key=len, reverse=True))) + r')\b',
            re.IGNORECASE
        ) if DiseasePredictor.clinical_terms else None
        self.symptom_mapper = DiseaseSymptomMapper.get_instance()
        self.umls_mapper = umls_mapper or UMLSMapper.get_instance()
        self.invalid_terms = {'mg', 'ms', 'g', 'ml', 'mm', 'ng', 'dl', 'hr'}  # Filter units and invalid terms
    
    def _load_patterns(self) -> List[Tuple[str, str]]:
        """Load regex patterns from the database."""
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT label, pattern FROM patterns")
                patterns = [(row['label'], row['pattern']) for row in cursor.fetchall()]
                logger.info(f"Loaded {len(patterns)} patterns from database")
                return patterns
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
            return DEFAULT_PATTERNS
    
    def extract_entities(self, text: str, doc=None) -> List[Tuple[str, str, dict]]:
        """Extract clinical entities from text with symptom grouping and cancer-specific focus."""
        start_time = time.time()
        if not doc:
            doc = self.nlp(text)
        
        # Extract temporal matches
        temporal_matches = []
        for pattern, label in self.temporal_patterns:
            temporal_matches.extend(match.group() for match in pattern.finditer(text))
        
        # Extract terms using regex
        term_matches = {match.group().lower() for match in self.terms_regex.finditer(text) 
                        if match.group().lower() not in self.invalid_terms} if self.terms_regex else set()
        
        # Define cancer-specific thresholds for lab results
        lab_thresholds = {
            'psa': {'threshold': 4.0, 'unit': 'ng/mL', 'cancer': 'prostate cancer'},
            'cea': {'threshold': 5.0, 'unit': 'ng/mL', 'cancer': 'colorectal cancer'},
            'ca-125': {'threshold': 35.0, 'unit': 'U/mL', 'cancer': 'ovarian cancer'},
            'ca 19-9': {'threshold': 37.0, 'unit': 'U/mL', 'cancer': 'pancreatic cancer'},
            'afp': {'threshold': 10.0, 'unit': 'ng/mL', 'cancer': 'liver cancer'},
            'wbc': {'threshold': 11000, 'unit': '/mm³', 'cancer': 'leukemia'},
            'hgb': {'threshold': 12.0, 'unit': 'g/dL', 'cancer': 'anemia-related cancer', 'condition': 'low'},
            'crp': {'threshold': 10.0, 'unit': 'mg/L', 'cancer': 'general inflammation'},
            'esr': {'threshold': 20.0, 'unit': 'mm/hr', 'cancer': 'general inflammation'},
        }
        
        # Cancer-specific symptoms for prioritization
        cancer_symptoms = {
            'unexplained weight loss', 'persistent fatigue', 'night sweats', 'persistent cough',
            'palpable lump', 'abnormal bleeding', 'chronic pain', 'hoarseness', 'dysphagia'
        }
        
        # Group similar symptoms
        symptom_groups = defaultdict(set)
        for term in term_matches:
            base_term = self.umls_mapper.normalize_symptom(term)
            if base_term not in self.invalid_terms:
                symptom_groups[base_term].add(term)
        
        # Create entities from grouped symptoms and lab results
        entities = []
        seen_entities = set()
        
        # Process symptom entities
        for base_term, variants in symptom_groups.items():
            if base_term not in seen_entities and base_term not in self.invalid_terms:
                representative = max(variants, key=len)
                context = {
                    "severity": 1.0,
                    "temporal": "UNSPECIFIED",
                    "variants": list(variants),
                    "cancer_relevance": 0.9 if base_term in cancer_symptoms else 0.5
                }
                
                for temp_text in temporal_matches:
                    if temp_text.lower() in text.lower():
                        context["temporal"] = temp_text.lower()
                        break
                
                entities.append((representative, "CLINICAL_TERM", context))
                seen_entities.add(base_term)
        
        # Extract other entities using patterns (including lab results)
        for label, pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
                match_text = match.group().lower()
                normalized = self.umls_mapper.normalize_symptom(match_text)
                
                if normalized not in seen_entities and normalized not in self.invalid_terms:
                    context = {"severity": 1.0, "temporal": "UNSPECIFIED"}
                    for temp_text in temporal_matches:
                        if temp_text.lower() in text.lower():
                            context["temporal"] = temp_text.lower()
                            break
                    
                    # Handle lab results specifically
                    if label in ['TUMOR_MARKER', 'BLOOD_COUNT', 'INFLAMMATORY_MARKER']:
                        try:
                            marker = match.group(1).lower()
                            value = float(match.group(2))
                            unit = match.group(3)
                            if marker not in self.invalid_terms:
                                context.update({"value": value, "unit": unit, "abnormal": False})
                                if marker in lab_thresholds:
                                    threshold_info = lab_thresholds[marker]
                                    is_abnormal = (value > threshold_info['threshold'] if threshold_info.get('condition') != 'low'
                                                else value < threshold_info['threshold'])
                                    if is_abnormal:
                                        context['abnormal'] = True
                                        context['potential_cancer'] = threshold_info['cancer']
                                        context['cancer_relevance'] = 0.95  # High relevance for abnormal labs
                                entities.append((marker, label, context))
                                seen_entities.add(normalized)
                        except (IndexError, ValueError) as e:
                            logger.warning(f"Failed to parse lab result for {match_text}: {e}")
                    else:
                        entities.append((match.group(), label, context))
                        seen_entities.add(normalized)
        
        # Get diseases for each symptom group using parallel processing
        symptom_disease_map = defaultdict(set)
        disease_symptom_count = defaultdict(int)
        disease_symptom_map = defaultdict(set)
        
        symptom_texts = [self.umls_mapper.normalize_symptom(ent[0]) for ent in entities 
                        if ent[0].lower() not in self.invalid_terms]
        
        with ThreadPoolExecutor() as executor:
            futures = {}
            for symptom_text in set(symptom_texts):
                futures[executor.submit(
                    self._get_diseases_for_symptom, 
                    symptom_text
                )] = symptom_text
            
            for future in futures:
                symptom_text = futures[future]
                try:
                    diseases = future.result()
                    symptom_disease_map[symptom_text] = diseases
                    for disease in diseases:
                        disease_symptom_count[disease] += 1
                        disease_symptom_map[disease].add(symptom_text)
                except Exception as e:
                    logger.error(f"Error processing symptom {symptom_text}: {e}")
        
        # Add disease associations to entities, prioritizing cancer-related diseases
        cancer_diseases = {'prostate cancer', 'colorectal cancer', 'ovarian cancer', 'pancreatic cancer',
                        'liver cancer', 'leukemia', 'lung cancer', 'breast cancer', 'lymphoma'}
        
        for i, (entity_text, entity_label, context) in enumerate(entities):
            normalized_text = self.umls_mapper.normalize_symptom(entity_text)
            diseases = symptom_disease_map.get(normalized_text, set())
            
            # Fallback for common symptoms if UMLS returns nothing
            if not diseases and normalized_text in COMMON_SYMPTOM_DISEASE_MAP:
                diseases = set(COMMON_SYMPTOM_DISEASE_MAP[normalized_text])
            
            # Filter diseases, prioritizing cancer-related ones
            filtered_diseases = [d for d in diseases if disease_symptom_count.get(d, 0) >= 2 or d.lower() in cancer_diseases]
            if filtered_diseases:
                context["associated_diseases"] = filtered_diseases
                context["disease_symptom_map"] = {d: list(disease_symptom_map[d]) for d in filtered_diseases}
                entities[i] = (entity_text, entity_label, context)
        
        logger.debug(f"Entity extraction found {len(entities)} entities in {time.time() - start_time:.3f} seconds")
        return entities
    
    def _get_diseases_for_symptom(self, symptom_text):
        """Get diseases for a symptom with enhanced fallbacks."""
        try:
            # First try to get CUI from our database
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT cui FROM symptoms WHERE name = ?", (symptom_text,))
                row = cursor.fetchone()
                symptom_cui = row['cui'] if row else None
            
            # If not found, try UMLS mapping
            if not symptom_cui:
                cuis = self.umls_mapper.map_term_to_cui(symptom_text)
                symptom_cui = cuis[0] if cuis else None
            
            # Get diseases from UMLS
            if symptom_cui:
                diseases = self.symptom_mapper.get_symptom_diseases(symptom_cui)
                return {disease['name'].lower() for disease in diseases}
            
            # Final fallback to hardcoded map
            return set(self.symptom_mapper.get_symptom_diseases_fallback(symptom_text))
        except Exception as e:
            logger.error(f"Error looking up diseases for symptom '{symptom_text}': {e}")
            return set()
    
    def extract_keywords_and_cuis(self, note: Dict) -> Tuple[List[str], List[str]]:
        """Extract keywords and CUIs from a SOAP note."""
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
        terms = {ent[0].lower() for ent in entities if ent and ent[0].lower() not in self.invalid_terms}
        
        expected_keywords = []
        reference_cuis = []
        
        try:
            disease_keywords = DiseasePredictor.disease_keywords
            symptom_cuis = DiseasePredictor.symptom_cuis
            
            for term in terms:
                # Look for disease keywords
                for keyword, cui in disease_keywords.items():
                    if keyword in term or term in keyword:
                        if keyword not in expected_keywords:
                            expected_keywords.append(keyword)
                            reference_cuis.append(cui)
                        break
                
                # Look for symptom CUIs
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
        
        logger.debug(f"Extracted {len(expected_keywords)} keywords and {len(reference_cuis)} CUIs in {time.time() - start_time:.3f} seconds")
        return list(set(expected_keywords)), list(set(reference_cuis))

class DiseasePredictor:
    """Predicts diseases from clinical text using NLP and UMLS mappings."""
    
    nlp = None
    clinical_terms = None
    disease_signatures = None
    disease_keywords = None
    symptom_cuis = None
    management_plans = None
    _initialized = False

    @classmethod
    def initialize(cls, force: bool = False):
        """Initialize shared resources for disease prediction."""
        if cls._initialized and not force:
            return
        
        logger.info("Initializing DiseasePredictor resources...")
        
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
            logger.info(f"Loaded {len(cls.clinical_terms)} clinical terms")
        
        if cls.disease_signatures is None:
            mapper = DiseaseSymptomMapper.get_instance()
            cls.disease_signatures = mapper.build_disease_signatures()
            logger.info(f"Loaded {len(cls.disease_signatures)} disease signatures")
        
        if cls.disease_keywords is None:
            cls.disease_keywords = cls._load_disease_keywords()
            logger.info(f"Loaded {len(cls.disease_keywords)} disease keywords")
            
        if cls.symptom_cuis is None:
            cls.symptom_cuis = cls._load_symptom_cuis()
            logger.info(f"Loaded {len(cls.symptom_cuis)} symptom CUIs")
            
        if cls.management_plans is None:
            cls.management_plans = cls._load_management_plans()
            logger.info(f"Loaded {len(cls.management_plans)} management plans")
        
        cls._initialized = True
        logger.info("DiseasePredictor initialization complete")

    @staticmethod
    def _load_disease_keywords() -> Dict:
        """Lazy load disease keywords."""
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT d.name, dk.keyword, dk.cui
                    FROM diseases d
                    JOIN disease_keywords dk ON d.id = dk.disease_id
                """)
                return {row['keyword'].lower(): row['cui'] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Failed to load disease keywords: {e}")
            return fallback_disease_keywords

    @staticmethod
    def _load_symptom_cuis() -> Dict:
        """Lazy load symptom CUIs."""
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name, cui FROM symptoms")
                return {row['name'].lower(): row['cui'] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Failed to load symptom CUIs: {e}")
            return fallback_symptom_cuis

    @staticmethod
    def _load_management_plans() -> Dict:
        """Lazy load management plans and lab tests for diseases."""
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                
                # Query to fetch management plans
                cursor.execute("""
                    SELECT d.name, dmp.plan
                    FROM disease_management_plans dmp
                    JOIN diseases d ON dmp.disease_id = d.id
                """)
                management_plans = {row['name'].lower(): {'plan': row['plan']} for row in cursor.fetchall()}
                
                # Query to fetch lab tests
                cursor.execute("""
                    SELECT d.name, dl.lab_test, dl.description
                    FROM disease_labs dl
                    JOIN diseases d ON dl.disease_id = d.id
                """)
                lab_tests = cursor.fetchall()
                
                # Combine lab tests with management plans
                for row in lab_tests:
                    disease_name = row['name'].lower()
                    if disease_name not in management_plans:
                        management_plans[disease_name] = {'plan': '', 'lab_tests': []}
                    if 'lab_tests' not in management_plans[disease_name]:
                        management_plans[disease_name]['lab_tests'] = []
                    management_plans[disease_name]['lab_tests'].append({
                        'test': row['lab_test'],
                        'description': row['description'] or ''
                    })
                
                # Add cancer-specific management plans
                cancer_plans = {
                    'prostate cancer': {
                        'plan': 'Refer to oncologist; order prostate biopsy',
                        'lab_tests': [{'test': 'PSA follow-up', 'description': 'Monitor PSA levels in 4-6 weeks'}],
                        'cancer_follow_up': 'Consider imaging (e.g., CT/MRI) and biopsy if indicated.'
                    },
                    'lymphoma': {
                        'plan': 'Order lymph node biopsy; consider PET scan',
                        'lab_tests': [{'test': 'LDH', 'description': 'Assess lymphoma activity'}],
                        'cancer_follow_up': 'Monitor symptoms; consider tumor marker tests.'
                    },
                    'leukemia': {
                        'plan': 'Refer to hematologist; order bone marrow biopsy',
                        'lab_tests': [{'test': 'CBC follow-up', 'description': 'Monitor WBC and other blood counts'}],
                        'cancer_follow_up': 'Consider cytogenetic testing.'
                    },
                    'lung cancer': {
                        'plan': 'Refer to oncologist; order chest CT and biopsy',
                        'lab_tests': [{'test': 'Sputum cytology', 'description': 'Assess for malignant cells'}],
                        'cancer_follow_up': 'Consider PET scan for staging.'
                    },
                    'colorectal cancer': {
                        'plan': 'Refer to oncologist; order colonoscopy and biopsy',
                        'lab_tests': [{'test': 'CEA', 'description': 'Monitor colorectal cancer markers'}],
                        'cancer_follow_up': 'Consider CT abdomen/pelvis.'
                    },
                    'ovarian cancer': {
                        'plan': 'Refer to oncologist; order pelvic ultrasound and biopsy',
                        'lab_tests': [{'test': 'CA-125', 'description': 'Monitor ovarian cancer markers'}],
                        'cancer_follow_up': 'Consider CT/MRI for staging.'
                    },
                    'pancreatic cancer': {
                        'plan': 'Refer to oncologist; order abdominal CT and biopsy',
                        'lab_tests': [{'test': 'CA 19-9', 'description': 'Monitor pancreatic cancer markers'}],
                        'cancer_follow_up': 'Consider endoscopic ultrasound.'
                    },
                    'liver cancer': {
                        'plan': 'Refer to oncologist; order liver ultrasound and biopsy',
                        'lab_tests': [{'test': 'AFP', 'description': 'Monitor liver cancer markers'}],
                        'cancer_follow_up': 'Consider MRI liver.'
                    },
                    'breast cancer': {
                        'plan': 'Refer to oncologist; order mammogram and biopsy',
                        'lab_tests': [{'test': 'BRCA testing', 'description': 'Assess genetic risk'}],
                        'cancer_follow_up': 'Consider breast MRI.'
                    }
                }
                management_plans.update(cancer_plans)
                
                return management_plans
        except Exception as e:
            logger.error(f"Failed to load management plans and lab tests: {e}")
            return fallback_management_plans

    def __init__(self, ner_model=None):
        self.umls_mapper = UMLSMapper.get_instance()
        self.ner = ner_model or ClinicalNER(umls_mapper=self.umls_mapper)
        self.primary_threshold = 2.0
        self.min_symptom_count = 2  # Minimum symptoms to consider a disease
    
    def predict_cancer_risk(self, text: str) -> float:
        """Predict cancer risk using DistilBERT."""
        try:
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            model = DistilBertForSequenceClassification.from_pretrained("distilbert_cancer_model")
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
            return probs[0][1].item()
        except Exception as e:
            logger.error(f"Error in DistilBERT prediction: {e}")
            return 0.0
    
    def predict_image_cancer(self, image_path: str) -> float:
        """Predict cancer risk from an image using MobileNetV2 (TFLite)."""
        try:
            interpreter = tf.lite.Interpreter(model_path="mobilenetv2_cancer.tflite")
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            img = Image.open(image_path).resize((224, 224))
            img = np.array(img, dtype=np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            return output[0][1]
        except Exception as e:
            logger.error(f"Error processing image with MobileNetV2: {e}")
            return 0.0

    def predict_from_text(self, text: str) -> Dict:
        """Predict diseases from clinical text with focus on early cancer detection."""
        start_time = time.time()
        text = bleach.clean(text)
        entities = self.ner.extract_entities(text)
        
        if not entities:
            logger.warning("No entities extracted from text")
            return {
                "primary_diagnosis": None,
                "differential_diagnoses": [],
                "lab_abnormalities": [],
                "cancer_risk_score": 0.0
            }
        
        # Get normalized symptom terms and lab abnormalities
        symptom_terms = set()
        lab_abnormalities = set()
        cancer_relevance_scores = {}
        
        for entity in entities:
            normalized = self.umls_mapper.normalize_symptom(entity[0])
            if normalized not in self.ner.invalid_terms:
                if entity[1] == 'CLINICAL_TERM':
                    symptom_terms.add(normalized)
                    cancer_relevance_scores[normalized] = entity[2].get('cancer_relevance', 0.5)
                elif entity[1] in ['TUMOR_MARKER', 'BLOOD_COUNT', 'INFLAMMATORY_MARKER'] and entity[2].get('abnormal'):
                    lab_abnormalities.add(entity[2]['potential_cancer'])
        
        logger.debug(f"Extracted symptoms: {symptom_terms}, Lab abnormalities: {lab_abnormalities}")
        
        # Define cancer-related diseases for prioritization
        cancer_diseases = {
            'prostate cancer', 'colorectal cancer', 'ovarian cancer', 'pancreatic cancer',
            'liver cancer', 'leukemia', 'lung cancer', 'breast cancer', 'lymphoma'
        }
        
        disease_scores = defaultdict(float)
        
        # Score diseases based on symptom matches and cancer relevance
        for disease, signature in self.disease_signatures.items():
            matches = len(symptom_terms.intersection(signature))
            # Boost score with cancer relevance
            for term in symptom_terms.intersection(signature):
                matches += cancer_relevance_scores.get(term, 0.5)  # Add relevance weight
            # Boost score for lab-confirmed cancers
            if disease.lower() in lab_abnormalities:
                matches += 2.0  # Significant boost for lab abnormalities
            # Lower threshold for cancer diseases to capture early signs
            if matches >= (self.min_symptom_count - 1 if disease.lower() in cancer_diseases else self.min_symptom_count):
                disease_scores[disease] = matches
        
        # Fallback: Match with fewer symptoms if no diseases meet threshold
        if not disease_scores and len(symptom_terms) >= self.min_symptom_count - 1:
            for disease, signature in self.disease_signatures.items():
                matches = len(symptom_terms.intersection(signature))
                for term in symptom_terms.intersection(signature):
                    matches += cancer_relevance_scores.get(term, 0.5)
                if disease.lower() in lab_abnormalities:
                    matches += 2.0
                if matches > 0:
                    disease_scores[disease] = matches
        
        # Sort diseases by score
        sorted_diseases = sorted(
            [{"disease": k, "score": v} for k, v in disease_scores.items()],
            key=lambda x: x["score"],
            reverse=True
        )[:5]
        
        # Determine primary and differential diagnoses
        primary_diagnosis = None
        differential_diagnoses = []
        
        if sorted_diseases:
            # Lower threshold for cancer diseases
            primary_threshold = self.primary_threshold - 0.5 if any(d["disease"].lower() in cancer_diseases for d in sorted_diseases) else self.primary_threshold
            if sorted_diseases[0]["score"] >= primary_threshold:
                primary_diagnosis = sorted_diseases[0]
                differential_diagnoses = sorted_diseases[1:] if len(sorted_diseases) > 1 else []
            else:
                differential_diagnoses = sorted_diseases
        
        # Integrate DistilBERT cancer risk score
        cancer_risk_score = self.predict_cancer_risk(text)
        if cancer_risk_score > 0.7 and not primary_diagnosis:
            primary_diagnosis = {"disease": "Potential Cancer", "score": cancer_risk_score}
        elif cancer_risk_score > 0.7:
            differential_diagnoses.append({"disease": "Potential Cancer", "score": cancer_risk_score})
        
        result = {
            "primary_diagnosis": primary_diagnosis,
            "differential_diagnoses": differential_diagnoses,
            "lab_abnormalities": list(lab_abnormalities),
            "cancer_risk_score": cancer_risk_score
        }
        
        logger.info(f"Predicted {len(sorted_diseases)} diseases from {len(symptom_terms)} symptoms and {len(lab_abnormalities)} lab abnormalities in {time.time() - start_time:.3f} seconds")
        return result
    
    def process_soap_note(self, note: Dict, image_path: Optional[str] = None) -> Dict:
        """Process a SOAP note for disease prediction and analysis with cancer detection focus."""
        from src.utils import prepare_note_for_nlp, generate_summary
        start_time = time.time()
        
        try:
            text = prepare_note_for_nlp(note)
            logger.debug(f"Text preparation took {time.time() - start_time:.3f} seconds")
            if not text:
                return {"error": "No valid text in note"}
            
            t = time.time()
            doc = self.nlp(text)
            logger.debug(f"spaCy processing took {time.time() - t:.3f} seconds")
            
            t = time.time()
            summary = generate_summary(text, soap_note=note, doc=doc)
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
                if clean_text and clean_text not in self.ner.invalid_terms:
                    terms.add(clean_text)
                    for word in clean_text.split():
                        if len(word) > 3:
                            lemma = self.ner.lemmatizer.lemmatize(word)
                            terms.add(lemma)
            
            terms.update([kw.lower() for kw in expected_keywords if kw.lower() not in self.ner.invalid_terms])
            symptom_cuis_map = self.umls_mapper.map_terms_to_cuis_batch(list(terms))
            logger.debug(f"UMLS mapping took {time.time() - t:.3f} seconds")
            
            t = time.time()
            predictions = self.predict_from_text(text)
            logger.debug(f"Prediction took {time.time() - t:.3f} seconds")
            
            # Handle image input with MobileNetV2 (if provided)
            image_cancer_risk = 0.0
            if image_path:
                image_cancer_risk = self.predict_image_cancer(image_path)
                if image_cancer_risk > 0.7 and not predictions["primary_diagnosis"]:
                    predictions["primary_diagnosis"] = {
                        "disease": "Potential Skin Cancer",
                        "score": image_cancer_risk
                    }
                elif image_cancer_risk > 0.7:
                    predictions["differential_diagnoses"].append({
                        "disease": "Potential Skin Cancer",
                        "score": image_cancer_risk
                    })
            
            t = time.time()
            management_plans = {}
            try:
                cancer_diseases = {
                    'prostate cancer', 'colorectal cancer', 'ovarian cancer', 'pancreatic cancer',
                    'liver cancer', 'leukemia', 'lung cancer', 'breast cancer', 'lymphoma'
                }
                # Add management plans for primary diagnosis
                if predictions["primary_diagnosis"]:
                    disease = predictions["primary_diagnosis"]["disease"].lower()
                    if disease in self.management_plans:
                        management_plans[disease] = self.management_plans[disease]
                    # Add cancer-specific follow-ups for high-risk cases
                    if disease in cancer_diseases or predictions["cancer_risk_score"] > 0.7:
                        management_plans[disease] = management_plans.get(disease, {})
                        management_plans[disease]["cancer_follow_up"] = (
                            "Refer to oncologist; consider imaging (e.g., CT/MRI) and biopsy if indicated."
                        )
                
                # Add management plans for differential diagnoses
                for disease in predictions["differential_diagnoses"]:
                    disease_name = disease["disease"].lower()
                    if disease_name in self.management_plans:
                        management_plans[disease_name] = self.management_plans[disease_name]
                    # Add cancer-specific follow-ups for high-risk differentials
                    if disease_name in cancer_diseases:
                        management_plans[disease_name] = management_plans.get(disease_name, {})
                        management_plans[disease_name]["cancer_follow_up"] = (
                            "Monitor symptoms; consider tumor marker tests and imaging."
                        )
                
                # Add lab-specific follow-ups based on abnormalities
                for lab_abnormality in predictions["lab_abnormalities"]:
                    if lab_abnormality in cancer_diseases:
                        management_plans[lab_abnormality] = management_plans.get(lab_abnormality, {})
                        management_plans[lab_abnormality]["lab_follow_up"] = (
                            f"Repeat {lab_abnormality} tumor marker tests in 4-6 weeks; refer to specialist."
                        )
            except Exception as e:
                logger.error(f"Error fetching management plans: {e}")
            logger.debug(f"Management plans retrieval took {time.time() - t:.3f} seconds")
            
            result = {
                "note_id": note["id"],
                "patient_id": note["patient_id"],
                "primary_diagnosis": predictions["primary_diagnosis"],
                "differential_diagnoses": predictions["differential_diagnoses"],
                "lab_abnormalities": predictions["lab_abnormalities"],
                "cancer_risk_score": predictions["cancer_risk_score"],
                "image_cancer_risk": image_cancer_risk,
                "keywords": expected_keywords,
                "cuis": reference_cuis,
                "entities": entities,
                "summary": summary,
                "management_plans": management_plans,
                "processed_at": time.time(),
                "processing_time": time.time() - start_time
            }
            
            logger.info(f"Processed note ID {note['id']} in {result['processing_time']:.3f} seconds")
            return result
        except Exception as e:
            logger.exception(f"Error processing SOAP note: {e}")
            return {
                "error": "Processing failed",
                "details": str(e)
            }