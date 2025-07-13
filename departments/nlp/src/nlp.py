import os
import logging
from typing import List, Dict, Optional, Tuple
from sqlalchemy.sql import text
import re
import time
from collections import defaultdict
from functools import lru_cache
import bleach
from cachetools import LRUCache
from concurrent.futures import ThreadPoolExecutor
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from threading import Lock
import numpy as np  # Added for probability handling

# Set environment variables to avoid TensorFlow usage
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["USE_TF"] = "0"

# Download NLTK data
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Local project imports
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
from resources.cancer_diseases import cancer_symptoms,BREAST_CANCER_TERMS, CANCER_PLANS,  BREAST_CANCER_PATTERNS, BREAST_CANCER_KEYWORDS, BREAST_CANCER_KEYWORD_CUIS,BREAST_CANCER_SYMPTOMS
from resources.clinical_markers import LAB_THRESHOLDS, CANCER_DISEASES
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HIMS-NLP")
HIMS_CONFIG = get_config()

# Initialize spaCy and NLTK
nlp = spacy.load("en_core_sci_sm", disable=["ner"])
nlp.add_pipe("sentencizer")
lemmatizer = WordNetLemmatizer()

# Load trained model and tokenizer
model_name = "emilyalsentzer/Bio_ClinicalBERT"
trained_model_path = "/home/mathu/projects/hospital/cancer_classifier"
tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
model = AutoModelForSequenceClassification.from_pretrained(trained_model_path)
model.eval()
if torch.cuda.is_available():  # Ensure model uses GPU if available
    model.to("cuda")
    logger.info("Model moved to GPU")

# Label mapping for cancer types
cancer_types = list(CANCER_DISEASES) 
label_map = {name: idx for idx, name in enumerate(cancer_types)}
id2label = {idx: name for idx, name in enumerate(cancer_types)}

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
    
    _instance = None
    _lock = Lock()

    @classmethod
    def get_instance(cls) -> 'UMLSMapper':
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self.term_cache = LRUCache(maxsize=10000)
        self.symptom_normalizations = self._load_symptom_normalizations()
        self.map_terms_to_cuis_batch(common_terms)
        self._initialized = True
        logger.info("UMLSMapper initialized successfully")
    
    def _load_symptom_normalizations(self) -> Dict[str, str]:
        try:
            return SYMPTOM_NORMALIZATIONS
        except Exception as e:
            logger.error(f"Failed to load symptom normalizations: {e}")
            return {}
    
    def normalize_symptom(self, symptom: str) -> str:
        symptom_lower = symptom.lower()
        return self.symptom_normalizations.get(symptom_lower, symptom_lower)
    
    @lru_cache(maxsize=10000)
    def map_term_to_cui(self, term: str) -> List[str]:
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
        start_time = time.time()
        if not terms:
            return {}
        normalized_terms = [self.normalize_symptom(t) for t in terms]
        
        results = {}
        uncached_terms = []
        
        for term in normalized_terms:
            if term in self.term_cache:
                results[term] = self.term_cache[term]
            else:
                uncached_terms.append(term)
        
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
                    
                    term_map = defaultdict(list)
                    for row in db_results:
                        term_map[row.term_str].append(row.cui)
                    
                    for term in uncached_terms:
                        cuis = term_map.get(term, [])
                        self.term_cache[term] = cuis
                        results[term] = cuis
            except Exception as e:
                logger.error(f"Error in batch UMLS query: {e}")
                for term in uncached_terms:
                    results[term] = self.map_term_to_cui(term)
        
        original_results = {orig_term: results.get(self.normalize_symptom(orig_term), []) 
                           for orig_term in terms}
        
        logger.debug(f"Batch UMLS mapping took {time.time() - start_time:.3f} seconds")
        return original_results

class DiseaseSymptomMapper:
    """Maps diseases to symptoms and vice versa using UMLS relationships."""
    
    _instance = None
    _lock = Lock()

    @classmethod
    def get_instance(cls) -> 'DiseaseSymptomMapper':
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self.cache = LRUCache(maxsize=1000)
        self.umls_mapper = UMLSMapper.get_instance()
        self._initialized = True
    
    @lru_cache(maxsize=1000)
    def get_disease_symptoms(self, disease_cui: str) -> List[Dict]:
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
        normalized = self.umls_mapper.normalize_symptom(symptom_text)
        return COMMON_SYMPTOM_DISEASE_MAP.get(normalized, [])
    
    def build_disease_signatures(self) -> Dict[str, set]:
        disease_signatures = defaultdict(set)
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, name FROM diseases")
                diseases = cursor.fetchall()
                from resources.cancer_diseases import cancer_diseases
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(self._process_disease_signature, disease_id, disease_name, self.umls_mapper)
                    for disease_id, disease_name in diseases
                ]
                
                for future in futures:
                    disease_name, symptoms = future.result()
                    if symptoms:
                        disease_signatures[disease_name] = symptoms
            
            for disease, data in cancer_diseases.items():
                disease_signatures[disease].update(data['symptoms'])
            
            logger.info(f"Built disease signatures for {len(disease_signatures)} diseases")
            return disease_signatures
        except Exception as e:
            logger.error(f"Error building disease signatures: {e}")
            return {}
    
    def _process_disease_signature(self, disease_id, disease_name, umls_mapper):
        symptoms = set()
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT cui FROM disease_keywords WHERE disease_id = ? LIMIT 1", (disease_id,))
                row = cursor.fetchone()
                disease_cui = row['cui'] if row else None
                
                if not disease_cui:
                    disease_cuis = umls_mapper.map_term_to_cui(disease_name)
                    disease_cui = disease_cuis[0] if disease_cuis else None
                
                if disease_cui:
                    symptom_data = self.get_disease_symptoms(disease_cui)
                    for symptom in symptom_data:
                        symptoms.add(symptom['name'].lower())
                
                if not symptoms:
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
        if DiseasePredictor.clinical_terms is None:
            DiseasePredictor.clinical_terms = cls._load_clinical_terms()
    
    @classmethod
    def initialize(cls):
        if DiseasePredictor.clinical_terms is None:
            DiseasePredictor.clinical_terms = cls._load_clinical_terms()
    
    @staticmethod
    def _load_clinical_terms() -> set:
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT name FROM symptoms
                    UNION
                    SELECT keyword FROM disease_keywords
                """)
                terms = {row['name'].lower() for row in cursor.fetchall()}
                # Add breast cancer-specific terms from the new module
                terms.update(BREAST_CANCER_TERMS)
                logger.info(f"Loaded {len(terms)} clinical terms from database")
                return terms
        except Exception as e:
            logger.error(f"Error loading clinical terms: {e}")
            terms = DEFAULT_CLINICAL_TERMS
            terms.update(BREAST_CANCER_TERMS)
            return terms

    def __init__(self, umls_mapper: UMLSMapper = None):
        self.nlp = DiseasePredictor.nlp
        self.negation_terms = {"no", "not", "denies", "without", "absent", "negative"}
        self.lemmatizer = lemmatizer
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
        self.invalid_terms = {'mg', 'ms', 'g', 'ml', 'mm', 'ng', 'dl', 'hr'}
    

    def _load_patterns(self) -> List[Tuple[str, str]]:
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT label, pattern FROM patterns")
                patterns = [(row['label'], row['pattern']) for row in cursor.fetchall()]
                # Add breast cancer-specific patterns from the new module
                patterns.extend(BREAST_CANCER_PATTERNS)
                logger.info(f"Loaded {len(patterns)} patterns from database")
                return patterns
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
            patterns = DEFAULT_PATTERNS
            patterns.extend(BREAST_CANCER_PATTERNS)
            return patterns
    
    def extract_entities(self, text: str, doc=None) -> List[Tuple[str, str, dict]]:
        start_time = time.time()
        if not doc:
            doc = self.nlp(text)
        
        temporal_matches = []
        for pattern, label in self.temporal_patterns:
            temporal_matches.extend(match.group() for match in pattern.finditer(text))
        
        term_matches = {match.group().lower() for match in self.terms_regex.finditer(text)
                        if match.group().lower() not in self.invalid_terms} if self.terms_regex else set()

        
        symptom_groups = defaultdict(set)
        for term in term_matches:
            base_term = self.umls_mapper.normalize_symptom(term)
            if base_term not in self.invalid_terms:
                symptom_groups[base_term].add(term)
        
        entities = []
        seen_entities = set()
        
        for base_term, variants in symptom_groups.items():
            if base_term not in seen_entities and base_term not in self.invalid_terms:
                representative = max(variants, key=len)
                context = {
                    "severity": 1.0,
                    "temporal": "UNSPECIFIED",
                    "variants": list(variants),
                    "cancer_relevance": 0.9 if base_term in cancer_symptoms else 0.5  # Ensure cancer_symptoms is defined
                }
                
                for temp_text in temporal_matches:
                    if temp_text.lower() in text.lower():
                        context["temporal"] = temp_text.lower()
                        break
                
                entities.append((representative, "CLINICAL_TERM", context))
                seen_entities.add(base_term)
        
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
                    
                    if label in ['TUMOR_MARKER', 'BLOOD_COUNT', 'INFLAMMATORY_MARKER']:
                        try:
                            marker = match.group(1).lower()
                            value = float(match.group(2))
                            unit = match.group(3)
                            if marker not in self.invalid_terms:
                                context.update({"value": value, "unit": unit, "abnormal": False})
                                if marker in LAB_THRESHOLDS:
                                    threshold_info = LAB_THRESHOLDS[marker]
                                    is_abnormal = (value > threshold_info['threshold'] if threshold_info.get('condition') != 'low'
                                                else value < threshold_info['threshold'])
                                    if is_abnormal:
                                        context['abnormal'] = True
                                        context['potential_cancer'] = threshold_info['cancer']
                                        context['cancer_relevance'] = 0.95
                                entities.append((marker, label, context))
                                seen_entities.add(normalized)
                        except (IndexError, ValueError) as e:
                            logger.warning(f"Failed to parse lab result for {match_text}: {e}")
                    else:
                        entities.append((match.group(), label, context))
                        seen_entities.add(normalized)
        
        symptom_disease_map = defaultdict(set)
        disease_symptom_count = defaultdict(int)
        disease_symptom_map = defaultdict(set)
        
        symptom_texts = [self.umls_mapper.normalize_symptom(ent[0]) for ent in entities
                        if ent[0].lower() not in self.invalid_terms]
        
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._get_diseases_for_symptom, symptom_text): symptom_text
                       for symptom_text in set(symptom_texts)}
            
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
        
        for i, (entity_text, entity_label, context) in enumerate(entities):
            normalized_text = self.umls_mapper.normalize_symptom(entity_text)
            diseases = symptom_disease_map.get(normalized_text, set())
            
            if not diseases and normalized_text in COMMON_SYMPTOM_DISEASE_MAP:
                diseases = set(COMMON_SYMPTOM_DISEASE_MAP[normalized_text])
            
            filtered_diseases = [d for d in diseases if disease_symptom_count.get(d, 0) >= 2 or d.lower() in CANCER_DISEASES]
            if filtered_diseases:
                context["associated_diseases"] = filtered_diseases
                context["disease_symptom_map"] = {d: list(disease_symptom_map[d]) for d in filtered_diseases}
                entities[i] = (entity_text, entity_label, context)
        
        logger.debug(f"Entity extraction found {len(entities)} entities in {time.time() - start_time:.3f} seconds")
        return entities
    
    def _get_diseases_for_symptom(self, symptom_text):
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT cui FROM symptoms WHERE name = ?", (symptom_text,))
                row = cursor.fetchone()
                symptom_cui = row['cui'] if row else None
            
            if not symptom_cui:
                cuis = self.umls_mapper.map_term_to_cui(symptom_text)
                symptom_cui = cuis[0] if cuis else None
            
            if symptom_cui:
                diseases = self.symptom_mapper.get_symptom_diseases(symptom_cui)
                return {disease['name'].lower() for disease in diseases}
            
            return set(self.symptom_mapper.get_symptom_diseases_fallback(symptom_text))
        except Exception as e:
            logger.error(f"Error looking up diseases for symptom '{symptom_text}': {e}")
            return set()
    
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
            terms = {ent[0].lower() for ent in entities if ent and ent[0].lower() not in self.invalid_terms}
            
            expected_keywords = []
            reference_cuis = []
            
            try:
                disease_keywords = DiseasePredictor.disease_keywords
                symptom_cuis = DiseasePredictor.symptom_cuis
                
                # Ensure breast cancer terms are included
                for term in terms:
                    for keyword, cui in disease_keywords.items():
                        if keyword in term or term in keyword:
                            if keyword not in expected_keywords:
                                expected_keywords.append(keyword)
                                reference_cuis.append(cui)
                            break
                    if term in BREAST_CANCER_KEYWORDS and term not in expected_keywords:
                        expected_keywords.append(term)
                        cui = self.umls_mapper.map_term_to_cui(term)
                        if cui:
                            reference_cuis.append(cui[0])
                    
                    if term in symptom_cuis and term not in expected_keywords:
                        expected_keywords.append(term)
                        if symptom_cuis[term]:
                            reference_cuis.append(symptom_cuis[term])
            except Exception as e:
                logger.error(f"Error fetching keywords from database: {e}")
                disease_keywords = DEFAULT_DISEASE_KEYWORDS
                disease_keywords.update(BREAST_CANCER_KEYWORD_CUIS)
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
    nlp = nlp
    clinical_terms = None
    disease_signatures = None
    disease_keywords = None
    symptom_cuis = None
    management_plans = None
    _initialized = False

    @classmethod
    def initialize(cls, force: bool = False):
        if cls._initialized and not force:
            return
        
        logger.info("Initializing DiseasePredictor resources...")
        
        ClinicalNER.initialize()
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
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT d.name, dk.keyword, dk.cui
                    FROM diseases d
                    JOIN disease_keywords dk ON d.id = dk.disease_id
                """)
                keywords = {row['keyword'].lower(): row['cui'] for row in cursor.fetchall()}
                # Add breast cancer-specific keywords
                keywords.update({
                    'breast lump': 'C0234450',
                    'nipple retraction': 'C0234451',
                    'nipple discharge': 'C0027408',
                    'breast mass': 'C0234450',
                    'ductal carcinoma': 'C0007124',
                    'brca': 'C0599878'
                })
                return keywords
        except Exception as e:
            logger.error(f"Failed to load disease keywords: {e}")
            keywords = fallback_disease_keywords
            keywords.update({
                'breast lump': 'C0234450',
                'nipple retraction': 'C0234451',
                'nipple discharge': 'C0027408',
                'breast mass': 'C0234450',
                'ductal carcinoma': 'C0007124',
                'brca': 'C0599878'
            })
            return keywords

    @staticmethod
    def _load_symptom_cuis() -> Dict:
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name, cui FROM symptoms")
                cuis = {row['name'].lower(): row['cui'] for row in cursor.fetchall()}
                # Add breast cancer-specific symptoms from resources.cancer_diseases
                cuis.update(BREAST_CANCER_SYMPTOMS)
                return cuis
        except Exception as e:
            logger.error(f"Failed to load symptom CUIs: {e}")
            cuis = fallback_symptom_cuis
            cuis.update(BREAST_CANCER_SYMPTOMS)
            return cuis



    @staticmethod
    def _load_management_plans() -> Dict:
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT d.name, dmp.plan
                    FROM disease_management_plans dmp
                    JOIN diseases d ON dmp.disease_id = d.id
                """)
                management_plans = {row['name'].lower(): {'plan': row['plan']} for row in cursor.fetchall()}
                
                cursor.execute("""
                    SELECT d.name, dl.lab_test, dl.description
                    FROM disease_labs dl
                    JOIN diseases d ON dl.disease_id = d.id
                """)
                lab_tests = cursor.fetchall()
                
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
                
                # Load cancer-specific plans from resources.cancer_diseases
                management_plans.update(CANCER_PLANS)
                
                return management_plans
        except Exception as e:
            logger.error(f"Failed to load management plans and lab tests: {e}")
            return fallback_management_plans

    def __init__(self, ner_model=None):
        self.umls_mapper = UMLSMapper.get_instance()
        self.ner = ner_model or ClinicalNER(umls_mapper=self.umls_mapper)
        self.primary_threshold = 1.0  # Lowered from 2.0 to ensure diagnosis assignment
        self.min_symptom_count = 2
    
    def predict_cancer_risk(self, text: str) -> Dict[str, float]:
        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0]
            return {id2label[i]: prob.item() for i, prob in enumerate(probs)}
        except Exception as e:
            logger.error(f"Error in Bio_ClinicalBERT prediction: {e}")
            return {cancer: 0.0 for cancer in cancer_types}

    def predict_from_text(self, text: str) -> Dict:
        start_time = time.time()
        text = bleach.clean(text)
        entities = self.ner.extract_entities(text)
        
        if not entities:
            logger.warning("No entities extracted from text")
            return {
                "primary_diagnosis": None,
                "differential_diagnoses": [],
                "lab_abnormalities": [],
                "cancer_probabilities": {cancer: 0.0 for cancer in cancer_types}
            }
        
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
        
        cancer_diseases = {'prostate cancer', 'colorectal cancer', 'ovarian cancer', 'pancreatic cancer',
                        'liver cancer', 'leukemia', 'lung cancer', 'breast cancer', 'lymphoma'}
        
        disease_scores = defaultdict(float)
        
        for disease, signature in self.disease_signatures.items():
            matches = len(symptom_terms.intersection(signature))
            for term in symptom_terms.intersection(signature):
                matches += cancer_relevance_scores.get(term, 0.5)
            if disease.lower() in lab_abnormalities:
                matches += 2.0
            if matches >= (self.min_symptom_count - 1 if disease.lower() in cancer_diseases else self.min_symptom_count):
                disease_scores[disease] = matches
        
        if not disease_scores and len(symptom_terms) >= self.min_symptom_count - 1:
            for disease, signature in self.disease_signatures.items():
                matches = len(symptom_terms.intersection(signature))
                for term in symptom_terms.intersection(signature):
                    matches += cancer_relevance_scores.get(term, 0.5)
                if disease.lower() in lab_abnormalities:
                    matches += 2.0
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
            primary_threshold = self.primary_threshold - 0.5 if any(d["disease"].lower() in cancer_diseases for d in sorted_diseases) else self.primary_threshold
            if sorted_diseases[0]["score"] >= primary_threshold:
                primary_diagnosis = sorted_diseases[0]
                differential_diagnoses = sorted_diseases[1:] if len(sorted_diseases) > 1 else []
            else:
                differential_diagnoses = sorted_diseases
        
        cancer_probabilities = self.predict_cancer_risk(text)
        max_cancer = max(cancer_probabilities, key=cancer_probabilities.get)
        max_prob = cancer_probabilities[max_cancer]
        if max_prob > 0.3 and not primary_diagnosis:  # Lowered from 0.7
            primary_diagnosis = {"disease": max_cancer, "score": max_prob}
        elif max_prob > 0.3:
            differential_diagnoses.append({"disease": max_cancer, "score": max_prob})
        
        result = {
            "primary_diagnosis": primary_diagnosis,
            "differential_diagnoses": differential_diagnoses,
            "lab_abnormalities": list(lab_abnormalities),
            "cancer_probabilities": cancer_probabilities
        }
        
        logger.info(f"Predicted {len(sorted_diseases)} diseases from {len(symptom_terms)} symptoms and {len(lab_abnormalities)} lab abnormalities in {time.time() - start_time:.3f} seconds")
        return result
    
    def process_soap_note(self, note: Dict) -> Dict:
        from src.utils import prepare_note_for_nlp, generate_summary
        start_time = time.time()
        
        try:
            text = prepare_note_for_nlp(note)
            logger.debug(f"Text preparation took {time.time() - start_time:.3f} seconds")
            if not text:
                return {"error": "No valid text in note", "note_id": note.get("id")}
            
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
            
            management_plans = {}
            try:
                cancer_diseases = {
                    'prostate cancer', 'colorectal cancer', 'ovarian cancer', 'pancreatic cancer',
                    'liver cancer', 'leukemia', 'lung cancer', 'breast cancer', 'lymphoma'
                }
                if predictions["primary_diagnosis"]:
                    disease = predictions["primary_diagnosis"]["disease"].lower()
                    if disease in self.management_plans:
                        management_plans[disease] = self.management_plans[disease]
                    if disease in cancer_diseases or max(predictions["cancer_probabilities"].values()) > 0.3:
                        management_plans[disease] = management_plans.get(disease, {})
                        management_plans[disease]["cancer_follow_up"] = (
                            "Refer to oncologist; consider imaging (e.g., CT/MRI) and biopsy if indicated."
                        )
                
                for disease in predictions["differential_diagnoses"]:
                    disease_name = disease["disease"].lower()
                    if disease_name in self.management_plans:
                        management_plans[disease_name] = self.management_plans[disease_name]
                    if disease_name in cancer_diseases:
                        management_plans[disease_name] = management_plans.get(disease_name, {})
                        management_plans[disease_name]["cancer_follow_up"] = (
                            "Monitor symptoms; consider tumor marker tests and imaging."
                        )
                
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
                "cancer_probabilities": predictions["cancer_probabilities"],
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
                "details": str(e),
                "note_id": note.get("id")
            }