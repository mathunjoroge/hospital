import os
import logging
from typing import List, Dict, Optional, Tuple
import re
import json
import time
from collections import defaultdict
from functools import lru_cache
import bleach
from concurrent.futures import ThreadPoolExecutor
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Local project imports
from src.database import get_sqlite_connection
from src.config import get_config
from src.utils import prepare_note_for_nlp, generate_summary
from resources.default_patterns import DEFAULT_PATTERNS
from resources.default_clinical_terms import DEFAULT_CLINICAL_TERMS
from resources.default_disease_keywords import DEFAULT_DISEASE_KEYWORDS
from resources.common_fallbacks import (
    fallback_disease_keywords,
    fallback_symptom_cuis,
    fallback_management_plans,
    COMMON_SYMPTOM_DISEASE_MAP,
)
from departments.nlp.resources.cancer_diseases import (
    cancer_symptoms,
    CANCER_TERMS,
    CANCER_PLANS,
    CANCER_PATTERNS,
    BREAST_CANCER_KEYWORD_CUIS,
    BREAST_CANCER_SYMPTOMS,
    CANCER_KEYWORDS_FILE,
)
from resources.clinical_markers import LAB_THRESHOLDS, CANCER_DISEASES
from .umls_mapper import UMLSMapper
from .disease_symptom_mapper import DiseaseSymptomMapper

# Set environment variables to avoid TensorFlow usage
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["USE_TF"] = "0"

# Ensure reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42) if torch.cuda.is_available() else None

# Download NLTK data
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/mathu/projects/hospital/logs/hims_nlp.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HIMS-NLP")
HIMS_CONFIG = get_config()

# Initialize spaCy with fallback
try:
    nlp = spacy.load("en_core_sci_sm", disable=["ner"])
    nlp.add_pipe("sentencizer")
except Exception as e:
    logger.warning(f"Failed to load en_core_sci_sm: {e}. Falling back to en_core_web_sm.")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    nlp.add_pipe("sentencizer")
lemmatizer = WordNetLemmatizer()

# Load cancer classifier model and tokenizer
model_name = "emilyalsentzer/Bio_ClinicalBERT"
trained_model_path = "/home/mathu/projects/hospital/cancer_classifier"
try:
    tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(trained_model_path)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
        logger.info("Cancer model moved to GPU")
except Exception as e:
    logger.error(f"Failed to load cancer classifier from {trained_model_path}: {e}")
    raise

# Label mapping for cancer types
cancer_types = list(CANCER_DISEASES)
label_map = {name: idx for idx, name in enumerate(cancer_types)}
id2label = {idx: name for idx, name in enumerate(cancer_types)}

# Load AMR/IPC classifier
amr_ipc_model_path = "/home/mathu/projects/hospital/amr_ipc_classifier"
try:
    amr_ipc_tokenizer = AutoTokenizer.from_pretrained(amr_ipc_model_path)
    amr_ipc_model = AutoModelForSequenceClassification.from_pretrained(amr_ipc_model_path)
    amr_ipc_categories = list(amr_ipc_model.config.id2label.values()) if hasattr(amr_ipc_model.config, 'id2label') else ["amr_high", "amr_low", "amr_none", "ipc_adequate", "ipc_inadequate", "ipc_none"]
    amr_ipc_id2label = {i: c for i, c in enumerate(amr_ipc_categories)}
    if amr_ipc_model.config.num_labels != len(amr_ipc_categories):
        logger.error(f"AMR/IPC model expects {amr_ipc_model.config.num_labels} labels, but {len(amr_ipc_categories)} categories defined")
        raise ValueError("AMR/IPC model label mismatch")
    amr_ipc_model.eval()
    if torch.cuda.is_available():
        amr_ipc_model.to("cuda")
        logger.info("AMR/IPC model moved to GPU")
except Exception as e:
    logger.error(f"Failed to load AMR/IPC classifier from {amr_ipc_model_path}: {e}")
    raise

class ClinicalNER:
    """Named Entity Recognition for clinical text with enhanced symptom and risk factor handling."""
    
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
                    UNION
                    SELECT risk_factor FROM disease_risk_factors
                """)
                terms = {row['name'].lower() for row in cursor.fetchall()}
                terms.update(CANCER_TERMS)
                logger.info(f"Loaded {len(terms)} clinical terms from database")
                return terms
        except Exception as e:
            logger.error(f"Error loading clinical terms: {e}")
            terms = DEFAULT_CLINICAL_TERMS
            terms.update(CANCER_TERMS)
            return terms

    def __init__(self, umls_mapper: UMLSMapper = None):
        self.nlp = DiseasePredictor.nlp
        self.negation_terms = {"no", "not", "denies", "without", "absent", "negative"}
        self.lemmatizer = lemmatizer
        self.patterns = self._load_patterns()
        self.compiled_patterns = []
        for label, pattern in self.patterns:
            try:
                self.compiled_patterns.append((label, re.compile(pattern, re.IGNORECASE)))
            except re.error as e:
                logger.error(f"Failed to compile pattern for label {label}: {pattern} ({e})")
        self.temporal_patterns = [
            (re.compile(r"\b(\d+\s*(day|days|week|weeks|month|months|year|years)\s*(ago)?)\b", re.IGNORECASE), "DURATION"),
            (re.compile(r"\b(since|for)\s*(\d+\s*(day|days|week|weeks|month|months|year|years))\b", re.IGNORECASE), "DURATION"),
            (re.compile(r"\b\d+\s*days?\b", re.IGNORECASE), "DURATION"),
            (re.compile(r"\bfor\s*(three|four|five|six|seven|eight|nine|ten|[1-9]\d*)\s*days?\b", re.IGNORECASE), "DURATION"),
        ]
        self.risk_factor_patterns = [
            (label, re.compile(pattern, re.IGNORECASE)) for label, pattern in [
                ("RISK_FACTOR", r"\b(smoking|tobacco use|cigarette|smoker)\b"),
                ("RISK_FACTOR", r"\b(obesity|overweight|bmi\s*>\s*30)\b"),
                ("RISK_FACTOR", r"\b(family history|genetic predisposition|hereditary)\b"),
                ("RISK_FACTOR", r"\b(alcohol consumption|heavy drinking|alcoholism)\b"),
                ("RISK_FACTOR", r"\b(hypertension|high blood pressure)\b"),
                ("RISK_FACTOR", r"\b(diabetes|type 2 diabetes)\b"),
            ]
        ]
        self.terms_regex = re.compile(
            r'\b(' + '|'.join(map(re.escape, sorted(DiseasePredictor.clinical_terms, key=len, reverse=True))) + r')\b',
            re.IGNORECASE
        ) if DiseasePredictor.clinical_terms else None
        self.symptom_mapper = DiseaseSymptomMapper.get_instance()
        self.umls_mapper = umls_mapper or UMLSMapper.get_instance()
        self.invalid_terms = {'mg', 'ms', 'g', 'ml', 'mm', 'ng', 'dl', 'hr'}
    
    def _load_patterns(self) -> List[Tuple[str, str]]:
        def _flatten_cancer_patterns() -> List[Tuple[str, str]]:
            flat = []
            for cancer_type, pattern_list in CANCER_PATTERNS.items():
                for label, pattern in pattern_list:
                    try:
                        re.compile(pattern, re.IGNORECASE)
                        flat.append((label, pattern))
                    except re.error as e:
                        logger.error(f"Invalid regex pattern for {cancer_type}, label {label}: {pattern} ({e})")
            return flat

        patterns = []
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT label, pattern FROM patterns")
                for row in cursor.fetchall():
                    try:
                        re.compile(row['pattern'], re.IGNORECASE)
                        patterns.append((row['label'], row['pattern']))
                    except re.error as e:
                        logger.error(f"Invalid regex pattern in database for label {row['label']}: {row['pattern']} ({e})")
                patterns.extend(_flatten_cancer_patterns())
                logger.info(f"Loaded {len(patterns)} valid patterns from database and cancer patterns")
                return patterns
        except Exception as e:
            logger.error(f"Error loading patterns from database: {e}")
            patterns = DEFAULT_PATTERNS
            patterns.extend(_flatten_cancer_patterns())
            logger.info(f"Loaded {len(patterns)} valid fallback and cancer patterns")
            return patterns
    
    def extract_entities(self, text: str, doc=None) -> List[Tuple[str, str, dict]]:
        start_time = time.time()
        if not text or not text.strip():
            logger.warning("Empty or invalid input text for entity extraction")
            return []
        if not doc:
            doc = self.nlp(text)
        
        temporal_matches = []
        for pattern, label in self.temporal_patterns:
            temporal_matches.extend(match.group() for match in pattern.finditer(text))
        
        term_matches = {match.group().lower() for match in self.terms_regex.finditer(text)
                        if match.group().lower() not in self.invalid_terms} if self.terms_regex else set()

        symptom_groups = defaultdict(set)
        risk_factor_groups = defaultdict(set)
        for term in term_matches:
            base_term = self.umls_mapper.normalize_symptom(term)
            if base_term not in self.invalid_terms:
                if base_term in DiseasePredictor.symptom_cuis:
                    symptom_groups[base_term].add(term)
                else:
                    with get_sqlite_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT risk_factor FROM disease_risk_factors WHERE risk_factor = ?", (term,))
                        if cursor.fetchone():
                            risk_factor_groups[base_term].add(term)
        
        entities = []
        seen_entities = set()
        
        for base_term, variants in symptom_groups.items():
            if base_term not in seen_entities and base_term not in self.invalid_terms:
                representative = max(variants, key=len)
                context = {
                    "severity": 1.0,
                    "temporal": "UNSPECIFIED",
                    "variants": list(variants),
                    "cancer_relevance": 0.9 if base_term in cancer_symptoms else 0.5,
                    "type": "SYMPTOM",
                }
                
                for temp_text in temporal_matches:
                    if temp_text.lower() in text.lower():
                        context["temporal"] = temp_text.lower()
                        break
                
                entities.append((representative, "CLINICAL_TERM", context))
                seen_entities.add(base_term)
        
        for base_term, variants in risk_factor_groups.items():
            if base_term not in seen_entities and base_term not in self.invalid_terms:
                representative = max(variants, key=len)
                context = {
                    "severity": 1.0,
                    "temporal": "UNSPECIFIED",
                    "variants": list(variants),
                    "type": "RISK_FACTOR",
                }
                
                for temp_text in temporal_matches:
                    if temp_text.lower() in text.lower():
                        context["temporal"] = temp_text.lower()
                        break
                
                entities.append((representative, "RISK_FACTOR", context))
                seen_entities.add(base_term)
        
        for label, pattern in self.compiled_patterns + self.risk_factor_patterns:
            for match in pattern.finditer(text):
                match_text = match.group().lower()
                normalized = self.umls_mapper.normalize_symptom(match_text)
                
                if normalized not in seen_entities and normalized not in self.invalid_terms:
                    context = {"severity": 1.0, "temporal": "UNSPECIFIED", "type": label}
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
        risk_factor_disease_map = defaultdict(set)
        disease_symptom_count = defaultdict(int)
        disease_symptom_map = defaultdict(set)
        
        symptom_texts = [self.umls_mapper.normalize_symptom(ent[0]) for ent in entities
                        if ent[0].lower() not in self.invalid_terms and ent[2].get('type') == 'SYMPTOM']
        risk_factor_texts = [self.umls_mapper.normalize_symptom(ent[0]) for ent in entities
                             if ent[0].lower() not in self.invalid_terms and ent[2].get('type') == 'RISK_FACTOR']
        
        with ThreadPoolExecutor(max_workers=HIMS_CONFIG.get("MAX_WORKERS", 4)) as executor:
            symptom_futures = {executor.submit(self._get_diseases_for_symptom, symptom_text): symptom_text
                               for symptom_text in set(symptom_texts)}
            risk_factor_futures = {executor.submit(self._get_diseases_for_risk_factor, rf_text): rf_text
                                   for rf_text in set(risk_factor_texts)}
            
            for future in symptom_futures:
                symptom_text = symptom_futures[future]
                try:
                    diseases = future.result()
                    symptom_disease_map[symptom_text] = diseases
                    for disease in diseases:
                        disease_symptom_count[disease] += 1
                        disease_symptom_map[disease].add(symptom_text)
                except Exception as e:
                    logger.error(f"Error processing symptom {symptom_text}: {e}")
            
            for future in risk_factor_futures:
                rf_text = risk_factor_futures[future]
                try:
                    diseases = future.result()
                    risk_factor_disease_map[rf_text] = diseases
                    for disease in diseases:
                        disease_symptom_count[disease] += 0.5
                        disease_symptom_map[disease].add(rf_text)
                except Exception as e:
                    logger.error(f"Error processing risk factor {rf_text}: {e}")
        
        for i, (entity_text, entity_label, context) in enumerate(entities):
            normalized_text = self.umls_mapper.normalize_symptom(entity_text)
            diseases = symptom_disease_map.get(normalized_text, set()) if context.get('type') == 'SYMPTOM' else risk_factor_disease_map.get(normalized_text, set())
            
            if not diseases and normalized_text in COMMON_SYMPTOM_DISEASE_MAP and context.get('type') == 'SYMPTOM':
                diseases = set(COMMON_SYMPTOM_DISEASE_MAP[normalized_text])
            
            filtered_diseases = [d for d in diseases if disease_symptom_count.get(d, 0) >= 2 or d.lower() in CANCER_DISEASES]
            if filtered_diseases:
                context["associated_diseases"] = filtered_diseases
                context["disease_symptom_map"] = {d: list(disease_symptom_map[d]) for d in filtered_diseases}
                entities[i] = (entity_text, entity_label, context)
        
        logger.debug(f"Entity extraction found {len(entities)} entities in {time.time() - start_time:.3f} seconds")
        return entities
    
    def _get_diseases_for_symptom(self, symptom_text: str) -> set:
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
    
    def _get_diseases_for_risk_factor(self, risk_factor_text: str) -> set:
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT d.name
                    FROM disease_risk_factors drf
                    JOIN diseases d ON drf.disease_id = d.id
                    WHERE drf.risk_factor = ?
                """, (risk_factor_text,))
                diseases = {row['name'].lower() for row in cursor.fetchall()}
                return diseases
        except Exception as e:
            logger.error(f"Error looking up diseases for risk factor '{risk_factor_text}': {e}")
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
            logger.warning("Empty note text for keyword/CUI extraction")
            return [], []
        
        entities = self.extract_entities(text)
        terms = {ent[0].lower() for ent in entities if ent and ent[0].lower() not in self.invalid_terms}
        
        expected_keywords = []
        reference_cuis = []
        
        try:
            disease_keywords = DiseasePredictor.disease_keywords
            symptom_cuis = DiseasePredictor.symptom_cuis
            risk_factors = DiseasePredictor.risk_factors
            
            for term in terms:
                for keyword, cui in disease_keywords.items():
                    if keyword in term or term in keyword:
                        if keyword not in expected_keywords:
                            expected_keywords.append(keyword)
                            reference_cuis.append(cui)
                        break
                if term in CANCER_TERMS and term not in expected_keywords:
                    expected_keywords.append(term)
                    cui = self.umls_mapper.map_term_to_cui(term)
                    if cui:
                        reference_cuis.append(cui[0])
                
                if term in symptom_cuis and term not in expected_keywords:
                    expected_keywords.append(term)
                    if symptom_cuis[term]:
                        reference_cuis.append(symptom_cuis[term])
                
                if term in risk_factors and term not in expected_keywords:
                    expected_keywords.append(term)
                    cui = self.umls_mapper.map_term_to_cui(term)
                    if cui:
                        reference_cuis.append(cui[0])
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
    risk_factors = None
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
        
        if cls.risk_factors is None:
            cls.risk_factors = cls._load_risk_factors()
            logger.info(f"Loaded {len(cls.risk_factors)} risk factors")
        
        cls._initialized = True
        logger.info("DiseasePredictor initialization complete")
    
    @staticmethod
    def _load_disease_keywords() -> Dict[str, str]:
        def load_fallback_keywords() -> Dict[str, str]:
            try:
                if os.path.exists(CANCER_KEYWORDS_FILE):
                    with open(CANCER_KEYWORDS_FILE, 'r') as f:
                        keywords = json.load(f)
                        logger.info(f"Loaded {len(keywords)} cancer keywords from {CANCER_KEYWORDS_FILE}")
                        return {k.lower(): v for k, v in keywords.items()}
                else:
                    logger.warning(f"External file {CANCER_KEYWORDS_FILE} not found. Using default keywords.")
                    return fallback_disease_keywords
            except Exception as e:
                logger.error(f"Failed to load keywords from {CANCER_KEYWORDS_FILE}: {e}")
                return fallback_disease_keywords

        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT d.name, dk.keyword, dk.cui
                    FROM diseases d
                    JOIN disease_keywords dk ON d.id = dk.disease_id
                """)
                keywords = {row['keyword'].lower(): row['cui'] for row in cursor.fetchall()}
                
                try:
                    if os.path.exists(CANCER_KEYWORDS_FILE):
                        with open(CANCER_KEYWORDS_FILE, 'r') as f:
                            external_keywords = json.load(f)
                            keywords.update({k.lower(): v for k, v in external_keywords.items()})
                            logger.info(f"Loaded {len(external_keywords)} additional cancer keywords from {CANCER_KEYWORDS_FILE}")
                    else:
                        logger.warning(f"External file {CANCER_KEYWORDS_FILE} not found. Skipping external keywords.")
                except Exception as e:
                    logger.error(f"Failed to load external keywords from {CANCER_KEYWORDS_FILE}: {e}")
                
                logger.info(f"Loaded {len(keywords)} total disease keywords from database and external file")
                return keywords
        except Exception as e:
            logger.error(f"Failed to load disease keywords from database: {e}")
            keywords = load_fallback_keywords()
            return keywords
    
    @staticmethod
    def _load_symptom_cuis() -> Dict:
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name, cui FROM symptoms")
                cuis = {row['name'].lower(): row['cui'] for row in cursor.fetchall()}
                cuis.update(BREAST_CANCER_SYMPTOMS)
                logger.info(f"Loaded {len(cuis)} symptom CUIs from database")
                return cuis
        except Exception as e:
            logger.error(f"Failed to load symptom CUIs: {e}")
            cuis = fallback_symptom_cuis
            cuis.update(BREAST_CANCER_SYMPTOMS)
            logger.info(f"Using {len(cuis)} fallback symptom CUIs")
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
                
                management_plans.update(CANCER_PLANS)
                logger.info(f"Loaded {len(management_plans)} management plans from database")
                return management_plans
        except Exception as e:
            logger.error(f"Failed to load management plans and lab tests: {e}")
            logger.info(f"Using {len(fallback_management_plans)} fallback management plans")
            return fallback_management_plans
    
    @staticmethod
    def _load_risk_factors() -> Dict:
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT d.name, drf.risk_factor, drf.description
                    FROM disease_risk_factors drf
                    JOIN diseases d ON drf.disease_id = d.id
                """)
                risk_factors = {}
                for row in cursor.fetchall():
                    disease_name = row['name'].lower()
                    if disease_name not in risk_factors:
                        risk_factors[disease_name] = []
                    risk_factors[disease_name].append({
                        'risk_factor': row['risk_factor'],
                        'description': row['description'] or ''
                    })
                logger.info(f"Loaded {sum(len(rf) for rf in risk_factors.values())} risk factors for {len(risk_factors)} diseases")
                return risk_factors
        except Exception as e:
            logger.error(f"Failed to load risk factors: {e}")
            return {}

    def __init__(self, ner_model=None):
        self.umls_mapper = UMLSMapper.get_instance()
        self.ner = ner_model or ClinicalNER(umls_mapper=self.umls_mapper)
        self.primary_threshold = HIMS_CONFIG.get("SIMILARITY_THRESHOLD", 1.0)
        self.min_symptom_count = 2
    
    def predict_cancer_risk(self, text: str) -> Dict[str, float]:
        try:
            text = text.strip()
            if len(text.split()) < 3:
                logger.debug("Input text too short for cancer prediction, padding")
                text = text + " " + text
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}
            logger.debug(f"Cancer model input tokens: {inputs}")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                logger.debug(f"Cancer model logits: {logits}")
                probs = torch.softmax(logits, dim=1)[0]
                logger.debug(f"Cancer model probabilities: {probs}")
            result = {id2label[i]: prob.item() for i, prob in enumerate(probs)}
            total = sum(result.values())
            if total == 0 or not all(0 <= v <= 1 for v in result.values()):
                logger.warning("Invalid cancer probabilities, returning uniform distribution")
                result = {cancer: 1.0 / len(cancer_types) for cancer in cancer_types}
            logger.debug(f"Final cancer prediction: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in cancer prediction: {e}")
            return {cancer: 1.0 / len(cancer_types) for cancer in cancer_types}
    
    def predict_amr_ipc(self, text: str) -> Dict[str, float]:
        """Predict AMR/IPC categories for the given clinical text.

        Args:
            text (str): Clinical text to analyze.

        Returns:
            Dict[str, float]: Probabilities for each AMR/IPC category.
        """
        if not text or not text.strip():
            logger.warning("Empty or invalid input text for AMR/IPC prediction")
            return {label: 0.0 for label in amr_ipc_categories}
        try:
            text = text.strip()
            if len(text.split()) < 3:
                logger.debug("Input text too short for AMR/IPC prediction, padding")
                text = text + " " + text
            inputs = amr_ipc_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}
            logger.debug(f"AMR/IPC input tokens: {inputs}")
            with torch.no_grad():
                outputs = amr_ipc_model(**inputs)
                logits = outputs.logits
                logger.debug(f"AMR/IPC model logits: {logits}")
                probs = torch.softmax(logits, dim=1)[0]
                logger.debug(f"AMR/IPC model probabilities: {probs}")
            result = {amr_ipc_id2label[i]: prob.item() for i, prob in enumerate(probs)}
            total = sum(result.values())
            if total == 0 or not all(0 <= v <= 1 for v in result.values()):
                logger.warning("Invalid AMR/IPC probabilities, returning uniform distribution")
                result = {label: 1.0 / len(amr_ipc_categories) for label in amr_ipc_categories}
            logger.debug(f"Final AMR/IPC prediction: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in AMR/IPC prediction: {e}")
            return {label: 1.0 / len(amr_ipc_categories) for label in amr_ipc_categories}
    
    def predict_from_text(self, text: str, amr_ipc_text: str = None) -> Dict:
        start_time = time.time()
        cleaned_text = bleach.clean(text)  # For entity extraction
        if not cleaned_text or not cleaned_text.strip():
            logger.warning("No entities extracted from text")
            return {
                "primary_diagnosis": None,
                "differential_diagnoses": [],
                "lab_abnormalities": [],
                "risk_factors": [],
                "cancer_probabilities": {cancer: 0.0 for cancer in cancer_types},
                "amr_ipc_probabilities": {label: 0.0 for label in amr_ipc_categories}
            }
        
        entities = self.ner.extract_entities(cleaned_text)
        
        symptom_terms = set()
        risk_factor_terms = set()
        lab_abnormalities = set()
        cancer_relevance_scores = {}
        
        for entity in entities:
            normalized = self.umls_mapper.normalize_symptom(entity[0])
            if normalized not in self.ner.invalid_terms:
                if entity[1] == 'CLINICAL_TERM' and entity[2].get('type') == 'SYMPTOM':
                    symptom_terms.add(normalized)
                    cancer_relevance_scores[normalized] = entity[2].get('cancer_relevance', 0.5)
                elif entity[1] == 'RISK_FACTOR':
                    risk_factor_terms.add(normalized)
                elif entity[1] in ['TUMOR_MARKER', 'BLOOD_COUNT', 'INFLAMMATORY_MARKER'] and entity[2].get('abnormal'):
                    lab_abnormalities.add(entity[2]['potential_cancer'])
        
        logger.debug(f"Extracted symptoms: {symptom_terms}, Risk factors: {risk_factor_terms}, Lab abnormalities: {lab_abnormalities}")
        
        cancer_diseases = {'prostate cancer', 'colorectal cancer', 'ovarian cancer', 'pancreatic cancer',
                           'liver cancer', 'leukemia', 'lung cancer', 'breast cancer', 'lymphoma'}
        
        disease_scores = defaultdict(float)
        
        for disease, signature in self.disease_signatures.items():
            matches = len(symptom_terms.intersection(signature))
            for term in symptom_terms.intersection(signature):
                matches += cancer_relevance_scores.get(term, 0.5)
            if disease.lower() in lab_abnormalities:
                matches += 2.0
            if disease.lower() in self.risk_factors:
                matches += len([rf for rf in risk_factor_terms if any(rf in r['risk_factor'].lower() for r in self.risk_factors[disease.lower()])]) * 0.5
            if matches >= (self.min_symptom_count - 1 if disease.lower() in cancer_diseases else self.min_symptom_count):
                disease_scores[disease] = matches
        
        if not disease_scores and len(symptom_terms) >= self.min_symptom_count - 1:
            for disease, signature in self.disease_signatures.items():
                matches = len(symptom_terms.intersection(signature))
                for term in symptom_terms.intersection(signature):
                    matches += cancer_relevance_scores.get(term, 0.5)
                if disease.lower() in lab_abnormalities:
                    matches += 2.0
                if disease.lower() in self.risk_factors:
                    matches += len([rf for rf in risk_factor_terms if any(rf in r['risk_factor'].lower() for r in self.risk_factors[disease.lower()])]) * 0.5
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
        
        cancer_probabilities = self.predict_cancer_risk(cleaned_text)  # Use cleaned text for cancer model
        max_cancer = max(cancer_probabilities, key=cancer_probabilities.get)
        max_prob = cancer_probabilities[max_cancer]
        if max_prob > HIMS_CONFIG.get("CANCER_CONFIDENCE_THRESHOLD", 0.3) and not primary_diagnosis:
            primary_diagnosis = {"disease": max_cancer, "score": max_prob}
        elif max_prob > HIMS_CONFIG.get("CANCER_CONFIDENCE_THRESHOLD", 0.3):
            differential_diagnoses.append({"disease": max_cancer, "score": max_prob})
        
        amr_ipc_probabilities = self.predict_amr_ipc(amr_ipc_text if amr_ipc_text else text)  # Use AMR/IPC-specific text if provided
        
        result = {
            "primary_diagnosis": primary_diagnosis,
            "differential_diagnoses": differential_diagnoses,
            "lab_abnormalities": list(lab_abnormalities),
            "risk_factors": list(risk_factor_terms),
            "cancer_probabilities": cancer_probabilities,
            "amr_ipc_probabilities": amr_ipc_probabilities
        }
        
        logger.info(f"Predicted {len(sorted_diseases)} diseases from {len(symptom_terms)} symptoms, {len(risk_factor_terms)} risk factors, and {len(lab_abnormalities)} lab abnormalities in {time.time() - start_time:.3f} seconds")
        return result
    
    def process_soap_note(self, note: Dict) -> Dict:
        start_time = time.time()
        
        try:
            if not isinstance(note, dict) or not note.get("id") or not note.get("patient_id"):
                logger.error("Invalid SOAP note: missing id or patient_id")
                return {
                    "error": "Invalid SOAP note",
                    "note_id": note.get("id"),
                    "details": "Note must be a dictionary with 'id' and 'patient_id' fields"
                }
            
            text = prepare_note_for_nlp(note)
            logger.debug(f"Text preparation took {time.time() - start_time:.3f} seconds")
            if not text:
                return {
                    "error": "No valid text in note",
                    "note_id": note.get("id"),
                    "details": "Note data is empty or missing required fields (situation, hpi, symptoms, assessment)"
                }
            
            # Prepare AMR/IPC-specific text
            amr_ipc_text = ' '.join(filter(None, [
                note.get('situation', ''),
                note.get('hpi', ''),
                note.get('medical_history', ''),
                note.get('medication_history', '')
            ])).strip()
            if not amr_ipc_text:
                logger.warning("No valid text for AMR/IPC prediction from specified fields")
                amr_ipc_text = text  # Fallback to full text
            
            t = time.time()
            doc = self.nlp(text)
            logger.debug(f"spaCy processing took {time.time() - t:.3f} seconds")
            
            t = time.time()
            summary = generate_summary(text, soap_note=note, doc=doc, nlp=self.nlp, clinical_terms=self.clinical_terms)
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
            predictions = self.predict_from_text(text, amr_ipc_text=amr_ipc_text)  # Pass AMR/IPC-specific text
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
                    if disease in cancer_diseases or max(predictions["cancer_probabilities"].values()) > HIMS_CONFIG.get("CANCER_CONFIDENCE_THRESHOLD", 0.3):
                        management_plans[disease] = management_plans.get(disease, {})
                        management_plans[disease]["cancer_follow_up"] = (
                            "Refer to oncologist; consider imaging (e.g., CT/MRI) and biopsy if indicated."
                        )
                    if disease in self.risk_factors:
                        management_plans[disease] = management_plans.get(disease, {})
                        management_plans[disease]["risk_factors"] = self.risk_factors[disease]
                
                for disease in predictions["differential_diagnoses"]:
                    disease_name = disease["disease"].lower()
                    if disease_name in self.management_plans:
                        management_plans[disease_name] = self.management_plans[disease_name]
                    if disease_name in cancer_diseases:
                        management_plans[disease_name] = management_plans.get(disease_name, {})
                        management_plans[disease_name]["cancer_follow_up"] = (
                            "Monitor symptoms; consider tumor marker tests and imaging."
                        )
                    if disease_name in self.risk_factors:
                        management_plans[disease_name] = management_plans.get(disease_name, {})
                        management_plans[disease_name]["risk_factors"] = self.risk_factors[disease_name]
                
                for lab_abnormality in predictions["lab_abnormalities"]:
                    if lab_abnormality in cancer_diseases:
                        management_plans[lab_abnormality] = management_plans.get(lab_abnormality, {})
                        management_plans[lab_abnormality]["lab_follow_up"] = (
                            f"Repeat {lab_abnormality} tumor marker tests in 4-6 weeks; refer to specialist."
                        )
                    if lab_abnormality in self.risk_factors:
                        management_plans[lab_abnormality] = management_plans.get(lab_abnormality, {})
                        management_plans[lab_abnormality]["risk_factors"] = self.risk_factors[lab_abnormality]
                
                amr_ipc_probabilities = predictions["amr_ipc_probabilities"]
                max_amr_ipc_category = max(amr_ipc_probabilities, key=amr_ipc_probabilities.get)
                max_amr_ipc_prob = amr_ipc_probabilities[max_amr_ipc_category]
                amr_ipc_threshold = HIMS_CONFIG.get("AMR_IPC_CONFIDENCE_THRESHOLD", 0.3)
                
                if max_amr_ipc_prob > amr_ipc_threshold:
                    amr_ipc_recommendations = {}
                    if "amr_high" in max_amr_ipc_category:
                        amr_ipc_recommendations["amr"] = {
                            "status": "High AMR Risk",
                            "recommendation": "Initiate culture-guided antibiotic therapy; consider multidrug-resistant organism protocols."
                        }
                    elif "amr_low" in max_amr_ipc_category:
                        amr_ipc_recommendations["amr"] = {
                            "status": "Low AMR Risk",
                            "recommendation": "Continue standard antibiotic therapy; monitor for resistance development."
                        }
                    elif "amr_none" in max_amr_ipc_category:
                        amr_ipc_recommendations["amr"] = {
                            "status": "No AMR Risk",
                            "recommendation": "No specific AMR interventions required."
                        }
                    
                    if "ipc_inadequate" in max_amr_ipc_category:
                        amr_ipc_recommendations["ipc"] = {
                            "status": "Inadequate IPC",
                            "recommendation": "Implement strict infection control measures; review hand hygiene and isolation protocols."
                        }
                    elif "ipc_adequate" in max_amr_ipc_category:
                        amr_ipc_recommendations["ipc"] = {
                            "status": "Adequate IPC",
                            "recommendation": "Maintain current infection prevention protocols."
                        }
                    elif "ipc_none" in max_amr_ipc_category:
                        amr_ipc_recommendations["ipc"] = {
                            "status": "No IPC Concerns",
                            "recommendation": "No additional IPC measures required."
                        }
                    
                    predictions["amr_ipc_recommendations"] = amr_ipc_recommendations
                
            except Exception as e:
                logger.error(f"Error fetching management plans or AMR/IPC recommendations: {e}")
            
            logger.debug(f"Management plans and AMR/IPC recommendations retrieval took {time.time() - t:.3f} seconds")
            
            result = {
                "note_id": note["id"],
                "patient_id": note["patient_id"],
                "primary_diagnosis": predictions["primary_diagnosis"],
                "differential_diagnoses": predictions["differential_diagnoses"],
                "lab_abnormalities": predictions["lab_abnormalities"],
                "risk_factors": predictions["risk_factors"],
                "cancer_probabilities": predictions["cancer_probabilities"],
                "amr_ipc_probabilities": predictions["amr_ipc_probabilities"],
                "amr_ipc_recommendations": predictions.get("amr_ipc_recommendations", {}),
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