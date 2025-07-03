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
from src.database import get_sqlite_connection, UMLSSession
from src.config import get_config
from resources.common_terms import common_terms
from resources.default_patterns import DEFAULT_PATTERNS
from resources.default_clinical_terms import DEFAULT_CLINICAL_TERMS
from resources.default_disease_keywords import DEFAULT_DISEASE_KEYWORDS
from resources.common_fallbacks import (
    fallback_disease_keywords,
    fallback_symptom_cuis,
    fallback_management_plans
)
import logging

logger = logging.getLogger("HIMS-NLP")
HIMS_CONFIG = get_config()

class UMLSMapper:
    """Maps clinical terms to UMLS CUIs with caching."""
    
    def __init__(self):
        self.term_cache = {}
        self.map_terms_to_cuis(common_terms)

    @classmethod
    def get_instance(cls) -> 'UMLSMapper':
        """Get a new instance of UMLSMapper."""
        return cls()
    
    @lru_cache(maxsize=5000)
    def map_term_to_cui(self, term: str) -> List[str]:
        """Map a single term to UMLS CUIs."""
        term = term.lower()
        if term in self.term_cache:
            return self.term_cache[term]
        
        # Mock implementation - replace with actual UMLS query
        cuis = []
        self.term_cache[term] = cuis
        return cuis
    
    def map_terms_to_cuis(self, terms: List[str]) -> Dict[str, List[str]]:
        """Batch map terms to UMLS CUIs."""
        start_time = time.time()
        terms = [t.lower() for t in terms]
        cached_results = {t: self.term_cache[t] for t in terms if t in self.term_cache}
        uncached_terms = [t for t in terms if t not in self.term_cache]
        
        if uncached_terms:
            try:
                with UMLSSession() as session:
                    query = text("""
                        SELECT cui, str FROM umls.mrconso
                        WHERE str IN :terms
                        AND lat = :language AND suppress = 'N'
                        AND sab IN :trusted_sources
                    """)
                    results = session.execute(
                        query,
                        {'terms': tuple(uncached_terms), 'language': HIMS_CONFIG["UMLS_LANGUAGE"], 'trusted_sources': tuple(HIMS_CONFIG["TRUSTED_SOURCES"])}
                    ).fetchall()
                    for term in uncached_terms:
                        cuis = [row.cui for row in results if row.str.lower() == term]
                        self.term_cache[term] = cuis
                        cached_results[term] = cuis
            except Exception as e:
                logger.error(f"Error in batch UMLS query: {e}")
        
        logger.debug(f"UMLS mapping took {time.time() - start_time:.3f} seconds")
        return cached_results

class DiseaseSymptomMapper:
    """Maps diseases to symptoms and vice versa using UMLS relationships."""
    
    def __init__(self):
        self.cache = {}
    
    @classmethod
    def get_instance(cls) -> 'DiseaseSymptomMapper':
        """Get a new instance of DiseaseSymptomMapper."""
        return cls()
    
    @lru_cache(maxsize=1000)
    def get_disease_symptoms(self, disease_cui: str) -> List[Dict]:
        """Get symptoms associated with a disease CUI from UMLS."""
        try:
            with UMLSSession() as session:
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
                
                return [{'name': row.symptom_name, 'cui': row.symptom_cui} for row in symptoms]
        except Exception as e:
            logger.error(f"Error fetching symptoms for disease CUI {disease_cui}: {e}")
            return []
    
    @lru_cache(maxsize=1000)
    def get_symptom_diseases(self, symptom_cui: str) -> List[Dict]:
        """Get diseases associated with a symptom CUI from UMLS."""
        try:
            with UMLSSession() as session:
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
                
                return [{'name': row.disease_name, 'cui': row.disease_cui} for row in diseases]
        except Exception as e:
            logger.error(f"Error fetching diseases for symptom CUI {symptom_cui}: {e}")
            return []
    
    def build_disease_signatures(self) -> Dict[str, set]:
        """Build disease signatures using UMLS relationships."""
        disease_signatures = defaultdict(set)
        umls_mapper = UMLSMapper.get_instance()
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, name FROM diseases")
                diseases = cursor.fetchall()
                
                for disease_id, disease_name in diseases:
                    disease_cui = None
                    cursor.execute("SELECT cui FROM disease_keywords WHERE disease_id = ? LIMIT 1", (disease_id,))
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

class ClinicalNER:
    """Named Entity Recognition for clinical text."""
    
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
    
    def _load_patterns(self) -> List[Tuple[str, str]]:
        """Load regex patterns from the database."""
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT label, pattern FROM patterns")
                return [(row['label'], row['pattern']) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
            return DEFAULT_PATTERNS
    
    def extract_entities(self, text: str, doc=None) -> List[Tuple[str, str, dict]]:
        """Extract clinical entities from text."""
        start_time = time.time()
        if not doc:
            doc = self.nlp(text)
        
        temporal_matches = []
        for pattern, label in self.temporal_patterns:
            temporal_matches.extend(match.group() for match in pattern.finditer(text))
        
        term_matches = {match.group().lower() for match in self.terms_regex.finditer(text)} if self.terms_regex else set()
        
        entities = []
        seen_entities = set()
        
        for term in term_matches:
            if term not in seen_entities:
                context = {"severity": 1, "temporal": "UNSPECIFIED"}
                for temp_text in temporal_matches:
                    if temp_text.lower() in text.lower():
                        context["temporal"] = temp_text.lower()
                        break
                entities.append((term, "CLINICAL_TERM", context))
                seen_entities.add(term)
        
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
                            for disease in diseases:
                                disease_name = disease['name']
                                symptom_disease_map[entity_text.lower()].add(disease_name)
                                disease_symptom_count[disease_name] += 1
                                disease_symptom_map[disease_name].add(entity_text.lower())
                except Exception as e:
                    logger.error(f"Error looking up symptom: {entity_text}: {e}")
        
        for i, (entity_text, entity_label, context) in enumerate(entities):
            if entity_label in ["CLINICAL_TERM"] + [label for label, _ in self.compiled_patterns]:
                diseases = symptom_disease_map.get(entity_text.lower(), set())
                filtered_diseases = [d for d in diseases if disease_symptom_count.get(d, 0) >= 2]
                if filtered_diseases:
                    context["associated_diseases"] = filtered_diseases
                    context["disease_symptom_map"] = disease_symptom_map
                    entities[i] = (entity_text, entity_label, context)
        
        logger.debug(f"Entity extraction took {time.time() - start_time:.3f} seconds")
        return entities
    
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
            mapper = DiseaseSymptomMapper.get_instance()
            cls.disease_signatures = mapper.build_disease_signatures()
            logger.info(f"Loaded {len(cls.disease_signatures)} disease signatures from UMLS")
        
        if cls.disease_keywords is None or cls.symptom_cuis is None or cls.management_plans is None:
            cls.disease_keywords, cls.symptom_cuis, cls.management_plans = cls._load_keyword_cui_plans()
        
        cls._initialized = True

    @staticmethod
    def _load_keyword_cui_plans() -> Tuple[Dict, Dict, Dict]:
        """Load keywords, CUIs, and management plans from databases."""
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT d.name, dk.keyword, dk.cui
                    FROM diseases d
                    JOIN disease_keywords dk ON d.id = dk.disease_id
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
            logger.error(f"Failed to load data from SQLite DB: {e}")
            return fallback_disease_keywords, fallback_symptom_cuis, fallback_management_plans

    def __init__(self, ner_model=None):
        self.ner = ner_model or ClinicalNER()
        self.umls_mapper = UMLSMapper.get_instance()
        self.primary_threshold = 2.0
    
    def predict_from_text(self, text: str) -> Dict:
        """Predict diseases from clinical text."""
        start_time = time.time()
        text = bleach.clean(text)
        entities = self.ner.extract_entities(text)
        if not entities:
            logger.debug(f"Prediction took {time.time() - start_time:.3f} seconds")
            return {"primary_diagnosis": None, "differential_diagnoses": []}
        
        terms = {ent[0].lower() for ent in entities}
        disease_scores = defaultdict(float)
        
        for disease, signature in self.disease_signatures.items():
            matches = len(terms.intersection(signature))
            if matches >= 2:
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
        """Process a SOAP note for disease prediction and analysis."""
        from src.utils import prepare_note_for_nlp, generate_summary
        start_time = time.time()
        text = prepare_note_for_nlp(note)
        logger.debug(f"Text preparation took {time.time() - start_time:.3f} seconds")
        if not text:
            return {"error": "No valid text in note"}
        
        t = time.time()
        doc = self.ner.nlp(text)
        logger.debug(f"spaCy processing took {time.time() - t:.3f} seconds")
        
        t = time.time()
        # Pass the note as soap_note to generate_summary to match its required signature
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
            "processed_at": time.time(),
            "processing_time": time.time() - start_time
        }
        
        logger.info(f"Total processing time for note ID {note['id']}: {result['processing_time']:.3f} seconds")
        return result