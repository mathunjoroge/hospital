from typing import List, Dict, Set, Tuple, Optional
import torch
import re
import os
import gc
import json
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import RealDictCursor
import spacy
import medspacy
from medspacy.target_matcher import TargetRule
from scispacy.linking import EntityLinker
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from departments.models.medicine import SOAPNote
from departments.models.records import Patient
from departments.nlp.logging_setup import get_logger
from departments.nlp.nlp_pipeline import get_nlp
from departments.nlp.config import SIMILARITY_THRESHOLD, CONFIDENCE_THRESHOLD, EMBEDDING_DIM, MONGO_URI, DB_NAME, KB_PREFIX, SYMPTOMS_COLLECTION, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, FALLBACK_CFG
from departments.nlp.nlp_utils import embed_text, preprocess_text, deduplicate, get_patient_info
from departments.nlp.helper_functions import extract_duration, classify_severity, extract_location, extract_aggravating_alleviating
from departments.nlp.models.transformer_model import model, tokenizer
from departments.nlp.symptom_tracker import SymptomTracker
from departments.nlp.knowledge_base_io import load_knowledge_base
from departments.nlp.kb_updater import KnowledgeBaseUpdater
from departments.nlp.nlp_pipeline import search_local_umls_cui, clean_term
FALLBACK_TERMS_PATH=FALLBACK_CFG
logger = get_logger()
load_dotenv()

class ClinicalAnalyzer:
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), 
           retry=retry_if_exception_type(Exception))
    def __init__(self):
        self.model = model
        self.tokenizer = tokenizer
        # Initialize PostgreSQL connection pool
        try:
            self.pool = SimpleConnectionPool(
                minconn=1, maxconn=10,
                host=POSTGRES_HOST, port=POSTGRES_PORT,
                dbname=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASSWORD
            )
            logger.info("Initialized PostgreSQL connection pool for hospital_umls")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL connection pool: {e}")
            self.pool = None
        mongo_uri = MONGO_URI
        db_name = DB_NAME
        kb_prefix = KB_PREFIX
        self.medical_stop_words: Set[str] = set()
        self.medical_terms: List[Dict] = []
        self.synonyms: Dict[str, List[str]] = {}
        self.clinical_pathways: Dict[str, Dict[str, Dict]] = {}
        self.history_diagnoses: Dict[str, List[str]] = {}
        self.diagnosis_relevance: Dict[str, List[str]] = {}
        self.management_config: Dict[str, str] = {}
        self.diagnosis_treatments: Dict[str, Dict] = {}

        # Load MongoDB data with improved error handling
        try:
            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            db = client[db_name]
            collections = {
                'medical_stop_words': f'{kb_prefix}medical_stop_words',
                'medical_terms': f'{kb_prefix}medical_terms',
                'synonyms': f'{kb_prefix}synonyms',
                'clinical_pathways': f'{kb_prefix}clinical_pathways',
                'history_diagnoses': f'{kb_prefix}history_diagnoses',
                'diagnosis_relevance': f'{kb_prefix}diagnosis_relevance',
                'management_config': f'{kb_prefix}management_config',
                'diagnosis_treatments': f'{kb_prefix}diagnosis_treatments'
            }
            
            # Efficient data loading with projection
            self.medical_stop_words = {doc['word'] for doc in db[collections['medical_stop_words']].find({}, {'word': 1})}
            
            self.medical_terms = list(db[collections['medical_terms']].find(
                {}, 
                {'term': 1, 'category': 1, 'umls_cui': 1, 'semantic_type': 1}
            ))
            
            self.synonyms = {
                doc['term']: doc['aliases'] 
                for doc in db[collections['synonyms']].find({}, {'term': 1, 'aliases': 1})
                if 'term' in doc and 'aliases' in doc
            }
            
            self.clinical_pathways = {
                doc['category']: doc['paths']
                for doc in db[collections['clinical_pathways']].find(
                    {}, 
                    {'category': 1, 'paths': 1}
                )
                if 'category' in doc and 'paths' in doc
            }
            
            self.history_diagnoses = {
                doc['key']: doc['value'] 
                for doc in db[collections['history_diagnoses']].find({}, {'key': 1, 'value': 1})
                if 'key' in doc and 'value' in doc
            }
            
            client.close()
            logger.info("Loaded MongoDB data successfully")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            kb = load_knowledge_base()
            self.medical_stop_words = set(kb.get('medical_stop_words', []))
            self.medical_terms = kb.get('medical_terms', [])
            self.synonyms = kb.get('synonyms', {})
            self.clinical_pathways = kb.get('clinical_pathways', {})
            self.history_diagnoses = kb.get('history_diagnoses', {})
        except Exception as e:
            logger.error(f"Error loading from MongoDB: {e}")
            raise RuntimeError("Critical knowledge base failure") from e

        self.kb_updater = KnowledgeBaseUpdater(mongo_uri=mongo_uri, db_name=db_name, kb_prefix=kb_prefix)

        # Initialize NLP pipeline
        try:
            self.nlp = get_nlp()
            logger.info("NLP pipeline initialized successfully")
        except Exception as e:
            logger.critical(f"Failed to initialize NLP pipeline: {e}")
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("sentencizer")
            logger.warning("Using fallback NLP pipeline (English sentencizer only)")

        # Build diagnoses list
        self.diagnoses_list: Set[str] = set()
        for category, pathways in self.clinical_pathways.items():
            for key, path in pathways.items():
                differentials = path.get('differentials', [])
                if isinstance(differentials, list):
                    self.diagnoses_list.update(d.lower() for d in differentials if isinstance(d, str))

        # Initialize symptom tracker
        try:
            self.common_symptoms = SymptomTracker(
                mongo_uri=mongo_uri,
                db_name=db_name,
                symptom_collection=SYMPTOMS_COLLECTION
            )
            if not self.common_symptoms.get_all_symptoms():
                logger.warning("No symptoms loaded into SymptomTracker")
        except Exception as e:
            logger.error(f"Failed to initialize SymptomTracker: {e}")
            self.common_symptoms = None

        # Load fallback dictionary
        self.fallback_dict = {}
        if os.path.exists(FALLBACK_TERMS_PATH):
            try:
                with open(FALLBACK_TERMS_PATH, 'r') as f:
                    self.fallback_dict = json.load(f)
                logger.info(f"Loaded fallback dictionary from {FALLBACK_TERMS_PATH}")
            except Exception as e:
                logger.warning(f"Failed to load fallback dictionary: {e}")

        logger.info("ClinicalAnalyzer initialized successfully")

    def _get_postgres_connection(self):
        """Get a PostgreSQL connection from the pool."""
        if not self.pool:
            logger.error("No PostgreSQL connection pool available")
            return None
        try:
            return self.pool.getconn()
        except Exception as e:
            logger.error(f"Failed to get PostgreSQL connection: {e}")
            return None

    def _put_postgres_connection(self, conn):
        """Return a connection to the pool."""
        if conn and self.pool:
            self.pool.putconn(conn)

    def _get_umls_cui(self, symptom: str) -> Tuple[Optional[str], Optional[str]]:
        """Map a term to a UMLS CUI using the local hospital_umls database or fallback dictionary."""
        symptom_lower = preprocess_text(symptom).lower()
        if not symptom_lower:
            return None, 'Unknown'

        # Check fallback dictionary
        if symptom_lower in self.fallback_dict:
            cui = self.fallback_dict[symptom_lower].get('cui')
            semantic_type = self.fallback_dict[symptom_lower].get('semantic_type', 'Unknown')
            logger.debug(f"Found fallback CUI for '{symptom_lower}': {cui}")
            return cui, semantic_type

        # Query hospital_umls
        try:
            result = search_local_umls_cui([symptom_lower])
            cui = result.get(clean_term(symptom_lower))
            if cui:
                # Get semantic type
                conn = self._get_postgres_connection()
                if conn:
                    try:
                        cursor = conn.cursor(cursor_factory=RealDictCursor)
                        cursor.execute("""
                            SELECT sty AS semantic_type
                            FROM umls.MRSTY
                            WHERE cui = %s
                            LIMIT 1
                        """, (cui,))
                        row = cursor.fetchone()
                        semantic_type = row['semantic_type'] if row else 'Unknown'
                        cursor.close()
                        logger.debug(f"Found UMLS CUI for '{symptom_lower}': {cui}, Semantic Type: {semantic_type}")
                        return cui, semantic_type
                    except Exception as e:
                        logger.error(f"Semantic type query failed: {e}")
                    finally:
                        self._put_postgres_connection(conn)
                return cui, 'Unknown'
            logger.warning(f"No UMLS match for '{symptom_lower}'")
            return None, 'Unknown'
        except Exception as e:
            logger.error(f"UMLS lookup failed for '{symptom_lower}': {e}")
            return None, 'Unknown'

    def extract_clinical_features(self, note: SOAPNote, expected_symptoms: Optional[List[str]] = None) -> Dict:
        base_features = {
            'chief_complaint': '',
            'symptoms': [],
            'assessment': '',
            'context': {},
            'hpi': '',
            'history': '',
            'medications': '',
            'recommendation': '',
            'additional_notes': '',
            'aggravating_factors': '',
            'alleviating_factors': ''
        }
        
        if not isinstance(note, SOAPNote):
            logger.error(f"Invalid note type: {type(note)}")
            return base_features

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            features = {
                'chief_complaint': (getattr(note, 'situation', '') or '').lower().strip() or 'Unknown',
                'hpi': (getattr(note, 'hpi', '') or '').lower().strip(),
                'history': (getattr(note, 'medical_history', '') or '').lower().strip(),
                'medications': (getattr(note, 'medication_history', '') or '').lower().strip(),
                'assessment': (getattr(note, 'assessment', '') or '').lower().strip(),
                'recommendation': (getattr(note, 'recommendation', '') or '').lower().strip(),
                'additional_notes': (getattr(note, 'additional_notes', '') or '').lower().strip(),
                'aggravating_factors': (getattr(note, 'aggravating_factors', '') or '').lower().strip(),
                'alleviating_factors': (getattr(note, 'alleviating_factors', '') or '').lower().strip(),
                'symptoms': []
            }

            text_components = [
                features['chief_complaint'],
                features['hpi'],
                features['assessment'],
                features['aggravating_factors'],
                features['alleviating_factors']
            ]
            text = " ".join([c for c in text_components if c]).strip()
            
            if not text:
                logger.warning(f"Empty text for note ID {getattr(note, 'id', 'unknown')}")
                text = features['chief_complaint'] or 'Unknown'

            max_chars = 100000
            if len(text) > max_chars:
                logger.warning(f"Large text ({len(text)} chars), processing in chunks")
                chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
                docs = list(self.nlp.pipe(chunks))
                doc = spacy.tokens.Doc.from_docs(docs)
            else:
                doc = self.nlp(text)

            spacy_symptoms = []
            try:
                for ent in doc.ents:
                    kb_ents = getattr(ent._, 'kb_ents', [])
                    for umls_ent in kb_ents:
                        if len(umls_ent) < 2:
                            continue
                        cui, score = umls_ent[0], umls_ent[1]
                        if score < 0.7:
                            continue
                            
                        concept = None
                        try:
                            if "scispacy_linker" in self.nlp.pipe_names:
                                linker = self.nlp.get_pipe("scispacy_linker")
                                concept = linker.kb.cui_to_entity.get(cui)
                        except Exception:
                            pass
                            
                        spacy_symptoms.append({
                            'description': ent.text.lower(),
                            'category': self.common_symptoms._infer_category(ent.text.lower(), features['chief_complaint']) if self.common_symptoms else 'Unknown',
                            'definition': getattr(concept, 'canonical_name', 'Unknown') if concept else 'Unknown',
                            'duration': extract_duration(text) or 'Unknown',
                            'severity': classify_severity(text) or 'Unknown',
                            'location': extract_location(text, ent.text.lower()) or 'Unknown',
                            'aggravating': features['aggravating_factors'],
                            'alleviating': features['alleviating_factors'],
                            'umls_cui': cui,
                            'semantic_type': concept.types[0] if concept and concept.types else 'Unknown'
                        })
            except Exception as e:
                logger.error(f"NLP symptom extraction error: {e}")

            tracker_symptoms = []
            if self.common_symptoms:
                tracker_symptoms = self.common_symptoms.process_note(note, features['chief_complaint'], expected_symptoms)
            
            tracker_descriptions = {s['description'] for s in tracker_symptoms}
            features['symptoms'] = tracker_symptoms + [
                s for s in spacy_symptoms 
                if s['description'] not in tracker_descriptions
            ]

            potential_symptoms = getattr(note, 'Symptoms', getattr(note, 'symptoms', [])) or []
            if not potential_symptoms:
                potential_symptoms = [s.strip() for s in text.split(',') if s.strip() and len(s.strip()) > 2]
                
            for symptom in potential_symptoms:
                symptom_lower = symptom.lower()
                if self.kb_updater.is_new_symptom(symptom_lower):
                    category = self.kb_updater.infer_category(symptom_lower, text)
                    synonyms = self.kb_updater.generate_synonyms(symptom_lower)
                    cui, semantic_type = self._get_umls_cui(symptom_lower)
                    self.kb_updater.add_symptom(symptom_lower, category, synonyms, text)
                    features['symptoms'].append({
                        'description': symptom_lower,
                        'category': category,
                        'definition': f"New symptom: {symptom_lower}",
                        'duration': extract_duration(text) or 'Unknown',
                        'severity': classify_severity(text) or 'Unknown',
                        'location': extract_location(text, symptom_lower) or 'Unknown',
                        'aggravating': features['aggravating_factors'],
                        'alleviating': features['alleviating_factors'],
                        'umls_cui': cui,
                        'semantic_type': semantic_type
                    })
                    logger.info(f"Added new symptom '{symptom_lower}'")

            if not features['symptoms'] and expected_symptoms:
                for symptom in expected_symptoms:
                    symptom_lower = symptom.lower()
                    if any(pattern in text for pattern in [symptom_lower] + self.synonyms.get(symptom_lower, [])):
                        cui, semantic_type = self._get_umls_cui(symptom_lower)
                        features['symptoms'].append({
                            'description': symptom_lower,
                            'category': self.common_symptoms._infer_category(symptom_lower, features['chief_complaint']) if self.common_symptoms else 'Unknown',
                            'definition': f"Expected symptom: {symptom_lower}",
                            'duration': extract_duration(text) or 'Unknown',
                            'severity': classify_severity(text) or 'Unknown',
                            'location': extract_location(text, symptom_lower) or 'Unknown',
                            'aggravating': features['aggravating_factors'],
                            'alleviating': features['alleviating_factors'],
                            'umls_cui': cui,
                            'semantic_type': semantic_type
                        })

            context = {}
            for factor in ['aggravating', 'alleviating']:
                field = f"{factor}_factors"
                value = getattr(note, field, '')
                if value:
                    try:
                        result = extract_aggravating_alleviating(value, factor)
                        if isinstance(result, dict):
                            context.update(result)
                    except Exception as e:
                        logger.error(f"Context extraction error: {e}")
                        context[factor] = value.lower().strip()
                        
            features['context'] = {
                'aggravating': context.get('aggravating', ''),
                'alleviating': context.get('alleviating', ''),
                'sedentary': 'sedentary' in features['hpi'] or 'sitting' in features['aggravating_factors'],
                'medication': features['medications']
            }

            seen = set()
            unique_symptoms = []
            for s in features['symptoms']:
                norm_desc = preprocess_text(s['description']).lower()
                if norm_desc and norm_desc not in seen:
                    seen.add(norm_desc)
                    unique_symptoms.append(s)
            features['symptoms'] = unique_symptoms

            logger.debug(f"Extracted {len(features['symptoms'])} symptoms")
            return features
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return base_features
        finally:
            if 'doc' in locals():
                del doc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def is_relevant_dx(self, dx: str, age: Optional[int], sex: Optional[str], symptom_type: str, symptom_category: str, features: Dict) -> bool:
        if not dx or not isinstance(dx, str):
            return False
            
        dx_lower = dx.lower()
        symptom_words = {s.get('description', '').lower() for s in features.get('symptoms', [])}
        symptom_cuis = {s.get('umls_cui') for s in features.get('symptoms', []) if s.get('umls_cui')}
        semantic_types = {s.get('semantic_type') for s in features.get('symptoms', []) if s.get('semantic_type')}
        history = features.get('history', '').lower()
        chief_complaint = features.get('chief_complaint', '').lower()
        assessment = features.get('assessment', '').lower()

        if dx_lower in assessment:
            return True
        if age is not None and 'pediatric' in dx_lower and age > 18:
            return False
        if sex:
            if 'prostate' in dx_lower and sex.lower() == 'female':
                return False
            if 'ovarian' in dx_lower and sex.lower() == 'male':
                return False

        required_symptoms = self.diagnosis_relevance.get(dx_lower, [])
        matches = sum(1 for req in required_symptoms 
                      if req in symptom_words or req in history or req in chief_complaint)
        if 'Sign or Symptom' in semantic_types:
            matches += 0.5
            
        critical_conditions = {'myocardial infarction', 'pulmonary embolism', 'aortic dissection'}
        min_matches = 2 if dx_lower in critical_conditions else 1
        relevance = matches >= min_matches or any(req in chief_complaint for req in required_symptoms)
        
        if not relevance:
            logger.debug(f"Excluded dx {dx_lower}: insufficient matches")
        return relevance

    def generate_differential_dx(self, features: Dict, patient: Optional[Patient] = None) -> List[Tuple[str, float, str]]:
        try:
            dx_scores = {}
            tiered_dx = {'tier1': [], 'tier2': [], 'tier3': []}
            symptoms = features.get('symptoms', [])
            symptom_descriptions = {s.get('description', '').lower() for s in symptoms}
            symptom_cuis = {s.get('umls_cui') for s in symptoms if s.get('umls_cui')}
            history = features.get('history', '').lower()
            additional_notes = features.get('additional_notes', '').lower()
            chief_complaint = features.get('chief_complaint', '').lower()
            assessment = features.get('assessment', '').lower()
            text = f"{chief_complaint} {features.get('hpi', '')} {additional_notes}".strip()

            text_embedding = None
            if text:
                try:
                    text_embedding = embed_text(text)
                except Exception as e:
                    logger.warning(f"Embedding failed: {e}")

            patient_info = {}
            if patient:
                try:
                    patient_info = get_patient_info(patient.patient_id)
                except Exception as e:
                    logger.error(f"Patient info error: {e}")
            sex = patient_info.get('sex')
            age = patient_info.get('age')

            if assessment:
                dx_match = re.search(r"primary assessment:\s*([^\.\n]+)", assessment, re.IGNORECASE)
                if dx_match:
                    dx_name = dx_match.group(1).strip().lower()
                    if self.is_relevant_dx(dx_name, age, sex, '', '', features):
                        matches = sum(1 for req in self.diagnosis_relevance.get(dx_name, []) 
                                  if req in symptom_descriptions)
                        if matches >= min(2, len(self.diagnosis_relevance.get(dx_name, []))):
                            dx_scores[dx_name] = (0.95, "Primary diagnosis")
                            tiered_dx['tier1'].append((dx_name, 0.95, "Primary diagnosis"))

            high_risk_conditions = {'pulmonary embolism', 'myocardial infarction', 'meningitis', 'aortic dissection'}
            rare_conditions = {'malaria', 'leptospirosis', 'dengue'}
            
            for symptom in symptoms:
                symptom_type = symptom.get('description', '').lower()
                symptom_category = symptom.get('category', '').lower()
                symptom_cui = symptom.get('umls_cui')
                location = symptom.get('location', '').lower()
                aggravating = symptom.get('aggravating', '').lower()
                alleviating = symptom.get('alleviating', '').lower()
                
                for category, pathways in self.clinical_pathways.items():
                    for key, path in pathways.items():
                        key_lower = key.lower()
                        synonyms = self.synonyms.get(symptom_type, [])
                        path_cui = path.get('metadata', {}).get('umls_cui')
                        
                        if not (any(k in {symptom_type, location, chief_complaint} for k in key_lower.split('|'))) and \
                           not (symptom_type in synonyms) and \
                           not (symptom_category == category.lower()) and \
                           not (symptom_cui and path_cui and symptom_cui == path_cui):
                            continue
                            
                        differentials = path.get('differentials', [])
                        contextual_triggers = path.get('contextual_triggers', [])
                        
                        for diff in differentials:
                            if not isinstance(diff, str):
                                continue
                            diff_lower = diff.lower()
                            if diff_lower == assessment:
                                continue
                            if not self.is_relevant_dx(diff, age, sex, symptom_type, symptom_category, features):
                                continue
                                
                            required_symptoms = self.diagnosis_relevance.get(diff_lower, [])
                            matches = sum(1 for req in required_symptoms 
                                          if req in symptom_descriptions or req in text.lower())
                            score = 0.5
                            if symptom_type in chief_complaint:
                                score += 0.2
                            if symptom_category == category.lower():
                                score += 0.15
                            if symptom_cui and path_cui and symptom_cui == path_cui:
                                score += 0.1
                            score += matches / max(len(required_symptoms), 1) * 0.25
                            reasoning = f"Matched symptom: {symptom_type} in {location}"
                            
                            if diff_lower in high_risk_conditions or diff_lower in rare_conditions:
                                if matches >= 3 and (not contextual_triggers or any(t.lower() in text for t in contextual_triggers)):
                                    tiered_dx['tier3'].append((diff, min(score, 0.7), reasoning + "; high-risk"))
                            elif matches >= 2:
                                tiered_dx['tier1'].append((diff, min(score, 0.85), reasoning))
                            else:
                                tiered_dx['tier2'].append((diff, min(score, 0.65), reasoning))
                                
                            dx_scores[diff] = (min(score, 0.95), reasoning)

            for condition, aliases in self.history_diagnoses.items():
                if any(alias.lower() in history for alias in aliases):
                    if condition.lower() != assessment and self.is_relevant_dx(condition, age, sex, '', '', features):
                        matches = sum(1 for req in self.diagnosis_relevance.get(condition.lower(), []) 
                                      if req in symptom_descriptions)
                        if matches >= 1:
                            tiered_dx['tier2'].append((condition, 0.7, "Historical diagnosis"))
                            dx_scores[condition] = (0.7, "Historical diagnosis")

            if text_embedding is not None:
                for dx in list(dx_scores.keys()):
                    try:
                        dx_embedding = embed_text(dx)
                        similarity = torch.cosine_similarity(
                            text_embedding.unsqueeze(0), 
                            dx_embedding.unsqueeze(0)
                        ).item()
                        if similarity < SIMILARITY_THRESHOLD:
                            continue
                        old_score, reasoning = dx_scores[dx]
                        adjusted_score = min(old_score + similarity * 0.05, 0.95)
                        dx_scores[dx] = (adjusted_score, reasoning)
                        
                        for tier in tiered_dx:
                            for i, (t_dx, t_score, t_reason) in enumerate(tiered_dx[tier]):
                                if t_dx == dx:
                                    tiered_dx[tier][i] = (dx, adjusted_score, t_reason)
                                    break
                    except Exception as e:
                        logger.warning(f"Similarity failed for {dx}: {e}")

            ranked_dx = tiered_dx['tier1'] + tiered_dx['tier2'] + tiered_dx['tier3']
            if not ranked_dx:
                return [("Undetermined", 0.1, "Insufficient data")]
                
            total_score = sum(score for _, score, _ in ranked_dx)
            if total_score > 0:
                ranked_dx = [(dx, score / total_score * 0.9, reason) for dx, score, reason in ranked_dx]
                
            ranked_dx = sorted(ranked_dx, key=lambda x: x[1], reverse=True)[:5]
            return ranked_dx
        except Exception as e:
            logger.error(f"Differential diagnosis failed: {e}")
            return [("Undetermined", 0.1, "Diagnosis error")]

    def generate_management_plan(self, features: Dict, differentials: List[Tuple[str, float, str]]) -> Dict:
        try:
            plan = {
                'workup': {'urgent': [], 'routine': []},
                'treatment': {'symptomatic': [], 'definitive': [], 'lifestyle': []},
                'follow_up': [],
                'references': []
            }
            
            symptoms = features.get('symptoms', [])
            symptom_descriptions = {s.get('description', '').lower() for s in symptoms}
            symptom_cuis = {s.get('umls_cui') for s in symptoms}
            primary_dx = features.get('assessment', '').lower()
            high_risk_conditions = {'temporal arteritis', 'atrial fibrillation', 'subarachnoid hemorrhage', 
                                   'myocardial infarction', 'pulmonary embolism', 'aortic dissection'}
            
            for category, pathways in self.clinical_pathways.items():
                for key, path in pathways.items():
                    key_parts = key.lower().split('|')
                    path_cui = path.get('metadata', {}).get('umls_cui')
                    differentials_list = path.get('differentials', [])
                    
                    if not any(d.lower() in primary_dx for d in differentials_list) and \
                       not any(k in symptom_descriptions or k in primary_dx for k in key_parts) and \
                       not any(symptom_cuis and path_cui and cui == path_cui for cui in symptom_cuis):
                        continue
                        
                    workup = path.get('workup', {})
                    for w in workup.get('urgent', []):
                        parsed = parse_conditional_workup(w, symptoms)
                        if parsed:
                            plan['workup']['urgent'].append(parsed)
                    for w in workup.get('routine', []):
                        parsed = parse_conditional_workup(w, symptoms)
                        if parsed:
                            plan['workup']['routine'].append(parsed)
                            
                    management = path.get('management', {})
                    plan['treatment']['symptomatic'].extend(management.get('symptomatic', []))
                    plan['treatment']['definitive'].extend(management.get('definitive', []))
                    plan['treatment']['lifestyle'].extend(management.get('lifestyle', []))
                    
                    plan['follow_up'].extend(path.get('follow_up', []))
                    plan['references'].extend(path.get('references', []))

            follow_up_match = re.search(r'Follow-Up:\s*([^\.]+)', features.get('recommendation', ''), re.IGNORECASE)
            if follow_up_match:
                plan['follow_up'] = [follow_up_match.group(1).strip()]
            else:
                high_risk = any(dx.lower() in high_risk_conditions for dx, _, _ in differentials)
                plan['follow_up'] = ['Follow-up in 3-5 days'] if high_risk else ['Follow-up in 2 weeks']

            for section in ['workup', 'treatment']:
                for sub in plan[section]:
                    plan[section][sub] = list(set(plan[section][sub]))
            plan['follow_up'] = list(set(plan['follow_up']))
            plan['references'] = list(set(plan['references']))

            return plan
        except Exception as e:
            logger.error(f"Management plan failed: {e}")
            return {
                'workup': {'urgent': [], 'routine': []},
                'treatment': {'symptomatic': [], 'definitive': ['Pending diagnosis'], 'lifestyle': []},
                'follow_up': ['Follow-up in 2 weeks'],
                'references': []
            }

    def __del__(self):
        if hasattr(self, 'pool') and self.pool:
            self.pool.closeall()
            logger.debug("Closed PostgreSQL connection pool")

def parse_conditional_workup(workup: str, symptoms: List[Dict]) -> str:
    if not isinstance(workup, str):
        return ''
    if 'if' not in workup.lower():
        return workup
        
    try:
        condition = workup.lower().split('if')[1].strip()
        for symptom in symptoms:
            desc = symptom.get('description', '').lower()
            category = symptom.get('category', '').lower()
            cui = symptom.get('umls_cui')
            if condition in desc or condition in category or (cui and cui in condition):
                return workup.split('if')[0].strip()
    except Exception:
        pass
    return ''