from typing import List, Dict, Optional
import os
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from departments.nlp.logging_setup import get_logger
from departments.nlp.knowledge_base_io import load_knowledge_base, save_knowledge_base
from departments.nlp.nlp_utils import embed_text
from departments.nlp.models.transformer_model import model, tokenizer
import torch
import requests
import time

logger = get_logger()

from departments.nlp.config import UTS_API_KEY, UTS_BASE_URL, MONGO_URI, DB_NAME

class KnowledgeBaseUpdater:
    def __init__(self, mongo_uri: str = None, db_name: str = None, kb_prefix: str = 'kb_'):
        self.mongo_uri = mongo_uri or MONGO_URI or 'mongodb://localhost:27017'
        self.db_name = db_name or DB_NAME or 'clinical_db'
        self.kb_prefix = kb_prefix
        self.model = model
        self.tokenizer = tokenizer
        self.uts_api_key = UTS_API_KEY or 'mock_api_key'
        self.uts_base_url = UTS_BASE_URL or 'https://uts-ws.nlm.nih.gov/rest'
        self.knowledge_base = load_knowledge_base()
        self.medical_terms = self.knowledge_base.get('medical_terms', [])
        self.synonyms = self.knowledge_base.get('synonyms', {})
        self.symptoms = self.knowledge_base.get('symptoms', {})
        self.version = self.knowledge_base.get('version', '1.1.0')

        # Cached category embeddings
        self.category_embeddings = {}

        # Initialize MongoDB
        try:
            self.client = MongoClient(self.mongo_uri)
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.medical_terms_collection = self.db[f'{kb_prefix}medical_terms']
            self.synonyms_collection = self.db[f'{kb_prefix}synonyms']
            self.pathways_collection = self.db[f'{kb_prefix}clinical_pathways']
            self.symptoms_collection = self.db[f'{kb_prefix}symptoms']
            self.umls_cache = self.db['umls_cache']
            logger.info("Connected to MongoDB for knowledge base updates.")
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {str(e)}. Falling back to JSON.")
            self.client = None
        except Exception as e:
            logger.error(f"Error initializing MongoDB: {str(e)}")
            raise

        # Initialize HTTP session with retries
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    def is_new_symptom(self, symptom: str, context: str = "") -> bool:
        """Check if a symptom is new."""
        symptom_lower = symptom.lower().strip()
        for term in self.medical_terms:
            if term.get('term', '').lower() == symptom_lower:
                return False
        for key, aliases in self.synonyms.items():
            if symptom_lower == key or symptom_lower in [a.lower() for a in aliases]:
                return False
        for cat, sym_dict in self.symptoms.items():
            if symptom_lower in sym_dict:
                return False
        logger.debug(f"Symptom '{symptom_lower}' is new.")
        return True

    def infer_category(self, symptom: str, context: str) -> str:
        """Infer symptom category using transformer model with cached embeddings."""
        try:
            text = f"{symptom} {context}".lower()[:512]
            embedding = embed_text(text)
            categories = [
                'musculoskeletal', 'respiratory', 'gastrointestinal',
                'cardiovascular', 'neurological', 'dermatological',
                'sensory', 'hematologic', 'endocrine', 'genitourinary', 'psychiatric'
            ]
            # Cache category embeddings
            if not self.category_embeddings:
                for cat in categories:
                    self.category_embeddings[cat] = embed_text(cat)
            category_scores = {}
            for cat in categories:
                similarity = torch.cosine_similarity(
                    embedding.unsqueeze(0), self.category_embeddings[cat].unsqueeze(0)
                ).item()
                category_scores[cat] = similarity
            inferred_category = max(category_scores, key=category_scores.get)
            logger.debug(f"Inferred category for '{symptom}': {inferred_category} (Scores: {category_scores})")
            return inferred_category
        except Exception as e:
            logger.error(f"Error inferring category for '{symptom}': {str(e)}")
            return 'general'

    def generate_synonyms(self, symptom: str) -> List[str]:
        """Generate synonyms using UMLS UTS API with caching and rate limiting."""
        symptom_lower = symptom.lower().strip()
        
        # Check cache
        try:
            if self.client:
                cached = self.umls_cache.find_one({'symptom': symptom_lower})
                if cached and 'synonyms' in cached:
                    logger.debug(f"Retrieved cached synonyms for '{symptom_lower}': {cached['synonyms']}")
                    return cached['synonyms']
        except Exception as e:
            logger.error(f"Error checking synonym cache for '{symptom_lower}': {str(e)}")

        try:
            if self.uts_api_key == 'mock_api_key':
                synonyms = self._mock_uts_synonyms(symptom_lower)
            else:
                ticket = self._get_uts_ticket()
                url = f"{self.uts_base_url}/search/current"
                params = {
                    'string': symptom_lower,
                    'ticket': ticket,
                    'searchType': 'exact',
                    'sabs': 'SNOMEDCT_US'
                }
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                synonyms = set()
                if data.get('result', {}).get('results'):
                    cui = data['result']['results'][0]['ui']
                    atoms_url = f"{self.uts_base_url}/content/current/CUI/{cui}/atoms"
                    atoms_response = self.session.get(atoms_url, params={'ticket': ticket}, timeout=10)
                    atoms_response.raise_for_status()
                    atoms_data = atoms_response.json()
                    for atom in atoms_data.get('result', []):
                        synonyms.add(atom['name'].lower())
                synonyms = [s for s in synonyms if s != symptom_lower and len(s) > 2]
                if not synonyms:
                    synonyms = self._generate_synonyms_fallback(symptom_lower)
                time.sleep(0.2)  # Rate limiting
            
            # Validate synonyms
            synonyms = [s for s in synonyms if s.lower() not in self.synonyms and s.lower() != symptom_lower]
            if len(synonyms) > 10:
                synonyms = synonyms[:10]  # Limit to 10 synonyms
            
            # Cache synonyms
            if self.client and synonyms:
                try:
                    self.umls_cache.update_one(
                        {'symptom': symptom_lower},
                        {'$set': {'synonyms': synonyms}},
                        upsert=True
                    )
                    logger.debug(f"Cached synonyms for '{symptom_lower}'")
                except Exception as e:
                    logger.error(f"Failed to cache synonyms for '{symptom_lower}': {str(e)}")
            
            logger.info(f"Synonyms for '{symptom_lower}': {synonyms}")
            return synonyms
        except Exception as e:
            logger.error(f"UMLS synonym generation failed for '{symptom_lower}': {str(e)}")
            return self._generate_synonyms_fallback(symptom_lower)

    def _get_uts_ticket(self) -> str:
        """Obtain a single-use UTS ticket with retry."""
        if self.uts_api_key == 'mock_api_key':
            return 'mock_ticket'
        try:
            ticket_url = f"{self.uts_base_url}/tickets"
            response = self.session.post(ticket_url, data={'apiKey': self.uts_api_key}, timeout=10)
            response.raise_for_status()
            time.sleep(0.2)  # Rate limiting
            return response.text
        except Exception as e:
            logger.error(f"Failed to get UTS ticket: {str(e)}")
            return ''

    def _mock_uts_synonyms(self, symptom: str) -> List[str]:
        """Mock UMLS API for testing."""
        mock_data = {
            'chest tightness': ['dyspnea', 'shortness of breath', 'chest discomfort'],
            'back pain': ['lumbago', 'backache', 'lower back pain'],
            'obesity': ['overweight', 'high bmi', 'adiposity'],
            'dizziness': ['vertigo', 'lightheadedness', 'unsteadiness']
        }
        synonyms = mock_data.get(symptom.lower(), [])
        logger.debug(f"Mock UMLS synonyms for '{symptom}': {synonyms}")
        return synonyms

    def _generate_synonyms_fallback(self, symptom: str) -> List[str]:
        """Fallback synonym generation with validation."""
        try:
            symptom_lower = symptom.lower()
            synonyms = [f"{symptom_lower} syndrome", f"acute {symptom_lower}", f"chronic {symptom_lower}"]
            return [s for s in synonyms if s.lower() != symptom_lower and len(s) > 2]
        except Exception as e:
            logger.error(f"Error generating fallback synonyms for '{symptom}': {str(e)}")
            return []

    def get_umls_metadata(self, symptom: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Retrieve UMLS CUI, semantic type, and ICD-10 code with caching."""
        symptom_lower = symptom.lower().strip()
        
        # Check cache
        try:
            if self.client:
                cached = self.umls_cache.find_one({'symptom': symptom_lower})
                if cached and 'cui' in cached:
                    logger.debug(f"Retrieved cached UMLS metadata for '{symptom_lower}'")
                    return cached['cui'], cached['semantic_type'], cached.get('icd10', None)
        except Exception as e:
            logger.error(f"Error checking UMLS cache for '{symptom_lower}': {str(e)}")

        # Mock data for testing
        if self.uts_api_key == 'mock_api_key':
            mock_metadata = {
                'chest tightness': {'cui': 'C0242209', 'semantic_type': 'Sign or Symptom', 'icd10': 'R07.89'},
                'back pain': {'cui': 'C0004604', 'semantic_type': 'Sign or Symptom', 'icd10': 'M54.9'},
                'obesity': {'cui': 'C0028754', 'semantic_type': 'Disease or Syndrome', 'icd10': 'E66.9'},
                'dizziness': {'cui': 'C0012833', 'semantic_type': 'Sign or Symptom', 'icd10': 'R42'}
            }
            metadata = mock_metadata.get(symptom_lower, {'cui': None, 'semantic_type': 'Unknown', 'icd10': None})
            cui, semantic_type, icd10 = metadata['cui'], metadata['semantic_type'], metadata['icd10']
            # Cache mock data
            if self.client:
                try:
                    self.umls_cache.update_one(
                        {'symptom': symptom_lower},
                        {'$set': {
                            'cui': cui,
                            'semantic_type': semantic_type,
                            'icd10': icd10
                        }},
                        upsert=True
                    )
                    logger.debug(f"Cached mock UMLS metadata for '{symptom_lower}'")
                except Exception as e:
                    logger.error(f"Failed to cache mock UMLS metadata: {str(e)}")
            return cui, semantic_type, icd10

        # UMLS API call
        try:
            ticket = self._get_uts_ticket()
            if not ticket:
                return None, 'Unknown', None
            cui_url = f"{self.uts_base_url}/search/current"
            params = {'string': symptom_lower, 'ticket': ticket, 'searchType': 'exact', 'sabs': 'SNOMEDCT_US'}
            response = self.session.get(cui_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            cui = semantic_type = icd10 = None
            if data.get('result', {}).get('results'):
                cui = data['result']['results'][0]['ui']
                concept_url = f"{self.uts_base_url}/content/current/CUI/{cui}"
                concept_response = self.session.get(concept_url, params={'ticket': ticket}, timeout=10)
                concept_response.raise_for_status()
                concept_data = concept_response.json()
                semantic_type = concept_data['result'].get('semanticTypes', [{}])[0].get('name', 'Unknown')
                # Fetch ICD-10 mapping (simplified, assumes SNOMED-to-ICD10 mapping available)
                xref_url = f"{self.uts_base_url}/crosswalk/current/source/SNOMEDCT_US/{cui}"
                xref_response = self.session.get(xref_url, params={'ticket': ticket, 'targetSource': 'ICD10CM'}, timeout=10)
                if xref_response.status_code == 200:
                    xref_data = xref_response.json()
                    icd10 = xref_data['result'][0].get('ui', None) if xref_data.get('result') else None
                time.sleep(0.2)  # Rate limiting
            
            # Cache result
            if self.client and cui:
                try:
                    self.umls_cache.update_one(
                        {'symptom': symptom_lower},
                        {'$set': {
                            'cui': cui,
                            'semantic_type': semantic_type,
                            'icd10': icd10
                        }},
                        upsert=True
                    )
                    logger.debug(f"Cached UMLS metadata for '{symptom_lower}'")
                except Exception as e:
                    logger.error(f"Failed to cache UMLS metadata: {str(e)}")
            
            return cui, semantic_type, icd10
        except Exception as e:
            logger.error(f"UMLS metadata retrieval failed for '{symptom_lower}': {str(e)}")
            return None, 'Unknown', None

    def validate_symptom_data(self, symptom: str, category: str, synonyms: List[str], cui: Optional[str], semantic_type: str) -> bool:
        """Validate symptom data before updating."""
        symptom_lower = symptom.lower().strip()
        if not symptom_lower or len(symptom_lower) < 2:
            logger.error(f"Invalid symptom name: '{symptom_lower}'")
            return False
        if category not in [
            'musculoskeletal', 'respiratory', 'gastrointestinal', 'cardiovascular',
            'neurological', 'dermatological', 'sensory', 'hematologic', 'endocrine',
            'genitourinary', 'psychiatric', 'general', 'infectious'
        ]:
            logger.warning(f"Invalid category: '{category}'. Defaulting to 'general'.")
            return False
        if not all(isinstance(s, str) and len(s) > 2 for s in synonyms):
            logger.error(f"Invalid synonyms for '{symptom_lower}': {synonyms}")
            return False
        if cui and not cui.startswith('C'):
            logger.warning(f"Invalid UMLS CUI for '{symptom_lower}': {cui}")
            return False
        if semantic_type not in ['Sign or Symptom', 'Disease or Syndrome', 'Finding', 'Unknown']:
            logger.warning(f"Unusual semantic type for '{symptom_lower}': {semantic_type}")
        return True

    def update_knowledge_base(self, symptom: str, category: str = None, synonyms: List[str] = None, context: str = "") -> bool:
        """Update knowledge base with a new symptom, including UMLS metadata and ICD-10."""
        try:
            symptom_lower = symptom.lower().strip()
            if not self.is_new_symptom(symptom_lower, context):
                logger.debug(f"Symptom '{symptom_lower}' already exists.")
                return False

            # Infer category if not provided
            category = category or self.infer_category(symptom_lower, context)
            if category not in self.symptoms:
                self.symptoms[category] = {}

            # Generate synonyms if not provided
            synonyms = synonyms or self.generate_synonyms(symptom_lower)

            # Get UMLS metadata and ICD-10
            cui, semantic_type, icd10 = self.get_umls_metadata(symptom_lower)

            # Validate data
            if not self.validate_symptom_data(symptom_lower, category, synonyms, cui, semantic_type):
                logger.error(f"Validation failed for symptom '{symptom_lower}'")
                return False

            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Update in-memory knowledge base
            self.medical_terms.append({
                'term': symptom_lower,
                'category': category,
                'umls_cui': cui,
                'semantic_type': semantic_type,
                'icd10': icd10
            })
            self.synonyms[symptom_lower] = synonyms
            self.symptoms[category][symptom_lower] = {
                'description': f"UMLS-derived: {symptom_lower}",
                'umls_cui': cui,
                'semantic_type': semantic_type,
                'icd10': icd10
            }
            self.knowledge_base['medical_terms'] = self.medical_terms
            self.knowledge_base['synonyms'][symptom_lower] = synonyms
            self.knowledge_base['symptoms'] = self.symptoms
            self.knowledge_base['version'] = self.version
            self.knowledge_base['last_updated'] = current_date

            # Update clinical pathways
            if 'clinical_pathways' not in self.knowledge_base:
                self.knowledge_base['clinical_pathways'] = {}
            if category not in self.knowledge_base['clinical_pathways']:
                self.knowledge_base['clinical_pathways'][category] = {}
            self.knowledge_base['clinical_pathways'][category][symptom_lower] = {
                'differentials': [f"{symptom_lower} disorder", f"Possible {category} condition"],
                'contextual_triggers': [f"Recent onset of {symptom_lower}"],
                'required_symptoms': [symptom_lower],
                'exclusion_criteria': [],
                'workup': {
                    'urgent': [f"Assess {symptom_lower} severity"],
                    'routine': [f"Diagnostic test for {category} conditions"]
                },
                'management': {
                    'symptomatic': [f"Symptomatic relief for {symptom_lower}"],
                    'definitive': [],
                    'lifestyle': ["Monitor symptom progression"]
                },
                'follow_up': ["Follow-up in 1-2 weeks"],
                'references': ["Clinical guidelines pending"],
                'metadata': {
                    'source': ["Automated update"],
                    'last_updated': current_date,
                    'umls_cui': cui,
                    'icd10': icd10,
                    'version': self.version
                }
            }

            # Update MongoDB
            if self.client:
                try:
                    self.medical_terms_collection.update_one(
                        {'term': symptom_lower},
                        {'$set': {
                            'term': symptom_lower,
                            'category': category,
                            'umls_cui': cui,
                            'semantic_type': semantic_type,
                            'icd10': icd10
                        }},
                        upsert=True
                    )
                    self.synonyms_collection.update_one(
                        {'term': symptom_lower},  # Align with symptoms schema
                        {'$set': {
                            'term': symptom_lower,
                            'aliases': synonyms
                        }},
                        upsert=True
                    )
                    self.symptoms_collection.update_one(
                        {'category': category, 'symptom': symptom_lower},
                        {'$set': {
                            'category': category,
                            'symptom': symptom_lower,
                            'description': f"UMLS-derived: {symptom_lower}",
                            'umls_cui': cui,
                            'semantic_type': semantic_type,
                            'icd10': icd10
                        }},
                        upsert=True
                    )
                    self.pathways_collection.update_one(
                        {'category': category, 'key': symptom_lower},
                        {'$set': {
                            'category': category,
                            'key': symptom_lower,
                            'path': self.knowledge_base['clinical_pathways'][category][symptom_lower]
                        }},
                        upsert=True
                    )
                    logger.debug(f"Updated MongoDB for symptom '{symptom_lower}'")
                except ConnectionFailure as e:
                    logger.warning(f"MongoDB connection failed: {str(e)}. Saving to JSON.")
                    save_knowledge_base(self.knowledge_base)
                except Exception as e:
                    logger.error(f"Error updating MongoDB: {str(e)}. Saving to JSON.")
                    save_knowledge_base(self.knowledge_base)
            else:
                logger.warning("No MongoDB connection. Saving to JSON.")
                save_knowledge_base(self.knowledge_base)

            # Save to JSON as fallback
            save_knowledge_base(self.knowledge_base)
            logger.info(f"Added new symptom '{symptom_lower}' to knowledge base (category: {category}, CUI: {cui}, ICD-10: {icd10}, semantic_type: {semantic_type})")
            return True
        except Exception as e:
            logger.error(f"Error updating knowledge base for '{symptom}': {str(e)}")
            return False

    def batch_update_knowledge_base(self, symptoms_data: List[Dict]) -> List[str]:
        """Batch update multiple symptoms."""
        failed = []
        for data in symptoms_data:
            symptom = data.get('symptom')
            category = data.get('category')
            synonyms = data.get('synonyms', [])
            context = data.get('context', '')
            if not symptom:
                logger.error("Missing symptom in batch data")
                failed.append("Unknown")
                continue
            if not self.update_knowledge_base(symptom, category, synonyms, context):
                failed.append(symptom)
        return failed