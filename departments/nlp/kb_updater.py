from typing import List, Dict, Optional
import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from departments.nlp.logging_setup import get_logger
logger = get_logger()
from departments.nlp.knowledge_base_io import load_knowledge_base, save_knowledge_base
from departments.nlp.nlp_utils import embed_text
from departments.nlp.models.transformer_model import model, tokenizer
import torch
import requests
from departments.nlp.config import UTS_API_KEY, UTS_BASE_URL
from departments.nlp.nlp_pipeline import get_nlp

class KnowledgeBaseUpdater:
    def __init__(self, mongo_uri: str = None, db_name: str = 'clinical_db', kb_prefix: str = 'kb_'):
        self.mongo_uri = mongo_uri or os.getenv('MONGO_URI', 'mongodb://localhost:27017')
        self.db_name = db_name
        self.kb_prefix = kb_prefix
        self.model = model
        self.tokenizer = tokenizer
        self.uts_api_key = UTS_API_KEY or 'mock_api_key'
        self.uts_base_url = UTS_BASE_URL or 'https://uts-ws.nlm.nih.gov/rest'
        self.knowledge_base = load_knowledge_base()
        self.medical_terms = self.knowledge_base.get('medical_terms', [])  # List of dicts
        self.synonyms = self.knowledge_base.get('synonyms', {})

        # Initialize MongoDB
        try:
            self.client = MongoClient(self.mongo_uri)
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.medical_terms_collection = self.db[f'{kb_prefix}medical_terms']
            self.synonyms_collection = self.db[f'{kb_prefix}synonyms']
            self.pathways_collection = self.db[f'{kb_prefix}clinical_pathways']
            self.umls_cache = self.db['umls_cache']  # Consistent with symptom_tracker.py
            logger.info("Connected to MongoDB for knowledge base updates.")
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {str(e)}. Falling back to JSON.")
            self.client = None
        except Exception as e:
            logger.error(f"Error initializing MongoDB: {str(e)}")
            raise

    def is_new_symptom(self, symptom: str, context: str = "") -> bool:
        """Check if a symptom is new."""
        symptom_lower = symptom.lower().strip()
        # Check medical_terms (list of dicts)
        for term in self.medical_terms:
            if term.get('term', '').lower() == symptom_lower:
                return False
        # Check synonyms
        for key, aliases in self.synonyms.items():
            if symptom_lower == key or symptom_lower in [a.lower() for a in aliases]:
                return False
        logger.debug(f"Symptom '{symptom_lower}' is new.")
        return True

    def infer_category(self, symptom: str, context: str) -> str:
        """Infer symptom category using transformer model with optimized embeddings."""
        try:
            text = f"{symptom} {context}".lower()[:512]  # Truncate to save memory
            embedding = embed_text(text)
            categories = [
                'musculoskeletal', 'respiratory', 'gastrointestinal',
                'cardiovascular', 'neurological', 'dermatological',
                'sensory', 'hematologic', 'endocrine', 'genitourinary', 'psychiatric'
            ]
            category_scores = {}
            for cat in categories:
                cat_embedding = embed_text(cat)
                similarity = torch.cosine_similarity(
                    embedding.unsqueeze(0), cat_embedding.unsqueeze(0)
                ).item()
                category_scores[cat] = similarity
            inferred_category = max(category_scores, key=category_scores.get)
            logger.debug(f"Inferred category for '{symptom}': {inferred_category}")
            return inferred_category
        except Exception as e:
            logger.error(f"Error inferring category for '{symptom}': {str(e)}")
            return 'general'

    def generate_synonyms(self, symptom: str) -> List[str]:
        """Generate synonyms using UMLS UTS API with caching."""
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
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                synonyms = set()
                if data.get('result', {}).get('results'):
                    cui = data['result']['results'][0]['ui']
                    atoms_url = f"{self.uts_base_url}/content/current/CUI/{cui}/atoms"
                    atoms_response = requests.get(atoms_url, params={'ticket': ticket})
                    atoms_response.raise_for_status()
                    atoms_data = atoms_response.json()
                    for atom in atoms_data.get('result', []):
                        synonyms.add(atom['name'].lower())
                synonyms = [s for s in synonyms if s != symptom_lower and len(s) > 2]
                if not synonyms:
                    synonyms = self._generate_synonyms_fallback(symptom_lower)
            
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
        """Obtain a single-use UTS ticket."""
        if self.uts_api_key == 'mock_api_key':
            return 'mock_ticket'
        try:
            ticket_url = f"{self.uts_base_url}/tickets"
            response = requests.post(ticket_url, data={'apiKey': self.uts_api_key})
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Failed to get UTS ticket: {str(e)}")
            return ''

    def _mock_uts_synonyms(self, symptom: str) -> List[str]:
        """Mock UMLS API for testing."""
        mock_data = {
            'chest tightness': ['dyspnea', 'shortness of breath', 'chest discomfort'],
            'back pain': ['lumbago', 'backache', 'lower back pain'],
            'obesity': ['overweight', 'high bmi', 'adiposity']
        }
        synonyms = mock_data.get(symptom.lower(), [])
        logger.debug(f"Mock UMLS synonyms for '{symptom}': {synonyms}")
        return synonyms

    def _generate_synonyms_fallback(self, symptom: str) -> List[str]:
        """Fallback synonym generation."""
        try:
            symptom_lower = symptom.lower()
            synonyms = [f"{symptom_lower} syndrome", f"acute {symptom_lower}", f"chronic {symptom_lower}"]
            return [s for s in synonyms if s != symptom_lower]
        except Exception as e:
            logger.error(f"Error generating fallback synonyms for '{symptom}': {str(e)}")
            return []

    def get_umls_metadata(self, symptom: str) -> tuple[Optional[str], Optional[str]]:
        """Retrieve UMLS CUI and semantic type with caching."""
        symptom_lower = symptom.lower().strip()
        
        # Check cache
        try:
            if self.client:
                cached = self.umls_cache.find_one({'symptom': symptom_lower})
                if cached and 'cui' in cached:
                    logger.debug(f"Retrieved cached UMLS metadata for '{symptom_lower}'")
                    return cached['cui'], cached['semantic_type']
        except Exception as e:
            logger.error(f"Error checking UMLS cache for '{symptom_lower}': {str(e)}")

        # Mock data for testing
        if self.uts_api_key == 'mock_api_key':
            mock_metadata = {
                'chest tightness': {'cui': 'C0242209', 'semantic_type': 'Sign or Symptom'},
                'back pain': {'cui': 'C0004604', 'semantic_type': 'Sign or Symptom'},
                'obesity': {'cui': 'C0028754', 'semantic_type': 'Disease or Syndrome'}
            }
            metadata = mock_metadata.get(symptom_lower, {'cui': None, 'semantic_type': 'Unknown'})
            cui, semantic_type = metadata['cui'], metadata['semantic_type']
            # Cache mock data
            if self.client:
                try:
                    self.umls_cache.update_one(
                        {'symptom': symptom_lower},
                        {'$set': {
                            'cui': cui,
                            'semantic_type': semantic_type
                        }},
                        upsert=True
                    )
                    logger.debug(f"Cached mock UMLS metadata for '{symptom_lower}'")
                except Exception as e:
                    logger.error(f"Failed to cache mock UMLS metadata: {str(e)}")
            return cui, semantic_type

        # UMLS API call
        try:
            ticket = self._get_uts_ticket()
            if not ticket:
                return None, 'Unknown'
            cui_url = f"{self.uts_base_url}/search/current"
            params = {'string': symptom_lower, 'ticket': ticket, 'searchType': 'exact', 'sabs': 'SNOMEDCT_US'}
            response = requests.get(cui_url, params=params)
            response.raise_for_status()
            data = response.json()
            cui = semantic_type = None
            if data.get('result', {}).get('results'):
                cui = data['result']['results'][0]['ui']
                concept_url = f"{self.uts_base_url}/content/current/CUI/{cui}"
                concept_response = requests.get(concept_url, params={'ticket': ticket})
                concept_response.raise_for_status()
                concept_data = concept_response.json()
                semantic_type = concept_data['result'].get('semanticTypes', [{}])[0].get('name', 'Unknown')
            
            # Cache result
            if self.client and cui:
                try:
                    self.umls_cache.update_one(
                        {'symptom': symptom_lower},
                        {'$set': {
                            'cui': cui,
                            'semantic_type': semantic_type
                        }},
                        upsert=True
                    )
                    logger.debug(f"Cached UMLS metadata for '{symptom_lower}'")
                except Exception as e:
                    logger.error(f"Failed to cache UMLS metadata: {str(e)}")
            
            return cui, semantic_type
        except Exception as e:
            logger.error(f"UMLS metadata retrieval failed for '{symptom_lower}': {str(e)}")
            return None, 'Unknown'

    def update_knowledge_base(self, symptom: str, category: str, synonyms: List[str], context: str) -> bool:
        """Update knowledge base with a new symptom, including UMLS metadata."""
        try:
            symptom_lower = symptom.lower().strip()
            if not self.is_new_symptom(symptom_lower, context):
                logger.debug(f"Symptom '{symptom_lower}' already exists.")
                return False

            # Get UMLS metadata
            cui, semantic_type = self.get_umls_metadata(symptom_lower)

            # Update in-memory knowledge base (medical_terms as list of dicts)
            self.medical_terms.append({
                'term': symptom_lower,
                'category': category,
                'umls_cui': cui,
                'semantic_type': semantic_type
            })
            self.synonyms[symptom_lower] = synonyms
            self.knowledge_base['medical_terms'] = self.medical_terms
            self.knowledge_base['synonyms'][symptom_lower] = synonyms

            # Update clinical pathways with realistic data
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
                    'last_updated': "2025-05-24",
                    'umls_cui': cui
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
                            'semantic_type': semantic_type
                        }},
                        upsert=True
                    )
                    self.synonyms_collection.update_one(
                        {'key': symptom_lower},
                        {'$set': {
                            'key': symptom_lower,
                            'aliases': synonyms
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
            logger.info(f"Added new symptom '{symptom_lower}' to knowledge base (category: {category}, CUI: {cui}, semantic_type: {semantic_type})")
            return True
        except Exception as e:
            logger.error(f"Error updating knowledge base for '{symptom}': {str(e)}")
            return False