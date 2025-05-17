# departments/nlp/kb_updater.py
from typing import List, Dict, Set, Optional
import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from departments.nlp.logging_setup import logger
from departments.nlp.knowledge_base import load_knowledge_base, save_knowledge_base
from departments.nlp.nlp_utils import embed_text
from departments.nlp.models.transformer_model import model, tokenizer
import torch
import requests
from departments.nlp.config import UTS_API_KEY, UTS_BASE_URL

class KnowledgeBaseUpdater:
    def __init__(self, mongo_uri: str = None, db_name: str = 'clinical_db', kb_prefix: str = 'kb_'):
        self.mongo_uri = mongo_uri or os.getenv('MONGO_URI', 'mongodb://localhost:27017')
        self.db_name = db_name
        self.kb_prefix = kb_prefix
        self.model = model
        self.tokenizer = tokenizer
        self.uts_api_key = UTS_API_KEY or 'mock_api_key'  # Fallback for testing
        self.uts_base_url = UTS_BASE_URL or 'http://mock-uts.local/rest'
        self.knowledge_base = load_knowledge_base()
        self.medical_terms = self.knowledge_base.get('medical_terms', set())
        self.synonyms = self.knowledge_base.get('synonyms', {})
        try:
            self.client = MongoClient(self.mongo_uri)
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.medical_terms_collection = self.db[f'{kb_prefix}medical_terms']
            self.synonyms_collection = self.db[f'{kb_prefix}synonyms']
            self.pathways_collection = self.db[f'{kb_prefix}clinical_pathways']
            logger.info("Connected to MongoDB for knowledge base updates.")
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {str(e)}. Falling back to JSON.")
            self.client = None
        except Exception as e:
            logger.error(f"Error initializing MongoDB: {str(e)}")
            raise

    def is_new_symptom(self, symptom: str) -> bool:
        """Check if a symptom is new (not in medical_terms or synonyms)."""
        symptom_lower = symptom.lower().strip()
        if symptom_lower in self.medical_terms:
            return False
        for key, aliases in self.synonyms.items():
            if symptom_lower == key or symptom_lower in [a.lower() for a in aliases]:
                return False
        return True

    def infer_category(self, symptom: str, context: str) -> str:
        """Infer symptom category using transformer model."""
        try:
            text = f"{symptom} {context}".lower()
            embedding = embed_text(text)
            categories = ['musculoskeletal', 'respiratory', 'gastrointestinal', 'cardiovascular', 'neurological', 'dermatological']
            category_scores = {}
            for cat in categories:
                cat_embedding = embed_text(cat)
                similarity = torch.cosine_similarity(embedding.unsqueeze(0), cat_embedding.unsqueeze(0)).item()
                category_scores[cat] = similarity
            return max(category_scores, key=category_scores.get)
        except Exception as e:
            logger.error(f"Error inferring category for {symptom}: {str(e)}")
            return 'unknown'

    def generate_synonyms(self, symptom: str) -> List[str]:
        """Generate synonyms using UMLS UTS API or mock API."""
        try:
            symptom_lower = symptom.lower().strip()
            if self.uts_api_key == 'mock_api_key':
                return self._mock_uts_synonyms(symptom_lower)
            ticket = self._get_uts_ticket()
            url = f"{self.uts_base_url}/search/current"
            params = {
                'string': symptom_lower,
                'ticket': ticket,
                'searchType': 'exact',
                'sabs': 'SNOMEDCT_US'  # Limit to SNOMED CT for clinical terms
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
            logger.info(f"UMLS synonyms for '{symptom_lower}': {synonyms}")
            return list(synonyms) or self._generate_synonyms_fallback(symptom_lower)
        except Exception as e:
            logger.error(f"UMLS synonym generation failed for '{symptom_lower}': {str(e)}")
            return self._generate_synonyms_fallback(symptom_lower)

    def _get_uts_ticket(self) -> str:
        """Obtain a single-use UTS ticket."""
        if self.uts_api_key == 'mock_api_key':
            return 'mock_ticket'
        ticket_url = f"{self.uts_base_url}/authentication/ticket"
        response = requests.post(ticket_url, data={'apiKey': self.uts_api_key})
        response.raise_for_status()
        return response.text

    def _mock_uts_synonyms(self, symptom: str) -> List[str]:
        """Mock UMLS API for testing."""
        mock_data = {
            'chest tightness': ['dyspnea', 'shortness of breath'],
            'back pain': ['lumbago', 'backache', 'lower back pain'],
            'obesity': ['overweight', 'high bmi']
        }
        synonyms = mock_data.get(symptom.lower(), [])
        logger.debug(f"Mock UMLS synonyms for '{symptom}': {synonyms}")
        return synonyms

    def _generate_synonyms_fallback(self, symptom: str) -> List[str]:
        """Fallback synonym generation (existing logic)."""
        try:
            symptom_lower = symptom.lower()
            synonyms = [f"{symptom_lower} syndrome", f"acute {symptom_lower}", f"chronic {symptom_lower}"]
            return [s for s in synonyms if s != symptom_lower]
        except Exception as e:
            logger.error(f"Error generating fallback synonyms for {symptom}: {str(e)}")
            return []

    def update_knowledge_base(self, symptom: str, category: str, synonyms: List[str], context: str) -> bool:
        """Update knowledge base with a new symptom, including UMLS metadata."""
        try:
            symptom_lower = symptom.lower().strip()
            if not self.is_new_symptom(symptom_lower):
                logger.debug(f"Symptom {symptom_lower} already exists.")
                return False

            # Get UMLS CUI and semantic type
            cui = semantic_type = None
            if self.uts_api_key != 'mock_api_key':
                try:
                    ticket = self._get_uts_ticket()
                    cui_url = f"{self.uts_base_url}/search/current"
                    params = {'string': symptom_lower, 'ticket': ticket, 'searchType': 'exact'}
                    response = requests.get(cui_url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    if data.get('result', {}).get('results'):
                        cui = data['result']['results'][0]['ui']
                        concept_url = f"{self.uts_base_url}/content/current/CUI/{cui}"
                        concept_response = requests.get(concept_url, params={'ticket': ticket})
                        concept_response.raise_for_status()
                        concept_data = concept_response.json()
                        semantic_type = concept_data['result'].get('semanticTypes', [{}])[0].get('name', 'Unknown')
                except Exception as e:
                    logger.error(f"UMLS metadata retrieval failed for '{symptom_lower}': {str(e)}")
            else:
                # Mock UMLS metadata
                mock_metadata = {
                    'chest tightness': {'cui': 'C0242209', 'semantic_type': 'Sign or Symptom'},
                    'back pain': {'cui': 'C0004604', 'semantic_type': 'Sign or Symptom'},
                    'obesity': {'cui': 'C0028754', 'semantic_type': 'Disease or Syndrome'}
                }
                metadata = mock_metadata.get(symptom_lower, {'cui': None, 'semantic_type': 'Unknown'})
                cui, semantic_type = metadata['cui'], metadata['semantic_type']

            # Update in-memory knowledge base
            self.medical_terms.add(symptom_lower)
            self.synonyms[symptom_lower] = synonyms
            self.knowledge_base['medical_terms'] = self.medical_terms
            self.knowledge_base['synonyms'][symptom_lower] = synonyms

            # Add to clinical pathways
            if 'clinical_pathways' not in self.knowledge_base:
                self.knowledge_base['clinical_pathways'] = {}
            if category not in self.knowledge_base['clinical_pathways']:
                self.knowledge_base['clinical_pathways'][category] = {}
            self.knowledge_base['clinical_pathways'][category][symptom_lower] = {
                'differentials': [f"Possible {symptom_lower} disorder"],
                'workup': {'urgent': [], 'routine': [f"Evaluate {symptom_lower}"]},
                'management': {'symptomatic': [], 'definitive': [], 'lifestyle': []},
                'follow_up': ['Follow-up in 2 weeks'],
                'references': [],
                'umls_cui': cui  # New field
            }

            # Update MongoDB
            if self.client:
                try:
                    self.medical_terms_collection.insert_one({
                        'term': symptom_lower,
                        'category': category,
                        'umls_cui': cui,
                        'semantic_type': semantic_type
                    })
                    self.synonyms_collection.insert_one({
                        'key': symptom_lower,
                        'aliases': synonyms
                    })
                    self.pathways_collection.insert_one({
                        'category': category,
                        'key': symptom_lower,
                        'path': self.knowledge_base['clinical_pathways'][category][symptom_lower]
                    })
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
            logger.info(f"Added new symptom {symptom_lower} to knowledge base (category: {category}, UMLS CUI: {cui}).")
            return True
        except Exception as e:
            logger.error(f"Error updating knowledge base for {symptom}: {str(e)}")
            return False