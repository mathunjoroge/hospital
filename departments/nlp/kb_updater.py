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

class KnowledgeBaseUpdater:
    def __init__(self, mongo_uri: str = None, db_name: str = 'clinical_db', kb_prefix: str = 'kb_'):
        self.mongo_uri = mongo_uri or os.getenv('MONGO_URI', 'mongodb://localhost:27017')
        self.db_name = db_name
        self.kb_prefix = kb_prefix
        self.model = model
        self.tokenizer = tokenizer
        self.knowledge_base = load_knowledge_base()
        self.medical_terms = self.knowledge_base.get('medical_terms', set())
        self.synonyms = self.knowledge_base.get('synonyms', {})

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
        """Generate synonyms for a symptom (placeholder for ontology API)."""
        try:
            symptom_lower = symptom.lower()
            synonyms = [f"{symptom_lower} syndrome", f"acute {symptom_lower}", f"chronic {symptom_lower}"]
            return [s for s in synonyms if s != symptom_lower]
        except Exception as e:
            logger.error(f"Error generating synonyms for {symptom}: {str(e)}")
            return []

    def update_knowledge_base(self, symptom: str, category: str, synonyms: List[str], context: str) -> bool:
        """Update knowledge base with a new symptom."""
        try:
            symptom_lower = symptom.lower().strip()
            if not self.is_new_symptom(symptom_lower):
                logger.debug(f"Symptom {symptom_lower} already exists.")
                return False

            # Update in-memory knowledge base
            self.medical_terms.add(symptom_lower)
            self.synonyms[symptom_lower] = synonyms
            self.knowledge_base['medical_terms'] = self.medical_terms
            self.knowledge_base['synonyms'][symptom_lower] = synonyms

            # Add to clinical pathways (basic differential and management)
            if 'clinical_pathways' not in self.knowledge_base:
                self.knowledge_base['clinical_pathways'] = {}
            if category not in self.knowledge_base['clinical_pathways']:
                self.knowledge_base['clinical_pathways'][category] = {}
            self.knowledge_base['clinical_pathways'][category][symptom_lower] = {
                'differentials': [f"Possible {symptom_lower} disorder"],
                'workup': {'urgent': [], 'routine': [f"Evaluate {symptom_lower}"]},
                'management': {'symptomatic': [], 'definitive': [], 'lifestyle': []},
                'follow_up': ['Follow-up in 2 weeks'],
                'references': []
            }

            # Update MongoDB
            try:
                client = MongoClient(self.mongo_uri)
                db = client[self.db_name]
                db[f'{self.kb_prefix}medical_terms'].insert_one({'term': symptom_lower})
                db[f'{self.kb_prefix}synonyms'].insert_one({'key': symptom_lower, 'aliases': synonyms})
                db[f'{self.kb_prefix}clinical_pathways'].insert_one({
                    'category': category,
                    'key': symptom_lower,
                    'path': self.knowledge_base['clinical_pathways'][category][symptom_lower]
                })
                client.close()
            except ConnectionFailure as e:
                logger.warning(f"MongoDB connection failed: {str(e)}. Saving to JSON.")
                save_knowledge_base(self.knowledge_base)
            except Exception as e:
                logger.error(f"Error updating MongoDB: {str(e)}. Saving to JSON.")
                save_knowledge_base(self.knowledge_base)

            # Save to JSON as fallback
            save_knowledge_base(self.knowledge_base)
            logger.info(f"Added new symptom {symptom_lower} to knowledge base (category: {category}).")
            return True
        except Exception as e:
            logger.error(f"Error updating knowledge base for {symptom}: {str(e)}")
            return False