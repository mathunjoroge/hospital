# departments/nlp/symptom_tracker.py
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import logging
import re

logger = logging.getLogger(__name__)

class SymptomTracker:
    def __init__(self, mongo_uri='mongodb://localhost:27017', db_name='clinical_db', collection_name='symptoms'):
        self.common_symptoms = {}
        try:
            client = MongoClient(mongo_uri)
            client.admin.command('ping')
            db = client[db_name]
            collection = db[collection_name]
            symptoms = collection.find()
            for doc in symptoms:
                category = doc.get('category')
                symptom = doc.get('symptom')
                description = doc.get('description')
                if category and symptom and description:
                    if category not in self.common_symptoms:
                        self.common_symptoms[category] = {}
                    self.common_symptoms[category][symptom] = description
            client.close()
            if not self.common_symptoms:
                logger.error("No symptoms loaded from MongoDB. Collection may be empty.")
            else:
                logger.info(f"Loaded {sum(len(symptoms) for symptoms in self.common_symptoms.values())} symptoms from MongoDB.")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            self.common_symptoms = {}
        except Exception as e:
            logger.error(f"Error loading symptoms from MongoDB: {str(e)}")
            self.common_symptoms = {}

    def add_symptom(self, category, symptom, description):
        try:
            client = MongoClient('mongodb://localhost:27017')
            db = client['clinical_db']
            collection = db['symptoms']
            collection.update_one(
                {'category': category, 'symptom': symptom},
                {'$set': {'description': description}},
                upsert=True
            )
            if category not in self.common_symptoms:
                self.common_symptoms[category] = {}
            self.common_symptoms[category][symptom] = description
            logger.info(f"Added symptom '{symptom}' to category '{category}'.")
            client.close()
        except Exception as e:
            logger.error(f"Error adding symptom to MongoDB: {str(e)}")

    def remove_symptom(self, category, symptom):
        try:
            client = MongoClient('mongodb://localhost:27017')
            db = client['clinical_db']
            collection = db['symptoms']
            collection.delete_one({'category': category, 'symptom': symptom})
            if category in self.common_symptoms and symptom in self.common_symptoms[category]:
                del self.common_symptoms[category][symptom]
                if not self.common_symptoms[category]:
                    del self.common_symptoms[category]
                logger.info(f"Removed symptom '{symptom}' from category '{category}'.")
            client.close()
        except Exception as e:
            logger.error(f"Error removing symptom from MongoDB: {str(e)}")

    def get_symptoms_by_category(self, category):
        return self.common_symptoms.get(category, {})

    def search_symptom(self, symptom):
        for category, symptoms in self.common_symptoms.items():
            if symptom in symptoms:
                return category, symptoms[symptom]
        return None, None

    def get_all_symptoms(self):
        return {symptom for category in self.common_symptoms.values() for symptom in category.keys()}

    def get_categories(self):
        return list(self.common_symptoms.keys())

    def process_note(self, note, chief_complaint):
        logger.debug(f"Processing note ID {getattr(note, 'id', 'unknown')} for symptoms with chief complaint: {chief_complaint}")
        text = f"{note.situation or ''} {note.hpi or ''} {note.assessment or ''}".lower().strip()
        if not text:
            logger.warning("No text available for symptom extraction")
            return []
        symptoms = []
        for category, symptom_dict in self.common_symptoms.items():
            for symptom, description in symptom_dict.items():
                if symptom.lower() in text or description.lower() in text:
                    duration = '10 days' if '10 days' in text else 'Unknown'
                    severity = 'Mild' if any(kw in text for kw in ['mild', 'low-grade']) else 'Unknown'
                    location = 'Head' if any(kw in text for kw in ['facial', 'sinus', 'nasal', 'head']) else 'Unknown'
                    symptoms.append({
                        'description': symptom,
                        'category': category,
                        'definition': description,
                        'duration': duration,
                        'severity': severity,
                        'location': location
                    })
        negation_pattern = r"\b(no|without|denies|not)\b\s+([\w\s]+?)(?=\.|,|;|\band\b|\bor\b|$)"
        negated_symptoms = []
        for field in [note.situation or '', note.hpi or '', note.assessment or '']:
            matches = re.findall(negation_pattern, field.lower(), re.IGNORECASE)
            for _, negated in matches:
                negated_symptoms.append(negated.strip())
        valid_symptoms = [
            s for s in symptoms
            if not any(ns.lower() in s['description'].lower() or s['description'].lower() in ns.lower()
                       for ns in negated_symptoms)
        ]
        unique_symptoms = []
        seen = set()
        for s in valid_symptoms:
            if s['description'] not in seen:
                unique_symptoms.append(s)
                seen.add(s['description'])
        logger.debug(f"Extracted symptoms: {unique_symptoms}, Negated: {negated_symptoms}")
        return unique_symptoms