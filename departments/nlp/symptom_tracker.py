# departments/nlp/symptom_tracker.py
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import logging

# Set up logging
logger = logging.getLogger(__name__)

class SymptomTracker:
    def __init__(self, mongo_uri='mongodb://localhost:27017', db_name='clinical_db', collection_name='symptoms'):
        """Initialize SymptomTracker by loading symptoms from MongoDB."""
        self.common_symptoms = {}
        try:
            # Connect to MongoDB
            client = MongoClient(mongo_uri)
            # Test connection
            client.admin.command('ping')
            db = client[db_name]
            collection = db[collection_name]

            # Load symptoms
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
            self.common_symptoms = {}  # Fallback to empty dict
        except Exception as e:
            logger.error(f"Error loading symptoms from MongoDB: {str(e)}")
            self.common_symptoms = {}

    def add_symptom(self, category, symptom, description):
        """Add a new symptom to the specified category and update MongoDB."""
        try:
            client = MongoClient('mongodb://localhost:27017')  # Update with your connection string
            db = client['clinical_db']
            collection = db['symptoms']
            # Add to MongoDB
            collection.update_one(
                {'category': category, 'symptom': symptom},
                {'$set': {'description': description}},
                upsert=True
            )
            # Add to in-memory dict
            if category not in self.common_symptoms:
                self.common_symptoms[category] = {}
            self.common_symptoms[category][symptom] = description
            logger.info(f"Added symptom '{symptom}' to category '{category}'.")
            client.close()
        except Exception as e:
            logger.error(f"Error adding symptom to MongoDB: {str(e)}")

    def remove_symptom(self, category, symptom):
        """Remove a symptom from the specified category and MongoDB."""
        try:
            client = MongoClient('mongodb://localhost:27017')  # Update with your connection string
            db = client['clinical_db']
            collection = db['symptoms']
            # Remove from MongoDB
            collection.delete_one({'category': category, 'symptom': symptom})
            # Remove from in-memory dict
            if category in self.common_symptoms and symptom in self.common_symptoms[category]:
                del self.common_symptoms[category][symptom]
                if not self.common_symptoms[category]:
                    del self.common_symptoms[category]
                logger.info(f"Removed symptom '{symptom}' from category '{category}'.")
            client.close()
        except Exception as e:
            logger.error(f"Error removing symptom from MongoDB: {str(e)}")

    def get_symptoms_by_category(self, category):
        """Return all symptoms in a specific category."""
        return self.common_symptoms.get(category, {})

    def search_symptom(self, symptom):
        """Search for a symptom across all categories."""
        for category, symptoms in self.common_symptoms.items():
            if symptom in symptoms:
                return category, symptoms[symptom]
        return None, None

    def get_all_symptoms(self):
        """Return a flat set of all symptom names."""
        return {symptom for category in self.common_symptoms.values() for symptom in category.keys()}

    def get_categories(self):
        """Return a list of all symptom categories."""
        return list(self.common_symptoms.keys())