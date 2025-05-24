from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "clinical_db")

# Default configurations
default_diagnosis_relevance = {
    "acute bacterial sinusitis": [
        {"symptom": "facial pain", "weight": 0.4},
        {"symptom": "nasal congestion", "weight": 0.3},
        {"symptom": "purulent nasal discharge", "weight": 0.3},
        {"symptom": "fever", "weight": 0.1},
        {"symptom": "headache", "weight": 0.05}
    ],
    "viral hepatitis": [
        {"symptom": "jaundice", "weight": 0.35},
        {"symptom": "nausea", "weight": 0.25},
        {"symptom": "loss of appetite", "weight": 0.2},
        {"symptom": "fatigue", "weight": 0.1},
        {"symptom": "fatty food intolerance", "weight": 0.1}
    ]
}
default_management_config = {
    "follow_up_default": "Follow-up in 2 weeks",
    "follow_up_urgent": "Follow-up in 24-48 hours or sooner if symptoms worsen",
    "urgent_threshold": "0.7",
    "min_symptom_match": 0.5,
    "critical_symptoms": ["fever", "jaundice", "chest pain", "shortness of breath"]
}

# Connect to MongoDB
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    db = client[DB_NAME]
    collection = db['configurations']

    # Insert or update configurations
    collection.update_one(
        {'config_type': 'diagnosis_relevance'},
        {'$set': {'config_type': 'diagnosis_relevance', 'data': default_diagnosis_relevance}},
        upsert=True
    )
    collection.update_one(
        {'config_type': 'management_config'},
        {'$set': {'config_type': 'management_config', 'data': default_management_config}},
        upsert=True
    )
    print("Successfully initialized configurations in MongoDB.")
except Exception as e:
    print(f"Failed to initialize configurations: {str(e)}")
finally:
    client.close()