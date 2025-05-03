from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017')
db = client['clinical_db']
db.symptoms.delete_many({})
db.symptoms.insert_many([
    {"category": "respiratory", "symptom": "facial pain", "description": "Pain in the facial region, often over sinuses"},
    {"category": "respiratory", "symptom": "nasal congestion", "description": "Blocked or stuffy nose"},
    {"category": "respiratory", "symptom": "purulent nasal discharge", "description": "Yellow or green nasal discharge"},
    {"category": "respiratory", "symptom": "fever", "description": "Elevated body temperature"},
    {"category": "neurological", "symptom": "headache", "description": "Pain in the head or neck"}
])