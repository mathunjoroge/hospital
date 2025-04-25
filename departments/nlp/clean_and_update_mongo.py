import json
import re
from pathlib import Path
from typing import List, Dict
from fuzzywuzzy import fuzz
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
from bson import ObjectId
from dotenv import load_dotenv
import os
from datetime import datetime
from transformers import pipeline
import torch
import sys

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_cleaning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Medical terms (for fallback rule-based filtering)
MEDICAL_TERMS = {
    'pain', 'fever', 'cough', 'fatigue', 'swelling', 'nausea', 'vomiting', 'diarrhea',
    'constipation', 'bleeding', 'rash', 'itch', 'dizziness', 'weakness', 'numbness',
    'tingling', 'shortness of breath', 'chest', 'headache', 'confusion', 'sleep',
    'appetite', 'weight', 'thirst', 'urination', 'vision', 'hearing', 'speech',
    'movement', 'tremor', 'palpitations', 'seizure', 'sweating', 'skin darkening',
    'staring spells', 'swelling near vagina', 'back pain', 'abdominal pain',
    'discoloration', 'bruising', 'tenderness', 'redness', 'irregular heartbeat',
    'pulsating sensation', 'excessive sweating', 'darkened skin', 'staring blankly',
    'velvety skin', 'sudden stop in movement', 'brief loss of awareness', 'jaundice',
    'nasal congestion', 'runny nose', 'sneezing', 'itchy eyes', 'wheezing',
    'sore throat', 'joint pain', 'muscle pain', 'chills', 'paleness', 'hair loss',
    'night sweats', 'difficulty swallowing', 'hoarseness', 'persistent cough',
    'inattention', 'hyperactivity', 'impulsivity', 'stiffness', 'sadness', 'anxiety',
    'depression', 'behavioral changes', 'hallucinations', 'delusions', 'weight gain',
    'weight loss', 'high blood pressure', 'muscle weakness', 'coordination problems',
    'heartburn', 'regurgitation', 'spitting up', 'irritability', 'popping sensation',
    'instability', 'blackheads', 'whiteheads', 'pimples', 'nodules', 'cysts',
    'painful lumps', 'abscesses', 'short stature', 'large head', 'bowed legs',
    'bone pain', 'petechiae', 'lymph nodes', 'fainting', 'heart palpitations',
    'skin lesions', 'inflammation', 'scarring', 'hormone changes', 'tumor',
    'aneurysm', 'infection'
}

# Non-medical phrases to filter out
NON_MEDICAL_PHRASES = [
    'see a doctor', 'call', 'emergency', 'contact', 'mayo clinic', 'cdc', 'webmd',
    'medlineplus', 'treatment', 'diagnosis', 'learn more', 'overview', 'prevention',
    'visit', 'website', 'test', 'history', 'family', 'risk', 'cause', 'condition',
    'manage', 'lifestyle', 'medicine', 'surgery', 'procedure', 'device', 'transplant',
    'symptom checker', 'play and learn', 'research', 'clinical trials', 'journal articles',
    'resources', 'reference desk', 'find an expert', 'patient handouts', 'summary',
    'care tips', 'self-care tips', 'say goodbye', 'add these products', 'it is not serious',
    'probably play a role', 'often blamed', 'how long ago', 'are they painful',
    'description', 'particularly in affected', 'usually in a person', 'at first',
    'if it grows large', 'it is important for them', 'or if they are men',
    'affecting about two', 'employees with', 'how to help someone', 'the goal of',
    'what are the symptoms', 'how to get rid', 'search results', 'click here',
    'your height and weight', 'help someone', 'methanol or ethylene glycol',
    'also inspanish', 'weight loss and alcohol', 'aiming for a healthy weight'
]

# Non-medical conditions to exclude
NON_MEDICAL_CONDITIONS = [
    'privacy policy', 'cookie policy', 'editorial policy', 'advertising policy',
    'terms of use', 'webmd app', 'corporate', 'site map', 'advertise with us',
    'see additional information', 'health topics', 'list of all topics', 'all',
    'health a-z news', 'health a-z reference', 'health a-z slideshows',
    'health a-z quizzes', 'health a-z videos', 'xyz', 'diagnostic tests',
    'transplantation and donation', 'socialfamily issues', 'subscribe to rss',
    'annual physical exam what to expect', 'cbd oil', 'keto diet',
    'diabetes warning signs', 'breast cancer screening', 'psoriatic arthritis symptoms',
    'types of crohns disease', 'food and nutrition', 'personal health issues',
    'wellness and lifestyle', 'blood heart and circulation', 'bones joints and muscles',
    'brain and nerves', 'digestive system', 'ear nose and throat', 'endocrine system',
    'eyes and vision', 'immune system', 'kidneys and urinary system', 'lungs and breathing',
    'mouth and teeth', 'skin hair and nails', 'female reproductive system', 'infections',
    'injuries and wounds', 'mental health and behavior', 'metabolic problems',
    'poisoning toxicology environmental health', 'pregnancy and reproduction',
    'substance use and disorders', 'drug therapy', 'surgery and rehabilitation',
    'symptoms', 'children and teenagers', 'men', 'older adults', 'population groups',
    'women', 'allergies', 'food allergy', 'latex allergy', 'peanut allergy'
]

# Minimum number of symptoms required to retain a condition
MIN_SYMPTOMS = 3

# Initialize NER pipeline
ner_pipeline = None
try:
    logger.info("Loading NER model (dslim/bert-base-NER)...")
    ner_pipeline = pipeline(
        "ner",
        model="dslim/bert-base-NER",
        tokenizer="dslim/bert-base-NER",
        device=0 if torch.cuda.is_available() else -1
    )
    # Test the model with a sample term
    test_term = "confusion"
    test_result = ner_pipeline(test_term)
    logger.info("NER test result for '%s': %s", test_term, test_result)
    logger.info("NER model loaded successfully")
except Exception as e:
    logger.error("Failed to load NER model: %s", str(e))
    logger.error("Ensure 'transformers' and 'torch' are installed, internet is available, and model 'dslim/bert-base-NER' is accessible.")
    ner_pipeline = None

class MongoDBHandler:
    def __init__(self, mongo_uri: str, db_name: str, collection_name: str):
        try:
            self.client = MongoClient(mongo_uri)
            self.client.admin.command('ping')
            logger.info("Connected to MongoDB at %s", mongo_uri)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
        except ConnectionFailure as e:
            logger.error("MongoDB connection failed: %s", str(e))
            raise

    def backup_collection(self):
        try:
            backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = Path('backups') / f"backup_{backup_time}.json"
            backup_path.parent.mkdir(exist_ok=True)
            with backup_path.open('w') as f:
                cursor = self.collection.find({})
                f.write('[\n')
                first = True
                for doc in cursor:
                    if not first:
                        f.write(',\n')
                    json.dump(dict(doc, _id=str(doc['_id'])), f)
                    first = False
                f.write('\n]')
            logger.info("Backed up %s to %s", self.collection.name, backup_path)
        except Exception as e:
            logger.error("Failed to backup %s: %s", self.collection.name, str(e))

    def delete_collection(self):
        try:
            self.collection.drop()
            logger.info("Dropped collection %s", self.collection.name)
        except PyMongoError as e:
            logger.error("Failed to drop collection %s: %s", self.collection.name, str(e))
            raise

    def insert_data(self, data: List[Dict]):
        try:
            if data:
                self.collection.insert_many(data)
                logger.info("Inserted %d documents into %s", len(data), self.collection.name)
                self.collection.create_index("condition", unique=True)
                self.collection.create_index([("required", 1)])
                logger.info("Created indexes on condition and required fields")
            else:
                logger.warning("No data to insert into %s", self.collection.name)
        except PyMongoError as e:
            logger.error("Failed to insert data into %s: %s", self.collection.name, str(e))
            raise

    def close(self):
        self.client.close()
        logger.info("Closed MongoDB connection")

def load_scraped_data(file_path: str) -> List[Dict]:
    try:
        with Path(file_path).open('r') as f:
            data = json.load(f)
        logger.info("Loaded %d conditions from %s", len(data), file_path)
        return data
    except Exception as e:
        logger.error("Failed to load data from %s: %s", file_path, str(e))
        return []

def is_medical_term_ner(term: str) -> bool:
    if not ner_pipeline:
        logger.warning("NER model not available. Falling back to rule-based check.")
        return is_medical_term_rule_based(term)
    
    try:
        entities = ner_pipeline(term)
        if entities:
            logger.debug("NER result for '%s': %s", term, entities)
        else:
            logger.debug("No entities found for '%s'", term)
        # Accept any entity (e.g., B-SYMPTOM, B-DISEASE, B-PER, etc.) as medical
        return len(entities) > 0 or is_medical_term_rule_based(term)
    except Exception as e:
        logger.warning("NER processing failed for term '%s': %s. Falling back to rule-based.", term, str(e))
        return is_medical_term_rule_based(term)

def is_medical_term_rule_based(term: str) -> bool:
    term_lower = term.lower().strip()
    if any(phrase in term_lower for phrase in NON_MEDICAL_PHRASES):
        logger.debug("Excluded '%s' due to non-medical phrase match", term)
        return False
    is_medical = any(keyword in term_lower for keyword in MEDICAL_TERMS) or re.search(r'\b(symptom|sign|clinical|manifestation)\b', term_lower)
    logger.debug("Rule-based result for '%s': %s", term, is_medical)
    return is_medical

def is_medical_condition(condition: str) -> bool:
    condition_lower = condition.lower().strip()
    is_medical = not any(non_condition in condition_lower for non_condition in NON_MEDICAL_CONDITIONS)
    logger.debug("Condition '%s' is medical: %s", condition, is_medical)
    return is_medical

def process_symptoms(symptoms: List[str]) -> List[str]:
    filtered = []
    for s in symptoms:
        s = s.strip()
        if len(s) <= 3 or re.match(r'^\s*$', s):
            logger.debug("Excluded '%s' due to length or empty", s)
            continue
        if is_medical_term_ner(s):
            filtered.append(s)
        else:
            logger.debug("Excluded '%s' by NER or rule-based check", s)
    filtered = list(dict.fromkeys(filtered))  # Remove duplicates
    return merge_similar_symptoms(filtered)

def merge_similar_symptoms(symptoms: List[str], threshold: int = 85) -> List[str]:
    merged = []
    used = set()
    for i, s1 in enumerate(symptoms):
        if i in used:
            continue
        merged.append(s1)
        used.add(i)
        for j, s2 in enumerate(symptoms[i+1:], i+1):
            if j not in used and fuzz.ratio(s1.lower(), s2.lower()) > threshold:
                used.add(j)
    return merged

def clean_data(data: List[Dict]) -> List[Dict]:
    cleaned_data = []
    for entry in data:
        try:
            condition = entry.get('condition', '')
            symptoms = entry.get('required', [])
            if not condition or not symptoms:
                logger.warning("Skipping entry with missing condition or symptoms: %s", entry.get('_id'))
                continue
            
            if not is_medical_condition(condition):
                logger.warning("Skipping non-medical condition: %s", condition)
                continue
            
            cleaned_symptoms = process_symptoms(symptoms)
            if len(cleaned_symptoms) < MIN_SYMPTOMS:
                logger.warning("Too few symptoms (%d) for %s", len(cleaned_symptoms), condition)
                continue
            
            entry_id = entry.get('_id')
            if isinstance(entry_id, dict) and '$oid' in entry_id:
                entry_id = entry_id['$oid']
            else:
                entry_id = str(ObjectId())
            
            cleaned_entry = {
                '_id': entry_id,
                'condition': condition,
                'required': cleaned_symptoms
            }
            cleaned_data.append(cleaned_entry)
            logger.info("Cleaned %s: retained %d symptoms", condition, len(cleaned_symptoms))
        except Exception as e:
            logger.error("Error cleaning entry for %s: %s", entry.get('condition', 'unknown'), str(e))
    return cleaned_data

def main():
    load_dotenv()
    mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
    db_name = os.getenv('DB_NAME', 'clinical_db')
    collection_name = 'kb_diagnosis_relevance'
    input_file = 'scraped_data.json'
    
    if not ner_pipeline:
        logger.error("NER model not loaded. Exiting to avoid rule-based fallback.")
        sys.exit(1)
    
    try:
        mongo_handler = MongoDBHandler(mongo_uri, db_name, collection_name)
    except ConnectionFailure as e:
        logger.error("Failed to connect to MongoDB: %s", str(e))
        return
    
    data = load_scraped_data(input_file)
    if not data:
        logger.error("No data to process. Exiting.")
        mongo_handler.close()
        return
    
    cleaned_data = clean_data(data)
    if not cleaned_data:
        logger.warning("No data retained after cleaning. Exiting.")
        mongo_handler.close()
        return
    
    try:
        mongo_handler.backup_collection()
        mongo_handler.delete_collection()
        mongo_handler.insert_data(cleaned_data)
        logger.info("Successfully updated %s with %d cleaned documents", collection_name, len(cleaned_data))
    except PyMongoError as e:
        logger.error("Failed to update MongoDB: %s", str(e))
    finally:
        mongo_handler.close()

if __name__ == "__main__":
    main()