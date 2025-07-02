import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
HIMS_CONFIG = {
    "SQLITE_DB_PATH": "/home/mathu/projects/hospital/instance/hims.db",
    "DB_VERSION": "1.0.0"
}

@contextmanager
def get_sqlite_connection():
    """Context manager for SQLite connections to the existing hims.db"""
    try:
        conn = sqlite3.connect(HIMS_CONFIG["SQLITE_DB_PATH"])
        conn.row_factory = sqlite3.Row
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        conn.close()

def initialize_database():
    """Initialize or update tables in the existing hims.db database for the HIMS Clinical NLP system"""
    try:
        with get_sqlite_connection() as conn:
            cursor = conn.cursor()

            # Create or verify tables with indexes, using IF NOT EXISTS to avoid overwriting
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS soap_notes (
                    id INTEGER PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    situation TEXT NOT NULL,
                    hpi TEXT NOT NULL,
                    aggravating_factors TEXT,
                    alleviating_factors TEXT,
                    medical_history TEXT,
                    medication_history TEXT,
                    assessment TEXT NOT NULL,
                    recommendation TEXT,
                    additional_notes TEXT,
                    symptoms TEXT,
                    ai_analysis TEXT,
                    ai_notes TEXT,
                    file_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS diseases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    cui TEXT NOT NULL,
                    description TEXT
                );
                CREATE TABLE IF NOT EXISTS symptoms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    cui TEXT,
                    description TEXT
                );
                CREATE TABLE IF NOT EXISTS disease_symptoms (
                    disease_id INTEGER,
                    symptom_id INTEGER,
                    PRIMARY KEY (disease_id, symptom_id),
                    FOREIGN KEY (disease_id) REFERENCES diseases(id) ON DELETE CASCADE,
                    FOREIGN KEY (symptom_id) REFERENCES symptoms(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS disease_management_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    disease_id INTEGER NOT NULL,
                    plan TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (disease_id) REFERENCES diseases(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    category TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS disease_keywords (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    disease_id INTEGER NOT NULL,
                    keyword TEXT NOT NULL,
                    cui TEXT,
                    FOREIGN KEY (disease_id) REFERENCES diseases(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_diseases_name ON diseases(name);
                CREATE INDEX IF NOT EXISTS idx_symptoms_name ON symptoms(name);
                CREATE INDEX IF NOT EXISTS idx_disease_keywords_keyword ON disease_keywords(keyword);
            """)
            logger.info("Ensured database tables and indexes exist in hims.db")

            # Check if seeding is necessary by checking if tables are empty
            cursor.execute("SELECT COUNT(*) FROM diseases")
            if cursor.fetchone()[0] == 0:
                # Seed diseases
                diseases = [
                    ("pneumonia", "C0032285", "A lung infection caused by bacteria, viruses, or fungi"),
                    ("myocardial_infarction", "C0027051", "Heart attack due to blocked coronary arteries"),
                    ("stroke", "C0038454", "Sudden interruption of blood flow to the brain"),
                    ("gastroenteritis", "C0017160", "Inflammation of the stomach and intestines"),
                    ("uti", "C0033578", "Urinary tract infection"),
                    ("asthma", "C0004096", "Chronic respiratory condition with airway inflammation"),
                    ("diabetes", "C0011849", "Chronic condition affecting blood sugar regulation"),
                    ("hypertension", "C0020538", "High blood pressure"),
                    ("musculoskeletal_back_pain", "C0026857", "Back pain due to musculoskeletal issues"),
                    ("malaria", "C0024530", "Parasitic infection caused by Plasmodium species"),
                    ("viral_hepatitis", "C0019158", "Liver inflammation caused by viral infection"),
                    ("dengue", "C0011311", "Viral infection transmitted by mosquitoes"),
                    ("typhoid", "C0041466", "Bacterial infection caused by Salmonella typhi")
                ]
                cursor.executemany("INSERT OR IGNORE INTO diseases (name, cui, description) VALUES (?, ?, ?)", diseases)
                logger.info("Seeded diseases table")
            else:
                logger.info("Diseases table already populated, skipping seeding")

            cursor.execute("SELECT COUNT(*) FROM symptoms")
            if cursor.fetchone()[0] == 0:
                # Seed symptoms
                symptoms = [
                    ("cough", "C0010200", "Expulsion of air from the lungs"),
                    ("fever", "C0015967", "Elevated body temperature"),
                    ("chest pain", "C0008031", "Pain in the chest area"),
                    ("shortness of breath", "C0013404", "Difficulty breathing"),
                    ("sweating", None, "Excessive perspiration"),
                    ("headache", "C0018681", "Pain in the head"),
                    ("weakness", None, "Reduced strength"),
                    ("speech difficulty", None, "Impaired speech"),
                    ("facial droop", None, "One-sided facial weakness"),
                    ("diarrhea", "C0011991", "Frequent loose stools"),
                    ("nausea", "C0027497", "Feeling of sickness"),
                    ("vomiting", "C0042963", "Forceful expulsion of stomach contents"),
                    ("abdominal pain", "C0000737", "Pain in the abdomen"),
                    ("dysuria", None, "Painful urination"),
                    ("frequency", None, "Frequent urination"),
                    ("urgency", None, "Urgent need to urinate"),
                    ("suprapubic pain", None, "Pain above the pubic bone"),
                    ("wheezing", None, "High-pitched breathing sound"),
                    ("thirst", None, "Excessive thirst"),
                    ("polyuria", None, "Excessive urination"),
                    ("fatigue", "C0015672", "Extreme tiredness"),
                    ("blurred vision", None, "Unclear vision"),
                    ("dizziness", "C0012833", "Lightheadedness"),
                    ("nosebleed", None, "Bleeding from the nose"),
                    ("back pain", "C0004604", "Pain in the back region"),
                    ("chills", "C0085593", "Sensation of cold with shivering"),
                    ("loss of appetite", "C023 Sophisticated Assistant: 4450", "Reduced desire to eat"),
                    ("jaundice", "C0022346", "Yellowing of skin or eyes")
                ]
                cursor.executemany("INSERT OR IGNORE INTO symptoms (name, cui, description) VALUES (?, ?, ?)", symptoms)
                logger.info("Seeded symptoms table")
            else:
                logger.info("Symptoms table already populated, skipping seeding")

            cursor.execute("SELECT COUNT(*) FROM disease_symptoms")
            if cursor.fetchone()[0] == 0:
                # Seed disease_symptoms
                disease_signatures = {
                    "pneumonia": ["cough", "fever", "chest pain"],
                    "myocardial_infarction": ["chest pain", "shortness of breath", "sweating"],
                    "stroke": ["headache", "weakness", "speech difficulty", "facial droop"],
                    "gastroenteritis": ["diarrhea", "nausea", "vomiting", "abdominal pain"],
                    "uti": ["dysuria", "frequency", "urgency", "suprapubic pain"],
                    "asthma": ["wheezing", "shortness of breath", "cough"],
                    "diabetes": ["thirst", "polyuria", "fatigue", "blurred vision"],
                    "hypertension": ["headache", "dizziness", "nosebleed"],
                    "musculoskeletal_back_pain": ["back pain"],
                    "malaria": ["fever", "chills", "headache", "nausea", "vomiting", "loss of appetite", "jaundice"],
                    "viral_hepatitis": ["jaundice", "nausea", "vomiting", "loss of appetite", "fatigue"],
                    "dengue": ["fever", "headache", "nausea", "vomiting", "fatigue"],
                    "typhoid": ["fever", "headache", "abdominal pain", "loss of appetite"]
                }
                for disease_name, symptom_list in disease_signatures.items():
                    cursor.execute("SELECT id FROM diseases WHERE name = ?", (disease_name,))
                    result = cursor.fetchone()
                    if not result:
                        logger.warning(f"Disease {disease_name} not found in diseases table")
                        continue
                    disease_id = result[0]
                    for symptom_name in symptom_list:
                        cursor.execute("SELECT id FROM symptoms WHERE name = ?", (symptom_name,))
                        result = cursor.fetchone()
                        if not result:
                            logger.warning(f"Symptom {symptom_name} not found in symptoms table")
                            continue
                        symptom_id = result[0]
                        cursor.execute(
                            "INSERT OR IGNORE INTO disease_symptoms (disease_id, symptom_id) VALUES (?, ?)",
                            (disease_id, symptom_id)
                        )
                logger.info("Seeded disease_symptoms table")
            else:
                logger.info("Disease_symptoms table already populated, skipping seeding")

            cursor.execute("SELECT COUNT(*) FROM disease_management_plans")
            if cursor.fetchone()[0] == 0:
                # Seed disease_management_plans
                management_plans = [
                    ("pneumonia", "Antibiotics (e.g., amoxicillin), oxygen therapy if needed, rest, and hydration"),
                    ("myocardial_infarction", "Aspirin, nitroglycerin, possible PCI, beta-blockers, and lifestyle changes"),
                    ("stroke", "Thrombolytics if ischemic, blood pressure management, and rehabilitation"),
                    ("gastroenteritis", "Oral rehydration therapy, antiemetics, and possible antibiotics"),
                    ("uti", "Antibiotics (e.g., nitrofurantoin), increased fluid intake, and urinary analgesics"),
                    ("asthma", "Inhaled corticosteroids, bronchodilators, and avoidance of triggers"),
                    ("diabetes", "Insulin or oral hypoglycemics, dietary management, and regular monitoring"),
                    ("hypertension", "Antihypertensive medications, lifestyle changes, and regular BP monitoring"),
                    ("musculoskeletal_back_pain", "Physical therapy, NSAIDs (e.g., ibuprofen), posture correction, and imaging if persistent"),
                    ("malaria", "Antimalarial drugs (e.g., artemisinin-based combination therapy), supportive care, and monitoring"),
                    ("viral_hepatitis", "Supportive care, antiviral therapy if indicated, and liver function monitoring"),
                    ("dengue", "Supportive care, hydration, and monitoring for complications"),
                    ("typhoid", "Antibiotics (e.g., ceftriaxone), hydration, and supportive care")
                ]
                for disease_name, plan in management_plans:
                    cursor.execute("SELECT id FROM diseases WHERE name = ?", (disease_name,))
                    result = cursor.fetchone()
                    if not result:
                        logger.warning(f"Disease {disease_name} not found for management plan")
                        continue
                    disease_id = result[0]
                    cursor.execute(
                        "INSERT OR IGNORE INTO disease_management_plans (disease_id, plan) VALUES (?, ?)",
                        (disease_id, plan)
                    )
                logger.info("Seeded disease_management_plans table")
            else:
                logger.info("Disease_management_plans table already populated, skipping seeding")

            cursor.execute("SELECT COUNT(*) FROM patterns")
            if cursor.fetchone()[0] == 0:
                # Seed patterns
                patterns = [
                    ("PAIN", r"\b(pain|ache|discomfort|tenderness)\b", "SYMPTOM"),
                    ("FEVER", r"\b(fever|pyrexia|hyperthermia)\b", "SYMPTOM"),
                    ("RESPIRATORY", r"\b(cough|dyspnea|shortness of breath|wheez)\b", "SYMPTOM"),
                    ("CARDIO", r"\b(chest pain|palpitation|tachycardia)\b", "SYMPTOM"),
                    ("GASTRO", r"\b(nausea|vomiting|diarrhea|constipation|abdominal pain)\b", "SYMPTOM"),
                    ("NEURO", r"\b(headache|dizziness|vertigo|confusion|seizure)\b", "SYMPTOM"),
                    ("BACK_PAIN", r"\b(back pain|backpain|lumbago)\b", "SYMPTOM"),
                    ("CHILLS", r"\b(chills|shivering)\b", "SYMPTOM"),
                    ("APPETITE_LOSS", r"\b(loss of appetite|anorexia)\b", "SYMPTOM"),
                    ("JAUNDICE", r"\b(jaundice|yellowing)\b", "SYMPTOM")
                ]
                cursor.executemany(
                    "INSERT OR IGNORE INTO patterns (label, pattern, category) VALUES (?, ?, ?)",
                    patterns
                )
                logger.info("Seeded patterns table")
            else:
                logger.info("Patterns table already populated, skipping seeding")

            cursor.execute("SELECT COUNT(*) FROM disease_keywords")
            if cursor.fetchone()[0] == 0:
                # Seed disease_keywords
                disease_keywords = [
                    ("malaria", "C0024530"),
                    ("pneumonia", "C0032285"),
                    ("meningitis", "C0025289"),
                    ("uti", "C0033578"),
                    ("urinary tract infection", "C0033578"),
                    ("influenza", "C0021400"),
                    ("tuberculosis", "C0041296"),
                    ("gastroenteritis", "C0017160"),
                    ("dengue", "C0011311"),
                    ("cholera", "C0008344"),
                    ("bronchitis", "C0006277"),
                    ("hepatitis", "C0019158"),
                    ("viral hepatitis", "C0019158"),
                    ("asthma", "C0004096"),
                    ("myocardial infarction", "C0027051"),
                    ("stroke", "C0038454"),
                    ("diabetes", "C0011849"),
                    ("hypertension", "C0020538"),
                    ("back pain", "C0004604"),
                    ("musculoskeletal back pain", "C0026857"),
                    ("typhoid", "C0041466")
                ]
                for keyword, cui in disease_keywords:
                    cursor.execute("SELECT id FROM diseases WHERE name = ? OR cui = ?", (keyword, cui))
                    result = cursor.fetchone()
                    if result:
                        disease_id = result[0]
                        cursor.execute(
                            "INSERT OR IGNORE INTO disease_keywords (disease_id, keyword, cui) VALUES (?, ?, ?)",
                            (disease_id, keyword, cui)
                        )
                logger.info("Seeded disease_keywords table")
            else:
                logger.info("Disease_keywords table already populated, skipping seeding")

            # Add or update version tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS db_version (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("SELECT version FROM db_version ORDER BY created_at DESC LIMIT 1")
            result = cursor.fetchone()
            if not result or result[0] != HIMS_CONFIG["DB_VERSION"]:
                cursor.execute(
                    "INSERT INTO db_version (version) VALUES (?)",
                    (HIMS_CONFIG["DB_VERSION"],)
                )
                logger.info(f"Updated database version to {HIMS_CONFIG['DB_VERSION']}")
            else:
                logger.info(f"Database version is already {HIMS_CONFIG['DB_VERSION']}")

            conn.commit()
            logger.info("Database initialization/update completed successfully in hims.db")

    except Exception as e:
        logger.error(f"Database initialization/update failed: {e}")
        raise

if __name__ == "__main__":
    initialize_database()