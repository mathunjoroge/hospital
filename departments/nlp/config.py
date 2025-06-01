# ~/projects/hospital/departments/nlp/config.py
import os
import torch

# MongoDB Configuration
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
DB_NAME = os.getenv('DB_NAME', 'clinical_db')
KB_PREFIX = os.getenv('KB_PREFIX', 'kb_')
SYMPTOMS_COLLECTION = os.getenv('SYMPTOMS_COLLECTION', 'symptoms')

# Local Terminology Configuration
LOCAL_TERMINOLOGY = os.getenv('LOCAL_TERMINOLOGY', 'umls_local')
LOCAL_TERMINOLOGY_PATH = os.getenv('LOCAL_TERMINOLOGY_PATH', 'postgresql://postgres:postgres@localhost:5432/hospital_umls')
LOCAL_TERMINOLOGY_TYPE = os.getenv('LOCAL_TERMINOLOGY_TYPE', 'postgres')
FALLBACK_CFG = os.getenv('FALLBACK_TERMS_PATH', './data/fallback_terms.json')
UMLS_KB_PATH = os.getenv('UMLS_KB_PATH', '/path/to/umls_2023aa')

# Model and Processing Parameters
MODEL_NAME = os.getenv('MODEL_NAME', 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
MAX_LENGTH = int(os.getenv('MAX_LENGTH', '512'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '8'))
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '768'))
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.9'))
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.85'))
MIN_CONFIDENCE_THRESHOLD = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.6'))
DEVICE = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

# PostgreSQL Configuration
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'hospital_umls')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')

# SQLite Configuration
SQLALCHEMY_DATABASE_URI = os.getenv('SQLALCHEMY_DATABASE_URI', 'sqlite:///hims.db')

# Cache and Backup Directories
CACHE_DIR = os.getenv('CACHE_DIR', 'data_cache')
BACKUP_DIR = os.getenv('BACKUP_DIR', 'backups')