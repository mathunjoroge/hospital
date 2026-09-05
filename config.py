# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


print("Loaded POSTGRES_DB from .env:", os.getenv('POSTGRES_DB'))
print("Loaded SQLALCHEMY_DATABASE_URI from .env:", os.getenv('SQLALCHEMY_DATABASE_URI'))
print("Loaded CACHE_DIR from .env:", os.getenv('CACHE_DIR'))

# PostgreSQL Configuration for hospital_umls
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'hospital_umls')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')
LOCAL_TERMINOLOGY_PATH = "postgresql://user:password@localhost:5432/hospital_umls"

# SQLAlchemy Database URI Configuration (for SQLite hims.db)
SQLALCHEMY_DATABASE_URI = os.getenv('SQLALCHEMY_DATABASE_URI', 'sqlite:///hims.db')
SQLALCHEMY_TRACK_MODIFICATIONS = False

# Cache Directory
CACHE_DIR = os.getenv('CACHE_DIR', 'data_cache')

# Model and Device Configuration
MODEL_NAME = os.getenv('MODEL_NAME', 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
DEVICE = os.getenv('DEVICE', 'cpu')

# Processing Parameters
MAX_LENGTH = int(os.getenv('MAX_LENGTH', 512))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 8))
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', 768))
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', 0.9))
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.85))
MIN_CONFIDENCE_THRESHOLD = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', 0.6))

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-default-secret-key')
    SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI
    SQLALCHEMY_TRACK_MODIFICATIONS = SQLALCHEMY_TRACK_MODIFICATIONS
    POSTGRES_HOST = POSTGRES_HOST
    POSTGRES_PORT = POSTGRES_PORT
    POSTGRES_DB = POSTGRES_DB
    POSTGRES_USER = POSTGRES_USER
    POSTGRES_PASSWORD = POSTGRES_PASSWORD
    CACHE_DIR = CACHE_DIR
    MODEL_NAME = MODEL_NAME
    DEVICE = DEVICE
    MAX_LENGTH = MAX_LENGTH
    BATCH_SIZE = BATCH_SIZE
    EMBEDDING_DIM = EMBEDDING_DIM
    SIMILARITY_THRESHOLD = SIMILARITY_THRESHOLD
    CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD
    MIN_CONFIDENCE_THRESHOLD = MIN_CONFIDENCE_THRESHOLD
