import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB Configuration
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
DB_NAME = os.getenv('DB_NAME', 'clinical_db')
KB_PREFIX = os.getenv('KB_PREFIX', 'kb_')

# UTS API Configuration
UTS_API_KEY = os.getenv('UTS_API_KEY')
UTS_BASE_URL = os.getenv('UTS_BASE_URL', 'https://uts-ws.nlm.nih.gov/rest')
UTS_AUTH_URL = os.getenv('UTS_AUTH_URL', 'https://utslogin.nlm.nih.gov/cas/v1/api-key')

# Model and Device Configuration
MODEL_NAME = os.getenv('MODEL_NAME', 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Processing Parameters
MAX_LENGTH = int(os.getenv('MAX_LENGTH', 512))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 8))
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', 768))
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', 0.9))
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.85))
MIN_CONFIDENCE_THRESHOLD = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', 0.6))

# PostgreSQL Configuration for hospital_umls
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'hospital_umls')
POSTGRES_USER= os.getenv('POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')
