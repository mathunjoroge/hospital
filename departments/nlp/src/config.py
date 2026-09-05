import logging
import multiprocessing
import os

import pytz
from dotenv import load_dotenv

from departments.nlp.resources.priority_symptoms import PRIORITY_SYMPTOMS

logger = logging.getLogger("HIMS-NLP")

# Load environment variables
load_dotenv()

class AppConfig:
    """Application configuration management."""
    _config = None

    @classmethod
    def load(cls) -> dict:
        """Load and validate configuration from environment variables."""
        if cls._config is None:
            cls._config = {
                "DEFAULT_DEPARTMENT": os.getenv("DEFAULT_DEPARTMENT", "emergency"),
                "PRIORITY_SYMPTOMS": PRIORITY_SYMPTOMS,
                "UMLS_THRESHOLD": float(os.getenv("UMLS_THRESHOLD", 0.7)),
                "SQLITE_DB_PATH": os.getenv("SQLITE_DB_PATH", "/home/mathu/projects/hospital/instance/hims.db"),
                "API_HOST": os.getenv("API_HOST", "0.0.0.0"),  # nosec B104
                "API_PORT": int(os.getenv("API_PORT", 8000)),
                "BATCH_SIZE": int(os.getenv("BATCH_SIZE", 50)),
                "UMLS_DB_URL": os.getenv("UMLS_DB_URL", "postgresql://postgres:postgres@localhost:5432/hospital_umls"),
                "TRUSTED_SOURCES": os.getenv("TRUSTED_SOURCES", "MSH,SNOMEDCT_US,ICD10CM,ICD9CM,LNC").split(','),
                "UMLS_LANGUAGE": os.getenv("UMLS_LANGUAGE", "ENG"),
                "MAX_WORKERS": int(os.getenv("MAX_WORKERS", multiprocessing.cpu_count())),
                "RATE_LIMIT": os.getenv("RATE_LIMIT", "10/minute"),
                "SIMILARITY_THRESHOLD": float(os.getenv("SIMILARITY_THRESHOLD", 1.0)),
                "CANCER_CONFIDENCE_THRESHOLD": float(os.getenv("CANCER_CONFIDENCE_THRESHOLD", 0.3)),
                "AMR_IPC_CONFIDENCE_THRESHOLD": float(os.getenv("AMR_IPC_CONFIDENCE_THRESHOLD", 0.1))
            }
        return cls._config

def get_config() -> dict:
    """Get the application configuration."""
    return AppConfig.load()

# Central timezone configuration
TIME_ZONE = pytz.timezone('Africa/Nairobi')

# Bootstrap styling constants for consistent UI
BOOTSTRAP_CLASSES = {
    "container": "container mt-4",
    "card": "card shadow-sm mb-4",
    "card_header": "card-header bg-primary text-white",
    "card_body": "card-body",  # Added to match usage in utils.py
    "alert_success": "alert alert-success",  # Added to match usage in utils.py
    "alert_danger": "alert alert-danger",  # Added to match usage in utils.py
    "badge_primary": "badge bg-primary ms-2",
    "badge_priority": "badge bg-danger ms-2",
    "badge_keyword": "badge bg-info text-dark",
    "badge_cui": "badge bg-secondary",
    "badge_warning": "badge bg-warning text-dark",
    "section_heading": "border-bottom pb-2 mb-3"  # Updated to match utils.py
}
