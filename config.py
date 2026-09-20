# config.py
import logging as _cfg_log
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


_cfg_log.getLogger(__name__).debug("POSTGRES_DB=%s", os.getenv("POSTGRES_DB"))
_cfg_log.getLogger(__name__).debug(
    "DATABASE_URI=%s", os.getenv("SQLALCHEMY_DATABASE_URI")
)

# PostgreSQL Configuration for hospital_umls
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "hospital_umls")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres" if os.getenv("FLASK_ENV") == "testing" else "")
LOCAL_TERMINOLOGY_PATH = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# SQLAlchemy Database URI Configuration
# Defaults to PostgreSQL. SQLite is only used when FLASK_ENV=testing.
_default_db_uri = (
    "sqlite:///hims.db"
    if os.getenv("FLASK_ENV") == "testing"
    else f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/hospital_core"
)
SQLALCHEMY_DATABASE_URI = os.getenv("SQLALCHEMY_DATABASE_URI", _default_db_uri)
SQLALCHEMY_TRACK_MODIFICATIONS = False

# Database Connection Pooling for high-traffic hospital environments (PostgreSQL)
if os.getenv("FLASK_ENV") == "testing" or SQLALCHEMY_DATABASE_URI.startswith("sqlite"):
    SQLALCHEMY_ENGINE_OPTIONS = {}
else:
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_size": 20,
        "max_overflow": 30,
        "pool_timeout": 30,
        "pool_recycle": 1800,
    }

# Cache Directory
CACHE_DIR = os.getenv("CACHE_DIR", "data_cache")

# Model and Device Configuration
MODEL_NAME = os.getenv(
    "MODEL_NAME", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
)
DEVICE = os.getenv("DEVICE", "cpu")

# Processing Parameters
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 512))  # noqa: PLW1508
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))  # noqa: PLW1508
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 768))  # noqa: PLW1508
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.9))  # noqa: PLW1508
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.85))  # noqa: PLW1508
MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", 0.6))  # noqa: PLW1508


class Config:
    # Billing sync configuration
    BILLING_SYNC_ENABLED = (
        os.environ.get("BILLING_SYNC_ENABLED", "true").lower() == "true"
    )

    SECRET_KEY = os.getenv("SECRET_KEY")
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
    SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI
    SQLALCHEMY_ENGINE_OPTIONS = SQLALCHEMY_ENGINE_OPTIONS

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
    # Cookie security settings
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = os.getenv("FLASK_ENV") == "production"
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SECURE = os.getenv("FLASK_ENV") == "production"

    ENABLE_TELEMEDICINE = os.getenv("ENABLE_TELEMEDICINE", "false").lower() == "true"

    # Enterprise SSO (OIDC / LDAP). Default OFF: the blueprint mounts
    # unauthenticated login endpoints, so it must be explicitly opted into
    # and only after OIDC_/LDAP_ credentials are actually configured.
    ENABLE_SSO = os.getenv("ENABLE_SSO", "false").lower() == "true"

    # WHO ICD-10 API (DECISIONS_PENDING #4 — resolved 2026-09-13)
    WHO_ICD_CLIENT_ID = os.getenv("WHO_ICD_CLIENT_ID", "")
    WHO_ICD_CLIENT_SECRET = os.getenv("WHO_ICD_CLIENT_SECRET", "")
    WHO_ICD_API_RELEASE = os.getenv("WHO_ICD_API_RELEASE", "2019")

    # UMLS (SNOMED CT & LOINC) API (DECISIONS_PENDING #22 & #25 — resolved 2026-09-13)
    UMLS_API_KEY = os.getenv("UMLS_API_KEY", "")
    UMLS_USERNAME = os.getenv("UMLS_USERNAME", "")
