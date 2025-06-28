import psycopg2
from departments.nlp.config import LOCAL_TERMINOLOGY_PATH
from departments.nlp.logging_setup import get_logger

logger = get_logger(__name__)

def get_postgres_connection():
    try:
        conn = psycopg2.connect(LOCAL_TERMINOLOGY_PATH)
        logger.info("Successfully connected to PostgreSQL")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}", exc_info=True)
        raise