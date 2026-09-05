import os
import psycopg2
from typing import Dict, Any

DRUGCENTRAL_DB_PARAMS: Dict[str, Any] = {
    'dbname': os.environ.get('DRUGCENTRAL_DB', 'drugcentral'),
    'user': os.environ.get('DRUGCENTRAL_USER', 'drugman'),
    'password': os.environ.get('DRUGCENTRAL_PASSWORD', 'dosage'),
    'host': os.environ.get('DRUGCENTRAL_HOST', 'unmtid-dbs.net'),
    'port': int(os.environ.get('DRUGCENTRAL_PORT', '5433'))
}

# Alias for backward compatibility
db_params = DRUGCENTRAL_DB_PARAMS

def get_drugcentral_connection():
    """Returns a psycopg2 database connection to DrugCentral."""
    return psycopg2.connect(**DRUGCENTRAL_DB_PARAMS)
