# departments/nlp/db_config.py
import logging
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import RealDictCursor
from psycopg2 import OperationalError, DatabaseError
from contextlib import contextmanager
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import atexit

from departments.nlp.logging_setup import get_logger
from departments.nlp.config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

logger = get_logger(__name__)

# Connection pool (initialized lazily)
pool = None

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    retry=retry_if_exception_type(OperationalError)
)
def initialize_pool():
    """Initialize PostgreSQL connection pool."""
    global pool
    if pool is None:
        try:
            pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                dbname=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                cursor_factory=RealDictCursor
            )
            logger.info("Initialized PostgreSQL connection pool")
        except OperationalError as e:
            logger.error(f"Failed to initialize pool: {e}")
            raise

def close_pool():
    """Close PostgreSQL connection pool."""
    global pool
    if pool is not None:
        try:
            pool.closeall()
            logger.info("Closed PostgreSQL connection pool")
        except Exception as e:
            logger.error(f"Error closing pool: {e}")
        finally:
            pool = None

atexit.register(close_pool)

@contextmanager
def get_postgres_connection():
    """Get a PostgreSQL connection from the pool."""
    if pool is None:
        initialize_pool()
    if pool is None:
        logger.error("Connection pool not initialized")
        yield None
        return
    conn = None
    try:
        conn = pool.getconn()
        # Use len(pool._pool) instead of .qsize() for compatibility with Python lists
        active_connections = pool.maxconn - len(pool._pool)
        logger.debug(f"Acquired connection, active connections: {active_connections}")
        yield conn
    except (OperationalError, DatabaseError) as e:
        logger.error(f"Failed to get connection: {e}")
        yield None
    finally:
        if conn:
            pool.putconn(conn)
            logger.debug("Returned connection to pool")