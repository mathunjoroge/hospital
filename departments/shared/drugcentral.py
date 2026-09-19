"""
departments/shared/drugcentral.py
──────────────────────────────────
DrugCentral PostgreSQL connection helper with pooling, keepalives, and TTL caching.

Safety requirements (P0 patient-safety fix):
- Every psycopg2.connect() call MUST include connect_timeout so a
  firewall, air-gap, or unreachable host fails fast instead of hanging
  indefinitely.
- A circuit breaker prevents hammering a repeatedly-unavailable host and
  ensures the local KNOWN_INTERACTIONS fallback always fires promptly.
- TCP Keepalives maintain active connections across WAN routers.
- Connection pooling reduces query overhead from 1,500ms+ down to ~150ms.

Environment variables:
  DRUGCENTRAL_DB            database name        (default: drugcentral)
  DRUGCENTRAL_USER          postgres user        (default: drugman)
  DRUGCENTRAL_PASSWORD      postgres password    (default: dosage)
  DRUGCENTRAL_HOST          hostname             (default: unmtid-dbs.net)
  DRUGCENTRAL_PORT          port                 (default: 5433)
  DRUGCENTRAL_CONNECT_TIMEOUT  connect timeout seconds (default: 3)
  DRUGCENTRAL_CB_THRESHOLD  failures before breaker opens (default: 3)
  DRUGCENTRAL_CB_COOLDOWN   cooldown seconds after breaker opens (default: 300)
"""

import logging
import os
import threading
import time
from typing import Any

import psycopg2
from psycopg2 import pool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection parameters & TCP Keepalives
# ---------------------------------------------------------------------------
DRUGCENTRAL_DB_PARAMS: dict[str, Any] = {
    "dbname": os.environ.get("DRUGCENTRAL_DB", "drugcentral"),
    "user": os.environ.get("DRUGCENTRAL_USER", "drugman"),
    "password": os.environ.get("DRUGCENTRAL_PASSWORD", "dosage"),
    "host": os.environ.get("DRUGCENTRAL_HOST", "unmtid-dbs.net"),
    "port": int(os.environ.get("DRUGCENTRAL_PORT", "5433")),
    "connect_timeout": int(os.environ.get("DRUGCENTRAL_CONNECT_TIMEOUT", "3")),
    # TCP Keepalive settings to prevent silent socket drop on WAN
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 5,
}

# Alias for backward compatibility
db_params = DRUGCENTRAL_DB_PARAMS

# ---------------------------------------------------------------------------
# Threaded Connection Pool
# ---------------------------------------------------------------------------
_pool_lock = threading.Lock()
_connection_pool: pool.ThreadedConnectionPool | None = None


def _get_pool() -> pool.ThreadedConnectionPool:
    """Lazily initialize and return the global connection pool."""
    global _connection_pool
    with _pool_lock:
        if _connection_pool is None or _connection_pool.closed:
            _connection_pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=5,
                **DRUGCENTRAL_DB_PARAMS,
            )
        return _connection_pool


class PooledConnectionProxy:
    """
    Transparent proxy for a pooled psycopg2 connection.
    Calling .close() or using context manager returns the connection to the pool.
    """

    def __init__(self, conn: Any, pool_obj: pool.ThreadedConnectionPool):
        self._conn = conn
        self._pool = pool_obj
        self._returned = False

    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        return self._conn.cursor(*args, **kwargs)

    def close(self) -> None:
        if not self._returned:
            self._returned = True
            try:
                if self._conn.closed:
                    self._pool.putconn(self._conn, close=True)
                else:
                    self._pool.putconn(self._conn)
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self) -> "PooledConnectionProxy":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_type is not None:
            try:
                self._conn.rollback()
            except Exception:  # noqa: BLE001
                pass
        self.close()
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


# ---------------------------------------------------------------------------
# In-Memory Query Cache (30 min TTL)
# ---------------------------------------------------------------------------
_QUERY_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL = float(os.environ.get("DRUGCENTRAL_CACHE_TTL", "1800"))


def get_cached_drug_query(cache_key: str) -> Any | None:
    """Retrieve cached query result if fresh."""
    with _CACHE_LOCK:
        if cache_key in _QUERY_CACHE:
            ts, val = _QUERY_CACHE[cache_key]
            if time.monotonic() - ts < _CACHE_TTL:
                return val
            del _QUERY_CACHE[cache_key]
    return None


def set_cached_drug_query(cache_key: str, val: Any) -> None:
    """Store query result in in-memory TTL cache."""
    with _CACHE_LOCK:
        _QUERY_CACHE[cache_key] = (time.monotonic(), val)


# ---------------------------------------------------------------------------
# Circuit breaker (module-level, thread-safe)
# ---------------------------------------------------------------------------
_CB_THRESHOLD: int = int(os.environ.get("DRUGCENTRAL_CB_THRESHOLD", "3"))
_CB_COOLDOWN: float = float(os.environ.get("DRUGCENTRAL_CB_COOLDOWN", "300"))

_cb_lock = threading.Lock()
_cb_failure_count: int = 0
_cb_open_since: float = 0.0  # epoch seconds when breaker last opened; 0 = closed


def _cb_is_open() -> bool:
    """Return True if the circuit breaker is currently open (blocking calls)."""
    with _cb_lock:
        if _cb_open_since == 0.0:
            return False
        if time.monotonic() - _cb_open_since >= _CB_COOLDOWN:  # noqa: SIM103
            return False  # cooldown elapsed; let probe through
        return True


def _cb_record_success() -> None:
    global _cb_failure_count, _cb_open_since
    with _cb_lock:
        if _cb_failure_count > 0 or _cb_open_since != 0.0:
            logger.info(
                "drugcentral.circuit_breaker.closed: DrugCentral connection "
                "restored after %d consecutive failures.",
                _cb_failure_count,
            )
        _cb_failure_count = 0
        _cb_open_since = 0.0


def _cb_record_failure(reason: str) -> None:
    global _cb_failure_count, _cb_open_since
    with _cb_lock:
        _cb_failure_count += 1
        logger.warning(
            "drugcentral.connection_failure count=%d reason=%s",
            _cb_failure_count,
            reason,
        )
        if _cb_failure_count >= _CB_THRESHOLD and _cb_open_since == 0.0:
            _cb_open_since = time.monotonic()
            logger.warning(
                "drugcentral.circuit_breaker.open: %d consecutive failures. "
                "Skipping live lookup for %.0f seconds.",
                _cb_failure_count,
                _CB_COOLDOWN,
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class DrugCentralUnavailable(Exception):
    """Raised when the circuit breaker is open or connection cannot be made."""


def get_drugcentral_connection():
    """
    Return a pooled connection to DrugCentral.

    Raises DrugCentralUnavailable if:
    - The circuit breaker is open (too many recent failures), or
    - The connection attempt itself fails or times out.

    Callers should catch DrugCentralUnavailable and fall through to the
    local KNOWN_INTERACTIONS matrix immediately.
    """
    if _cb_is_open():
        raise DrugCentralUnavailable(
            "Circuit breaker open: DrugCentral lookups suspended during cooldown."
        )

    t0 = time.monotonic()
    try:
        pool_obj = _get_pool()
        conn = pool_obj.getconn()

        # Ping connection to check if it's still healthy
        if conn.closed:
            pool_obj.putconn(conn, close=True)
            conn = psycopg2.connect(**DRUGCENTRAL_DB_PARAMS)
        else:
            try:
                # Test query to ensure socket is alive
                with conn.cursor() as check_cur:
                    check_cur.execute("SELECT 1;")
            except Exception:  # noqa: BLE001
                pool_obj.putconn(conn, close=True)
                conn = psycopg2.connect(**DRUGCENTRAL_DB_PARAMS)

        _cb_record_success()
        latency_ms = (time.monotonic() - t0) * 1000
        logger.debug("drugcentral.connected latency_ms=%.1f", latency_ms)
        return PooledConnectionProxy(conn, pool_obj)
    except (psycopg2.OperationalError, psycopg2.Error) as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        reason = f"{type(exc).__name__}: {exc}"
        _cb_record_failure(reason)
        raise DrugCentralUnavailable(reason) from exc
    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        reason = f"unexpected {type(exc).__name__}: {exc}"
        _cb_record_failure(reason)
        raise DrugCentralUnavailable(reason) from exc
