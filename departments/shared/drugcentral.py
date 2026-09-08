"""
departments/shared/drugcentral.py
──────────────────────────────────
DrugCentral PostgreSQL connection helper.

Safety requirements (P0 patient-safety fix):
- Every psycopg2.connect() call MUST include connect_timeout so a
  firewall, air-gap, or unreachable host fails fast instead of hanging
  indefinitely.
- A circuit breaker prevents hammering a repeatedly-unavailable host and
  ensures the local KNOWN_INTERACTIONS fallback always fires promptly.

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
from typing import Any, Dict

import psycopg2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection parameters
# ---------------------------------------------------------------------------
DRUGCENTRAL_DB_PARAMS: Dict[str, Any] = {
    "dbname": os.environ.get("DRUGCENTRAL_DB", "drugcentral"),
    "user": os.environ.get("DRUGCENTRAL_USER", "drugman"),
    "password": os.environ.get("DRUGCENTRAL_PASSWORD", "dosage"),
    "host": os.environ.get("DRUGCENTRAL_HOST", "unmtid-dbs.net"),
    "port": int(os.environ.get("DRUGCENTRAL_PORT", "5433")),
    # connect_timeout is the critical safety parameter: without it a TCP
    # connection attempt to an unreachable host can block for minutes
    # (kernel default ~2 min), stalling the prescribing workflow entirely.
    "connect_timeout": int(os.environ.get("DRUGCENTRAL_CONNECT_TIMEOUT", "3")),
}

# Alias for backward compatibility
db_params = DRUGCENTRAL_DB_PARAMS

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
        if time.monotonic() - _cb_open_since >= _CB_COOLDOWN:
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
    Return a psycopg2 connection to DrugCentral.

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
        conn = psycopg2.connect(**DRUGCENTRAL_DB_PARAMS)
        _cb_record_success()
        latency_ms = (time.monotonic() - t0) * 1000
        logger.debug("drugcentral.connected latency_ms=%.1f", latency_ms)
        return conn
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
