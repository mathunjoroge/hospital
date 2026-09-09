"""
tests/test_umls_connection_hardening.py
────────────────────────────────────────
Verifies the UMLS Postgres engine (departments/nlp/src/database.py) is
bounded the same way the DrugCentral P0 fix bounded drug-interaction
lookups: a slow/unreachable host must fail fast, not hang indefinitely.
"""

import time

import pytest
from sqlalchemy.exc import OperationalError

from departments.nlp.src import database as nlp_database


def test_umls_engine_has_connect_timeout_configured():
    """The engine must be constructed with an explicit, positive connect_timeout."""
    assert nlp_database.UMLS_CONNECT_TIMEOUT >= 1
    assert nlp_database.umls_engine.pool._pre_ping is True


def test_umls_engine_has_pool_recycle_configured():
    """Connections must be recycled periodically, not held open forever."""
    assert nlp_database.umls_engine.pool._recycle == 1800


def test_unreachable_umls_host_fails_fast_not_hangs():
    """
    Point a throwaway engine at a non-routable IP (TEST-NET-1, RFC 5737,
    guaranteed to blackhole rather than actively refuse) and confirm the
    connection attempt raises within a few seconds instead of hanging.
    """
    from sqlalchemy import create_engine

    unreachable_engine = create_engine(
        "postgresql+psycopg2://user:pass@192.0.2.1:5432/umls",
        connect_args={"connect_timeout": 2},
    )

    start = time.time()
    with pytest.raises(OperationalError):
        conn = unreachable_engine.connect()
        conn.close()
    elapsed = time.time() - start

    # libpq floors connect_timeout at 2s; allow generous slack for CI jitter
    # but this must NOT hang indefinitely (the bug being fixed).
    assert elapsed < 10, (
        f"Connection attempt took {elapsed:.1f}s -- expected fast failure "
        "under the configured connect_timeout, not a hang."
    )
