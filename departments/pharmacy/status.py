"""
departments/pharmacy/status.py
──────────────────────────────
Canonical DispensedDrug.status values and shared query helpers.

DispensedDrug.status has historically been written inconsistently
("0", 1, "Pending", "COMPLETED", "DISPENSED", "VOIDED"). Rather than
attempting a risky data migration across every writer, all READ sites
must use these helpers so that reporting is consistent:

  - Anything marked VOIDED is excluded from consumption, billing and
    analytics aggregations.
  - Everything else (any legacy value) counts as a dispensed unit.
"""

from sqlalchemy import or_

VOIDED = "VOIDED"


def not_voided(column):
    """SQLAlchemy filter: row is not voided (NULL-safe)."""
    return or_(column.is_(None), column != VOIDED)
