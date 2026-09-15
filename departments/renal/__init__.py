"""
departments/renal/__init__.py
──────────────────────────────
Renal / Dialysis Unit department package.
Covers Haemodialysis (HD) and Continuous Renal Replacement Therapy (CRRT).
Peritoneal Dialysis (PD) deferred — see DECISIONS_PENDING.md §23 decision #1.
"""

from flask import Blueprint

bp = Blueprint("renal", __name__, url_prefix="/renal")
renal_bp = bp

from . import routes  # noqa: F401
