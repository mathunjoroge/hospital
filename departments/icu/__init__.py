"""
departments/icu/__init__.py
────────────────────────────
ICU / HDU department package.
"""

from flask import Blueprint

bp = Blueprint("icu", __name__, url_prefix="/icu")
icu_bp = bp

from . import routes  # noqa: F401
