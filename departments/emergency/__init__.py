"""
departments/emergency/__init__.py
───────────────────────────────────
Phase D — Emergency blueprint
"""

from flask import Blueprint

bp = Blueprint('emergency', __name__)

from . import routes  # noqa: E402, F401
