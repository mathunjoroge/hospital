"""
departments/telemedicine/__init__.py
──────────────────────────────────────
Phase E — Telemedicine & Virtual Consultation Blueprint
"""

from flask import Blueprint

bp = Blueprint('telemedicine', __name__, template_folder='templates')

from . import routes  # noqa: F401, E402
