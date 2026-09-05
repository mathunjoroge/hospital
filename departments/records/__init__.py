from flask import Blueprint

from departments.models.records import Patient  # noqa: F401
from extensions import db  # noqa: F401

# Create the blueprint
bp = Blueprint('records', __name__, template_folder='templates')

# Import routes to register them with the blueprint
from . import routes  # noqa: F401, E402
