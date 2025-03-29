from flask import Blueprint
from extensions import db  # Import db from extensions.py
from departments.models.records import Patient  # Import the Patient model

# Create the blueprint
bp = Blueprint('records', __name__, template_folder='templates')

# Import routes to register them with the blueprint
from . import routes
