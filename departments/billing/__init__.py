from flask import Blueprint



# Create the blueprint
bp = Blueprint('billing', __name__, template_folder='templates')

# Import routes to register them with the blueprint
from . import routes
