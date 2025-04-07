from flask import Blueprint

bp = Blueprint('imaging', __name__, template_folder='templates')

# Import routes after defining bp
from . import routes