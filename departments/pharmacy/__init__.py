from flask import Blueprint

bp = Blueprint('pharmacy', __name__, template_folder='templates')

from . import routes
