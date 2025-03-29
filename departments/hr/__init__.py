from flask import Blueprint

bp = Blueprint('hr', __name__, template_folder='templates')

from . import routes