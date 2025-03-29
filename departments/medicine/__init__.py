from flask import Blueprint

bp = Blueprint('medicine', __name__, template_folder='templates')

from . import routes