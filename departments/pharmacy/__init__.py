from flask import Blueprint

bp = Blueprint('pharmacy', __name__, template_folder='templates')

from . import inventory, dispensing, reports, ai_discovery

