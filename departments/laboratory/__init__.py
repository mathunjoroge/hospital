from flask import Blueprint

bp = Blueprint('laboratory', __name__, template_folder='templates')

from . import routes