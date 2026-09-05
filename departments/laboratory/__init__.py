from flask import Blueprint

bp = Blueprint('laboratory', __name__, template_folder='templates')

from . import reagents, results, tests  # noqa: F401, E402
