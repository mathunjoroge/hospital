from flask import Blueprint

bp = Blueprint("theatre", __name__, url_prefix="/theatre")

from . import routes  # noqa: E402, F401
