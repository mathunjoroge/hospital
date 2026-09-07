from flask import Blueprint

bp = Blueprint("mch", __name__, url_prefix="/mch")

from . import routes  # noqa: E402, F401
