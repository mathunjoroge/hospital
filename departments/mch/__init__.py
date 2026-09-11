from flask import Blueprint

bp = Blueprint("mch", __name__, url_prefix="/mch")

from . import routes  # noqa: F401
