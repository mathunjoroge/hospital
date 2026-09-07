from flask import Blueprint

bp = Blueprint("rcm", __name__, url_prefix="/rcm")

from . import routes  # noqa: E402, F401
